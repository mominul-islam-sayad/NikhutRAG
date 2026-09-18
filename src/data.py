"""Loading, integrity checking, group splitting and label alignment.

Two things this module exists to prevent, both called out in CLAUDE.md:

1. Split leakage. Every context appears twice, once faithful and once
   hallucinated. Splitting rows at random puts the same context on both sides
   and the reported F1 stops meaning anything. Splits here are always grouped on
   the base id, and additionally stratified by domain so a small domain cannot
   vanish from a split.

2. Silent label misalignment. Answer subword labels are derived through
   ``word_ids()`` on the pre-split answer words, never reconstructed from
   character offsets, and any answer word that receives zero subwords is
   counted and reported rather than dropped quietly.

The CSV in data/ is a sample; the real dataset is built elsewhere. So the
schema is validated explicitly and the sequence length is measured rather than
assumed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as cfg

BENGALI_RE = re.compile(r"[ঀ-৿]")


# --------------------------------------------------------------------------
# Integrity reporting
# --------------------------------------------------------------------------


@dataclass
class IntegrityReport:
    """What the loader found wrong. Printed, never swallowed."""

    n_rows: int = 0
    n_faithful: int = 0
    n_hallucinated: int = 0
    n_groups: int = 0
    group_sizes: dict[int, int] = field(default_factory=dict)

    #: label == 1 but every token label is 0. Unlearnable: the example score is
    #: the max over answer-token probabilities, so supervision that says "no
    #: token is hallucinated" directly contradicts the row label.
    label_says_hallucinated_tokens_say_clean: list[str] = field(default_factory=list)
    #: label == 0 but some token is flagged. Equally contradictory.
    label_says_clean_tokens_say_hallucinated: list[str] = field(default_factory=list)
    #: Flagged words do not match hallucinated_span. Affects span-level metrics.
    span_token_mismatch: list[str] = field(default_factory=list)
    #: Whole answer flagged. Token-level localization is degenerate here.
    fully_flagged: list[str] = field(default_factory=list)
    #: No Bengali script in context or answer.
    non_bangla: list[str] = field(default_factory=list)
    #: Answer words that tokenize to zero subwords (labels would be lost).
    zero_subword_words: int = 0
    #: Examples whose answer alone overflows max_length.
    answer_overflow: list[str] = field(default_factory=list)
    #: Examples where context had to be truncated to fit.
    context_truncated: list[str] = field(default_factory=list)

    def summary(self) -> str:
        # Each rate is quoted against the population it can actually occur in.
        # Dividing a hallucinated-only defect by all rows halves it and makes
        # the dataset look twice as clean as it is.
        def pct(n: int, denom: int, unit: str) -> str:
            return f"{n} ({n / max(denom, 1):.1%} of {unit})"

        rows, faith, halluc = self.n_rows, self.n_faithful, self.n_hallucinated
        lines = [
            (
                f"rows={rows} (faithful={faith}, hallucinated={halluc})  "
                f"groups={self.n_groups}  group sizes={self.group_sizes}"
            ),
            (
                f"  label=1 but no token flagged : "
                f"{pct(len(self.label_says_hallucinated_tokens_say_clean), halluc, 'hallucinated')}"
                f"  <-- unlearnable"
            ),
            (
                f"  label=0 but token flagged    : "
                f"{pct(len(self.label_says_clean_tokens_say_hallucinated), faith, 'faithful')}"
            ),
            (
                f"  flagged words != span        : "
                f"{pct(len(self.span_token_mismatch), halluc, 'hallucinated')}"
            ),
            (
                f"  entire answer flagged        : "
                f"{pct(len(self.fully_flagged), halluc, 'hallucinated')}"
            ),
            f"  no Bengali script            : {pct(len(self.non_bangla), rows, 'all rows')}",
        ]
        if self.zero_subword_words:
            lines.append(f"  answer words -> 0 subwords   : {self.zero_subword_words}  <-- labels lost")
        if self.answer_overflow:
            lines.append(
                f"  answer overflows max_length  : "
                f"{pct(len(self.answer_overflow), rows, 'all rows')}  <-- labels lost"
            )
        if self.context_truncated:
            lines.append(
                f"  context truncated to fit     : {pct(len(self.context_truncated), rows, 'all rows')}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def _parse_token_labels(raw: object) -> list[tuple[str, int]]:
    if isinstance(raw, (list, tuple)):
        pairs = raw
    else:
        pairs = json.loads(str(raw))
    return [(str(w), int(l)) for w, l in pairs]


def load_dataframe(
    csv_path: Path | str | None = None,
    schema: cfg.ColumnSchema = cfg.COLUMNS,
    drop_non_bangla: bool | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, IntegrityReport]:
    """Read the CSV, validate it, and attach derived columns.

    Raises on problems that make training meaningless (missing columns,
    unparseable token labels, a token/word count mismatch, an id that does not
    carry the twin suffix). Merely suspicious things go in the report.
    """
    csv_path = Path(csv_path or cfg.CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)

    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns {missing}. "
            f"Found: {list(df.columns)}. If the dataset schema changed, update "
            f"ColumnSchema in src/config.py rather than editing data.py."
        )

    # --- token labels -----------------------------------------------------
    try:
        df["_tokens"] = df[schema.token_labels].map(_parse_token_labels)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse {schema.token_labels!r}: {exc}") from exc

    df["_words"] = df[schema.answer].astype(str).str.split()

    bad_len = df[df["_tokens"].map(len) != df["_words"].map(len)]
    if len(bad_len):
        raise ValueError(
            f"{len(bad_len)} rows where len(token_labels) != len(answer.split()); "
            f"label alignment is impossible. First: {bad_len.iloc[0][schema.id]}"
        )

    mismatched = [
        r[schema.id]
        for _, r in df.iterrows()
        if any(tw != aw for (tw, _), aw in zip(r["_tokens"], r["_words"]))
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} rows where token_labels words differ from "
            f"answer.split(). First: {mismatched[0]}"
        )

    # --- base id ----------------------------------------------------------
    df["base_id"] = df[schema.id].astype(str).str.replace(cfg.ID_SUFFIX_PATTERN, "", regex=True)
    unchanged = (df["base_id"] == df[schema.id].astype(str)).sum()
    if unchanged == len(df):
        raise ValueError(
            f"ID_SUFFIX_PATTERN {cfg.ID_SUFFIX_PATTERN!r} matched no id. Without a "
            f"working base id the group split cannot prevent leakage. Example id: "
            f"{df[schema.id].iloc[0]!r}"
        )

    df["_n_pos"] = df["_tokens"].map(lambda t: sum(l for _, l in t))
    df["_any_pos"] = (df["_n_pos"] > 0).astype(int)

    # --- report -----------------------------------------------------------
    rep = IntegrityReport(n_rows=len(df), n_groups=df["base_id"].nunique())
    rep.group_sizes = dict(pd.Series(df.groupby("base_id").size()).value_counts().sort_index())

    labels = df[schema.label].astype(int)
    rep.n_hallucinated = int((labels == 1).sum())
    rep.n_faithful = int((labels == 0).sum())
    rep.label_says_hallucinated_tokens_say_clean = df.loc[
        (labels == 1) & (df["_any_pos"] == 0), schema.id
    ].tolist()
    rep.label_says_clean_tokens_say_hallucinated = df.loc[
        (labels == 0) & (df["_any_pos"] == 1), schema.id
    ].tolist()
    rep.fully_flagged = df.loc[
        (df["_n_pos"] > 0) & (df["_n_pos"] == df["_tokens"].map(len)), schema.id
    ].tolist()

    if schema.hallucinated_span in df.columns:
        for _, r in df[labels == 1].iterrows():
            span = r[schema.hallucinated_span]
            if not isinstance(span, str) or not span.strip():
                continue
            flagged = " ".join(w for w, l in r["_tokens"] if l).strip()
            if flagged and flagged != span.strip():
                rep.span_token_mismatch.append(r[schema.id])

    has_bn = lambda s: bool(BENGALI_RE.search(str(s)))  # noqa: E731
    non_bangla_mask = ~df[schema.context].map(has_bn) | ~df[schema.answer].map(has_bn)
    rep.non_bangla = df.loc[non_bangla_mask, schema.id].tolist()

    drop_non_bangla = cfg.DROP_NON_BANGLA if drop_non_bangla is None else drop_non_bangla
    if drop_non_bangla:
        df = df[~non_bangla_mask].reset_index(drop=True)

    if verbose:
        print(f"[data] loaded {csv_path.name}")
        print(rep.summary())
        if rep.label_says_hallucinated_tokens_say_clean:
            print(
                f"[data] WARNING: {len(rep.label_says_hallucinated_tokens_say_clean)} rows "
                f"carry label=1 with no flagged token. They cannot be fit under "
                f"max-over-tokens scoring and will cap achievable recall."
            )
        print(f"[data] non-Bangla rows are {'DROPPED' if drop_non_bangla else 'KEPT'} (config.DROP_NON_BANGLA)")

    return df, rep


# --------------------------------------------------------------------------
# Grouped, domain-stratified split
# --------------------------------------------------------------------------


def _partition(n: int, fracs: tuple[float, float, float], rng: np.random.Generator) -> list[np.ndarray]:
    """Split ``range(n)`` (shuffled) into three parts, keeping small n usable."""
    idx = rng.permutation(n)
    if n == 0:
        return [idx[:0], idx[:0], idx[:0]]
    n_train = int(round(n * fracs[0]))
    n_val = int(round(n * fracs[1]))
    # With a handful of groups, rounding can starve val/test; guarantee one each.
    if n >= 3:
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))
    n_train = max(n_train, 0)
    return [idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]]


def group_split(
    df: pd.DataFrame,
    schema: cfg.ColumnSchema = cfg.COLUMNS,
    fracs: tuple[float, float, float] | None = None,
    seed: int = cfg.SPLIT_SEED,
    stratify_by_domain: bool | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Split on ``base_id`` so twin rows never separate.

    CLAUDE.md suggests GroupShuffleSplit or StratifiedGroupKFold. This does the
    grouping *and* the domain stratification in one pass by partitioning groups
    within each domain, which is what actually guarantees a 44-row domain
    survives into val and test.
    """
    fracs = fracs or (cfg.TRAIN_FRAC, cfg.VAL_FRAC, cfg.TEST_FRAC)
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1, got {fracs} = {sum(fracs)}")
    stratify = cfg.STRATIFY_BY_DOMAIN if stratify_by_domain is None else stratify_by_domain

    if "base_id" not in df.columns:
        raise ValueError("call load_dataframe() first; base_id is missing")

    domain_col = schema.domain if (stratify and schema.domain in df.columns) else None

    groups = df.groupby("base_id")
    if domain_col:
        gdom = groups[domain_col].agg(lambda s: s.iloc[0])
        impure = groups[domain_col].nunique()
        if (impure > 1).any():
            n = int((impure > 1).sum())
            raise ValueError(f"{n} base_ids span more than one domain; grouping is unsafe")
        strata = {d: gdom.index[gdom == d].to_numpy() for d in sorted(gdom.unique())}
    else:
        strata = {"_all": np.array(sorted(df["base_id"].unique()))}

    rng = np.random.default_rng(seed)
    buckets: list[list[str]] = [[], [], []]
    for _, gids in strata.items():
        parts = _partition(len(gids), fracs, rng)
        for b, part in zip(buckets, parts):
            b.extend(gids[part].tolist())

    names = ("train", "val", "test")
    out = {
        name: df[df["base_id"].isin(set(ids))].reset_index(drop=True)
        for name, ids in zip(names, buckets)
    }

    # Leakage assertion: base_id sets must be pairwise disjoint.
    sets = [set(out[n]["base_id"]) for n in names]
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = sets[i] & sets[j]
            if overlap:
                raise AssertionError(
                    f"LEAKAGE: {len(overlap)} base_ids in both {names[i]} and {names[j]}"
                )

    if verbose:
        print("\n[split] grouped on base_id" + (", stratified by domain" if domain_col else ""))
        for name in names:
            d = out[name]
            share = f"{len(d) / max(len(df), 1):.0%}"
            pos = d[schema.label].astype(int).mean() if len(d) else float("nan")
            print(f"  {name:<6} rows={len(d):<6} groups={d['base_id'].nunique():<6} ({share})  pos_rate={pos:.3f}")
        if domain_col:
            print("  domain coverage:")
            for name in names:
                counts = out[name][domain_col].value_counts().to_dict()
                print(f"    {name:<6} {counts}")
        print("  leakage check: PASSED (no base_id shared across splits)")

    return out


# --------------------------------------------------------------------------
# Tokenization and label alignment
# --------------------------------------------------------------------------


def _build_pair(question: str, context: str, sep: str) -> list[str]:
    """Sequence A: question [SEP] context, as whitespace words.

    ``is_split_into_words=True`` applies to both halves of a pair, so A is
    pre-split too. Its word boundaries are irrelevant -- every A token is
    labelled IGNORE_INDEX -- only the answer's alignment has to be exact.
    """
    return str(question).split() + [sep] + str(context).split()


def encode_example(
    question: str,
    context: str,
    answer_words: list[str],
    token_labels: list[int],
    tokenizer,
    max_length: int,
    label_all_subwords: bool = cfg.LABEL_ALL_SUBWORDS,
) -> tuple[dict, dict]:
    """Encode one example to ``[CLS] question [SEP] context [SEP] answer [SEP]``.

    Truncation is ``only_first``: the context is sacrificed before the answer.
    Truncating the answer would silently discard labels, which is worse than
    losing grounding evidence -- and either way the caller is told it happened.
    """
    a_words = _build_pair(question, context, tokenizer.sep_token or "[SEP]")
    enc = tokenizer(
        a_words,
        answer_words,
        is_split_into_words=True,
        truncation="only_first",
        max_length=max_length,
    )

    seq_ids = enc.sequence_ids(0)
    word_ids = enc.word_ids(0)

    labels = [cfg.IGNORE_INDEX] * len(enc["input_ids"])
    seen: set[int] = set()
    for pos, (sid, wid) in enumerate(zip(seq_ids, word_ids)):
        if sid != 1 or wid is None:  # not an answer token
            continue
        if wid >= len(token_labels):
            continue
        if label_all_subwords or wid not in seen:
            labels[pos] = int(token_labels[wid])
        seen.add(wid)

    enc_out = {k: v for k, v in enc.items()}
    enc_out["labels"] = labels

    info = {
        "n_tokens": len(enc["input_ids"]),
        "answer_words_covered": len(seen),
        "answer_words_total": len(answer_words),
        "overflowed": len(enc["input_ids"]) > max_length,
        "context_truncated": sum(1 for s in seq_ids if s == 0) < len(a_words),
    }
    return enc_out, info


def choose_max_length(
    df: pd.DataFrame,
    tokenizer,
    schema: cfg.ColumnSchema = cfg.COLUMNS,
    sample: int = 1000,
    verbose: bool = True,
) -> int:
    """Measure the data rather than assuming a length.

    Returns the configured int when MAX_LENGTH is pinned; otherwise covers
    AUTO_LENGTH_PERCENTILE of examples, rounded up and capped at the smaller of
    AUTO_LENGTH_CAP and the backbone's position limit.
    """
    ceiling = min(cfg.AUTO_LENGTH_CAP, cfg.backbone_spec(tokenizer.name_or_path).model_max_length)

    if isinstance(cfg.MAX_LENGTH, int):
        if cfg.MAX_LENGTH > ceiling:
            raise ValueError(f"MAX_LENGTH={cfg.MAX_LENGTH} exceeds backbone limit {ceiling}")
        return cfg.MAX_LENGTH

    probe = df if len(df) <= sample else df.sample(sample, random_state=cfg.SPLIT_SEED)
    sep = tokenizer.sep_token or "[SEP]"
    lengths = []
    for _, r in probe.iterrows():
        enc = tokenizer(
            _build_pair(r[schema.question], r[schema.context], sep),
            list(r["_words"]),
            is_split_into_words=True,
            truncation=False,
        )
        lengths.append(len(enc["input_ids"]))

    arr = np.array(lengths)
    target = int(np.ceil(np.quantile(arr, cfg.AUTO_LENGTH_PERCENTILE)))
    m = cfg.AUTO_LENGTH_MULTIPLE
    chosen = min(int(np.ceil(target / m) * m), ceiling)

    if verbose:
        print(
            f"\n[length] measured on {len(probe)} rows: median={int(np.median(arr))} "
            f"p95={int(np.quantile(arr, 0.95))} p99={int(np.quantile(arr, 0.99))} max={arr.max()}"
        )
        print(f"[length] max_length={chosen} (covers {(arr <= chosen).mean():.1%}, ceiling {ceiling})")
        if chosen == ceiling and arr.max() > ceiling:
            print(
                f"[length] WARNING: {(arr > ceiling).sum()} examples exceed the ceiling and "
                f"will lose context. A backbone with more positions may be needed."
            )
    return chosen


class TokenClassificationDataset(Dataset):
    """Encoded examples, ready for a collator. Unpadded; the collator pads."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        schema: cfg.ColumnSchema = cfg.COLUMNS,
        report: IntegrityReport | None = None,
    ):
        self.features: list[dict] = []
        self.ids: list[str] = []
        self.meta: list[dict] = []

        for _, r in df.iterrows():
            words = list(r["_words"])
            labels = [l for _, l in r["_tokens"]]
            feat, info = encode_example(
                r[schema.question], r[schema.context], words, labels, tokenizer, max_length
            )
            self.features.append(feat)
            self.ids.append(r[schema.id])
            self.meta.append(
                {
                    "id": r[schema.id],
                    "label": int(r[schema.label]),
                    "words": words,
                    "word_labels": labels,
                    "domain": r.get(schema.domain),
                    "hallucination_type": r.get(schema.hallucination_type),
                    "hallucinated_span": r.get(schema.hallucinated_span),
                }
            )
            if report is not None:
                lost = info["answer_words_total"] - info["answer_words_covered"]
                if lost > 0:
                    report.zero_subword_words += lost
                if info["overflowed"]:
                    report.answer_overflow.append(r[schema.id])
                if info["context_truncated"]:
                    report.context_truncated.append(r[schema.id])

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, i: int) -> dict:
        return self.features[i]


@dataclass
class Collator:
    """Dynamic padding. Padded positions get IGNORE_INDEX, never 0."""

    tokenizer: object
    pad_to_multiple_of: int | None = 8

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        labels = [f["labels"] for f in batch]
        no_labels = [{k: v for k, v in f.items() if k != "labels"} for f in batch]
        padded = self.tokenizer.pad(
            no_labels, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt"
        )
        width = padded["input_ids"].shape[1]
        padded["labels"] = torch.tensor(
            [lab + [cfg.IGNORE_INDEX] * (width - len(lab)) for lab in labels], dtype=torch.long
        )
        return padded


def build_datasets(
    csv_path: Path | str | None = None,
    backbone: str | None = None,
    verbose: bool = True,
) -> tuple[dict[str, TokenClassificationDataset], dict[str, pd.DataFrame], IntegrityReport, int]:
    """Full pipeline: load -> validate -> split -> measure length -> encode."""
    from transformers import AutoTokenizer

    backbone = backbone or cfg.BACKBONE
    df, report = load_dataframe(csv_path, verbose=verbose)
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    if not tokenizer.is_fast:
        raise ValueError(
            f"{backbone} has no fast tokenizer; word_ids() is unavailable and the "
            f"labelling scheme cannot work."
        )

    splits = group_split(df, verbose=verbose)
    max_length = choose_max_length(df, tokenizer, verbose=verbose)

    datasets = {
        name: TokenClassificationDataset(part, tokenizer, max_length, report=report)
        for name, part in splits.items()
    }
    if verbose:
        print("\n[encode] done")
        if report.zero_subword_words:
            print(f"[encode] WARNING: {report.zero_subword_words} answer words got no subword")
        if report.answer_overflow:
            print(f"[encode] WARNING: {len(report.answer_overflow)} answers overflow max_length")
        if report.context_truncated:
            print(f"[encode] {len(report.context_truncated)} examples had context truncated")
    return datasets, splits, report, max_length


if __name__ == "__main__":
    datasets, splits, report, max_length = build_datasets()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.BACKBONE)
    ds = datasets["train"]
    feat, meta = ds[0], ds.meta[0]

    print("\n--- alignment spot check ---")
    print(f"id={meta['id']}  label={meta['label']}  max_length={max_length}")
    toks = tok.convert_ids_to_tokens(feat["input_ids"])
    print("decoded:", tok.decode(feat["input_ids"])[:220], "...")
    print("\nlabelled positions (answer only):")
    for pos, (t, l) in enumerate(zip(toks, feat["labels"])):
        if l != cfg.IGNORE_INDEX:
            print(f"  [{pos:>3}] {t!r:<24} label={l}")
    print(f"\nwords={meta['words']}")
    print(f"word_labels={meta['word_labels']}")
    n_labelled = sum(1 for l in feat["labels"] if l != cfg.IGNORE_INDEX)
    print(f"labelled tokens={n_labelled}  answer words={len(meta['words'])}  (equal => first-subword scheme OK)")
