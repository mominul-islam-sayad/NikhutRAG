"""The lexical baseline that has to appear in every results table.

CLAUDE.md is blunt about why: flagging any answer word absent from the context
already scores F1 ~0.90 at example level, which is *above* the paper's stated
>75% target. A fine-tuned SLM that scores 0.95 has therefore proved nothing
unless this row sits next to it. Hiding it is worse than the weakness.

Two matching modes are provided, and the gap between them is itself a finding:

  exact      -- a word counts as grounded only if it appears verbatim in the
                context. Bangla is agglutinative, so "গ্রিসে" (locative) does not
                match "গ্রিস" in the context and gets falsely flagged.
  morph      -- prefix/containment matching, which absorbs case suffixes and the
                danda. This is the *fair* lexical baseline; `exact` flatters the
                model by handicapping the baseline with a morphology bug.

Report `morph`. Reporting only `exact` would overstate the model's margin.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from . import config as cfg
from . import evaluate as ev
from .data import load_dataframe, group_split


@dataclass
class LexicalOverlapBaseline:
    """No model, no training. Pure string comparison against the context."""

    mode: str = "morph"  # "exact" | "morph"
    strip_punct: bool = True
    lowercase: bool = True
    #: Minimum stem length before prefix matching is allowed, so that very short
    #: tokens do not match everything.
    min_stem: int = 3

    def __post_init__(self):
        if self.mode not in ("exact", "morph"):
            raise ValueError(f"unknown mode {self.mode!r}")

    @property
    def name(self) -> str:
        return f"lexical-overlap[{self.mode}]"

    def _tokens(self, text: str) -> list[str]:
        return ev.normalize(text, self.strip_punct, self.lowercase).split()

    def _grounded(self, word: str, ctx_tokens: set[str]) -> bool:
        if not word:
            return True
        if word in ctx_tokens:
            return True
        if self.mode == "exact":
            return False
        # Absorb Bangla case endings in either direction: the answer may inflect
        # a context word, or the context may inflect the answer's stem.
        for c in ctx_tokens:
            if len(word) >= self.min_stem and c.startswith(word):
                return True
            if len(c) >= self.min_stem and word.startswith(c):
                return True
        return False

    def predict_words(self, context: str, answer_words: list[str]) -> list[int]:
        ctx = set(self._tokens(context))
        out = []
        for w in answer_words:
            nw = ev.normalize(w, self.strip_punct, self.lowercase)
            out.append(0 if self._grounded(nw, ctx) else 1)
        return out

    def score(self, preds: list[int]) -> float:
        """Continuous example score so AUROC is defined: the fraction of answer
        words that look ungrounded."""
        return sum(preds) / max(len(preds), 1)

    def run(self, df, schema: cfg.ColumnSchema = cfg.COLUMNS) -> list[ev.Record]:
        records = []
        for _, r in df.iterrows():
            words = list(r["_words"])
            preds = self.predict_words(r[schema.context], words)
            records.append(
                ev.Record(
                    id=r[schema.id],
                    words=words,
                    gold_word_labels=[l for _, l in r["_tokens"]],
                    pred_word_labels=preds,
                    gold_label=int(r[schema.label]),
                    score=self.score(preds),
                    gold_span=r.get(schema.hallucinated_span),
                    hallucination_type=r.get(schema.hallucination_type),
                    domain=r.get(schema.domain),
                )
            )
        return records


def run_all(split: str = "test", save: bool = False) -> list[dict]:
    df, report = load_dataframe(verbose=False)
    splits = group_split(df, verbose=False)
    target = splits[split] if split in splits else df

    print(f"[baseline] split={split}  rows={len(target)}")
    print(
        f"[baseline] NOTE: {len(report.label_says_hallucinated_tokens_say_clean)} rows in the "
        f"full dataset carry label=1 with no flagged word, so word-level recall is "
        f"capped below 1.0 for any system."
    )

    reports = []
    for mode in ("exact", "morph"):
        bl = LexicalOverlapBaseline(mode=mode)
        recs = bl.run(target)
        rep = ev.full_report(recs, bl.name)
        reports.append(rep)
        if save:
            ev.save_report(rep)

    print("\n" + ev.format_table(reports))

    print("\nper hallucination type (morph):")
    for t, m in reports[-1]["per_type"].items():
        rec = m["example_recall"]
        rec_s = f"{rec:.3f}" if rec is not None else "  n/a"
        print(f"  {t:<22} n={m['n']:<6} hallucinated={m['n_hallucinated']:<6} ex_recall={rec_s}  word_F1={m['word_f1']:.3f}")

    sm = reports[-1]["span_level"]
    if sm["n_unusable"]:
        print(
            f"\nspan scoring skipped {sm['n_unusable']} hallucinated rows whose gold span "
            f"disagrees with their gold token labels (dataset defect, not model error)"
        )
    return reports


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--save", action="store_true", help="write JSON to outputs/metrics")
    args = ap.parse_args()
    run_all(split=args.split, save=args.save)
