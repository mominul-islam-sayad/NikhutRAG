"""Metrics, span extraction, latency and efficiency reporting.

The FYDP-I paper names Precision, Recall, F1, Accuracy and AUROC. CLAUDE.md
adds word-level, span-level and per-hallucination-type breakdowns, plus latency
and efficiency. All of it lives here so the baseline and the model are scored by
exactly the same code -- a results table is only honest if every row was
computed the same way.

Everything is built around a list of `Record`s, so anything that can produce a
per-word prediction can be evaluated: the lexical baseline, a fine-tuned model,
or an external judge.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from . import config as cfg

#: Bangla sentence terminator plus ASCII/Unicode punctuation we ignore when
#: comparing surface forms. The danda attaches to the word with no space, so
#: leaving it in makes "গ্রিস।" and "গ্রিস" look like different words.
_PUNCT = re.compile(r"[।,.!?;:\"'`()\[\]{}<>/\\|@#$%^&*_~+=-]+")
_WS = re.compile(r"\s+")


def normalize(text: str, strip_punct: bool = True, lowercase: bool = True) -> str:
    s = str(text)
    if strip_punct:
        s = _PUNCT.sub("", s)
    if lowercase:
        s = s.lower()
    return _WS.sub(" ", s).strip()


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass
class Record:
    """One evaluated example. `score` is the example-level hallucination
    probability -- for the model, the max over answer-token probabilities."""

    id: str
    words: list[str]
    gold_word_labels: list[int]
    pred_word_labels: list[int]
    gold_label: int
    score: float = 0.0
    pred_label: int | None = None
    gold_span: str | None = None
    hallucination_type: str | None = None
    domain: str | None = None

    def __post_init__(self):
        if len(self.words) != len(self.gold_word_labels):
            raise ValueError(f"{self.id}: words/gold length mismatch")
        if len(self.words) != len(self.pred_word_labels):
            raise ValueError(f"{self.id}: words/pred length mismatch")
        if self.pred_label is None:
            self.pred_label = int(any(self.pred_word_labels))


# --------------------------------------------------------------------------
# Span handling
# --------------------------------------------------------------------------


def extract_spans(words: Sequence[str], labels: Sequence[int]) -> list[tuple[int, int, str]]:
    """Merge runs of consecutive flagged words. Returns (start, end, text)."""
    spans: list[tuple[int, int, str]] = []
    start: int | None = None
    for i, lab in enumerate(list(labels) + [0]):
        if lab and start is None:
            start = i
        elif not lab and start is not None:
            spans.append((start, i, " ".join(words[start:i])))
            start = None
    return spans


# --------------------------------------------------------------------------
# Metric blocks
# --------------------------------------------------------------------------


@dataclass
class WordMetrics:
    precision: float
    recall: float
    f1: float
    support: int
    tp: int
    fp: int
    fn: int


@dataclass
class ExampleMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    auroc: float | None
    n: int


@dataclass
class SpanMetrics:
    #: Share of hallucinated examples where every gold span was predicted with
    #: exactly its boundaries.
    exact: float
    #: Share of hallucinated examples where some predicted span overlaps a gold one.
    partial: float
    #: Exact-boundary span P/R/F1 pooled over all examples, faithful ones
    #: included, so a span predicted on a faithful answer is a false positive.
    precision: float
    recall: float
    f1: float
    n_gold: int
    #: Hallucinated examples whose gold token labels flag nothing, so there is
    #: no gold span to match. Excluded rather than charged to the system.
    n_unusable: int


def word_level_metrics(records: Iterable[Record]) -> WordMetrics:
    y_true: list[int] = []
    y_pred: list[int] = []
    for r in records:
        y_true.extend(r.gold_word_labels)
        y_pred.extend(r.pred_word_labels)
    if not y_true:
        return WordMetrics(0.0, 0.0, 0.0, 0, 0, 0, 0)
    p, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0
    )
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return WordMetrics(float(p), float(rec), float(f1), int(sum(y_true)), tp, fp, fn)


def example_level_metrics(records: Sequence[Record]) -> ExampleMetrics:
    y_true = [r.gold_label for r in records]
    y_pred = [int(r.pred_label) for r in records]
    y_score = [r.score for r in records]
    if not y_true:
        return ExampleMetrics(0.0, 0.0, 0.0, 0.0, None, 0)
    p, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    auroc: float | None = None
    if len(set(y_true)) > 1 and len(set(y_score)) > 1:
        auroc = float(roc_auc_score(y_true, y_score))
    return ExampleMetrics(float(p), float(rec), float(f1), float(acc), auroc, len(y_true))


def span_level_metrics(records: Iterable[Record]) -> SpanMetrics:
    """Span match by word positions, gold spans taken from the gold word labels.

    Gold and predicted spans are both runs of flagged words over the same word
    list, so they are compared by (start, end) rather than by text. Text
    comparison against ``hallucinated_span`` stopped working once punctuation
    became its own word, and the gold token labels are what the model is
    trained on, so they are the consistent target. The 4,000-row dataset's
    token labels agree with ``hallucinated_span`` on ~94% of rows.
    """
    exact = partial = n = unusable = 0
    tp = n_pred = n_gold_spans = 0
    for r in records:
        gold = {(s, e) for s, e, _ in extract_spans(r.words, r.gold_word_labels)}
        pred = {(s, e) for s, e, _ in extract_spans(r.words, r.pred_word_labels)}
        tp += len(gold & pred)
        n_pred += len(pred)
        n_gold_spans += len(gold)

        if r.gold_label != 1:
            continue
        if not gold:
            unusable += 1
            continue
        n += 1
        if gold <= pred:
            exact += 1
        if any(ps < ge and gs < pe for ps, pe in pred for gs, ge in gold):
            partial += 1
    denom = max(n, 1)
    p = tp / n_pred if n_pred else 0.0
    rc = tp / n_gold_spans if n_gold_spans else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    return SpanMetrics(exact / denom, partial / denom, p, rc, f1, n, unusable)


def per_type_metrics(records: Sequence[Record]) -> dict[str, dict]:
    """Per hallucination type. Test-split counts per type are small (tens), so
    the interval is wide and n is always reported beside the score."""
    out: dict[str, dict] = {}
    types = sorted({r.hallucination_type for r in records if r.hallucination_type})
    for t in types:
        subset = [r for r in records if r.hallucination_type == t]
        pos = [r for r in subset if r.gold_label == 1]
        detected = sum(1 for r in pos if r.pred_label == 1)
        wm = word_level_metrics(subset)
        out[t] = {
            "n": len(subset),
            "n_hallucinated": len(pos),
            "example_recall": detected / max(len(pos), 1) if pos else None,
            "word_f1": wm.f1,
        }
    return out


def per_domain_metrics(records: Sequence[Record]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in sorted({r.domain for r in records if r.domain}):
        subset = [r for r in records if r.domain == d]
        em = example_level_metrics(subset)
        out[d] = {"n": len(subset), "example_f1": em.f1, "accuracy": em.accuracy}
    return out


# --------------------------------------------------------------------------
# Latency and efficiency
# --------------------------------------------------------------------------


def latency_benchmark(
    fn: Callable[[], object],
    warmup: int = cfg.LATENCY_WARMUP_RUNS,
    runs: int = cfg.LATENCY_MEASURED_RUNS,
    device: str = "cpu",
) -> dict:
    """Wall-clock ms per call at batch size 1.

    Must be run on the machine the claim is about. The sub-200ms requirement in
    the paper is about consumer hardware, so a Colab number does not support it.
    """
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "device": device,
        "batch_size": cfg.LATENCY_BATCH_SIZE,
        "warmup": warmup,
        "runs": runs,
        "mean_ms": statistics.fmean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * (len(times) - 1))],
        "min_ms": times[0],
        "max_ms": times[-1],
        "meets_200ms": statistics.median(times) < 200.0,
    }


def efficiency_stats(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
        "param_budget_500m_ok": total <= 500_000_000,
        "fp32_size_mb": total * 4 / (1024**2),
    }


# --------------------------------------------------------------------------
# Full report
# --------------------------------------------------------------------------


def full_report(records: Sequence[Record], name: str) -> dict:
    wm = word_level_metrics(records)
    em = example_level_metrics(records)
    sm = span_level_metrics(records)
    return {
        "name": name,
        "n_examples": len(records),
        "word_level": asdict(wm),
        "example_level": asdict(em),
        "span_level": asdict(sm),
        "per_type": per_type_metrics(records),
        "per_domain": per_domain_metrics(records),
    }


def format_table(reports: Sequence[dict]) -> str:
    """One row per system. The lexical baseline must always be one of them."""
    head = (
        f"{'system':<44}{'ex_P':>7}{'ex_R':>7}{'ex_F1':>7}{'ex_Acc':>8}{'AUROC':>8}"
        f"{'wd_P':>7}{'wd_R':>7}{'wd_F1':>7}{'sp_F1':>7}{'sp_ex':>7}{'sp_pt':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in reports:
        e, w, s = r["example_level"], r["word_level"], r["span_level"]
        auroc = f"{e['auroc']:.3f}" if e["auroc"] is not None else "  n/a"
        lines.append(
            f"{r['name']:<44}{e['precision']:>7.3f}{e['recall']:>7.3f}{e['f1']:>7.3f}"
            f"{e['accuracy']:>8.3f}{auroc:>8}"
            f"{w['precision']:>7.3f}{w['recall']:>7.3f}{w['f1']:>7.3f}"
            f"{s['f1']:>7.3f}{s['exact']:>7.3f}{s['partial']:>7.3f}"
        )
    return "\n".join(lines)


def save_report(report: dict, path: Path | str | None = None) -> Path:
    path = Path(path) if path else cfg.METRICS_DIR / f"{report['name'].replace('/', '_')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
