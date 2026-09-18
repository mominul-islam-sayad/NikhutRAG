r"""RAGTruth -> the word-level schema the rest of the pipeline already speaks.

The English arm approved on 2026-09-18 exists so that Luna and LettuceDetect,
which the FYDP-I paper names as the baseline and the system to beat, remain
comparable. Neither exists in Bangla.

Get the data:
    curl -sL -o data/ragtruth/response.jsonl \
      https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl
    curl -sL -o data/ragtruth/source_info.jsonl \
      https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl

Then:
    python -m src.data_ragtruth --dir data/ragtruth --out data/ragtruth_qa.csv --task QA

Three things this adapter has to get right, all measured on the real files:

1. **Leakage.** Every source_id has exactly 6 responses, one per generating
   model. Splitting rows at random scatters one context across all splits --
   the same trap as the Bangla `_F`/`_H` twins, six times worse. Emitted ids
   are `RT_{source_id}_m{k}`, so set in config.py:

       ID_SUFFIX_PATTERN = r"_m\d+$"

   and src/data.py groups the six siblings together automatically.

2. **Char spans vs words.** RAGTruth annotates character offsets; this pipeline
   labels whitespace words. 5,727 of 14,289 spans (40.1%) cut into the middle
   of a word, so those get widened to the whole word. That inflates the
   positive region slightly and must be reported, not hidden -- the count is
   printed and stored.

3. **Length.** Contexts are median 318 words, p95 905, max 1749; responses are
   median 124 words. A 512-position encoder cannot hold question + context +
   answer, and `truncation="only_first"` would delete the very evidence the
   task is about. Use a long-context backbone (ModernBERT-base, 8192) for this
   arm. This is precisely the Luna limitation the paper's gap analysis row 1
   describes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

TASK_INSTRUCTION = {
    "Summary": "Summarize the passage below.",
    "Data2txt": "Write a review of the business described below.",
    "QA": None,  # the real question lives in source_info["question"]
}


def _render_source(task_type: str, source_info) -> tuple[str, str]:
    """Return (question, context) for one source record."""
    if task_type == "QA" and isinstance(source_info, dict):
        question = str(source_info.get("question", "")).strip()
        passages = source_info.get("passages", "")
        if isinstance(passages, (list, tuple)):
            context = "\n".join(str(p) for p in passages)
        else:
            context = str(passages)
        return question, context.strip()

    if task_type == "Data2txt" and isinstance(source_info, dict):
        parts = []
        for k in sorted(source_info):
            v = source_info[k]
            parts.append(f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}")
        return TASK_INSTRUCTION["Data2txt"], "\n".join(parts)

    if isinstance(source_info, dict):
        return TASK_INSTRUCTION.get(task_type) or "", json.dumps(source_info, ensure_ascii=False)
    return TASK_INSTRUCTION.get(task_type) or "", str(source_info)


def word_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    words, offs, i = text.split(), [], 0
    for w in words:
        j = text.find(w, i)
        if j < 0:  # should not happen for whitespace splits
            j = i
        offs.append((j, j + len(w)))
        i = j + len(w)
    return words, offs


def spans_to_word_labels(text: str, spans: list[dict]) -> tuple[list[str], list[int], int]:
    """Mark any word overlapping an annotated char span.

    Returns (words, labels, n_widened) where n_widened counts spans that did
    not align to word boundaries and therefore claimed more text than the
    annotator marked.
    """
    words, offs = word_offsets(text)
    labels = [0] * len(words)
    widened = 0
    for sp in spans:
        s, e = sp.get("start"), sp.get("end")
        if s is None or e is None:
            continue
        touched = [k for k, (ws, we) in enumerate(offs) if ws < e and we > s]
        if not touched:
            continue
        for k in touched:
            labels[k] = 1
        ws0, we0 = offs[touched[0]][0], offs[touched[-1]][1]
        if ws0 < s or we0 > e:
            widened += 1
    return words, labels, widened


def load_ragtruth(
    directory: Path | str,
    task: str | None = None,
    split: str | None = None,
    drop_bad_quality: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    directory = Path(directory)
    resp_path, src_path = directory / "response.jsonl", directory / "source_info.jsonl"
    for p in (resp_path, src_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; see this module's docstring for the download commands")

    responses = [json.loads(l) for l in resp_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    sources = [json.loads(l) for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {s["source_id"]: s for s in sources}

    per_source_seen: Counter = Counter()
    rows, total_widened, skipped = [], 0, 0

    for r in responses:
        src = by_id.get(r["source_id"])
        if src is None:
            skipped += 1
            continue
        task_type = src["task_type"]
        if task and task_type != task:
            continue
        if split and r.get("split") != split:
            continue
        if drop_bad_quality and r.get("quality") != "good":
            continue

        k = per_source_seen[r["source_id"]]
        per_source_seen[r["source_id"]] += 1

        question, context = _render_source(task_type, src.get("source_info"))
        answer = r["response"]
        spans = r.get("labels") or []
        words, labels, widened = spans_to_word_labels(answer, spans)
        total_widened += widened

        if not words:
            skipped += 1
            continue

        rows.append(
            {
                "id": f"RT_{r['source_id']}_m{k}",
                "domain": task_type,
                "context": context,
                "question": question,
                "answer": answer,
                "label": int(bool(spans)),
                "hallucination_type": spans[0]["label_type"] if spans else "none",
                "hallucinated_span": spans[0]["text"] if spans else None,
                "token_labels": json.dumps([[w, l] for w, l in zip(words, labels)], ensure_ascii=False),
                "is_human_verified": True,  # RAGTruth is fully human annotated
                "verification_status": "human_annotated",
                "split_official": r.get("split"),
                "model": r.get("model"),
            }
        )

    df = pd.DataFrame(rows)
    if verbose and len(df):
        _report(df, total_widened, skipped)
    return df


def _report(df: pd.DataFrame, widened: int, skipped: int) -> None:
    n = len(df)
    pos = int(df["label"].sum())
    ans_len = sorted(len(str(a).split()) for a in df["answer"])
    ctx_len = sorted(len(str(c).split()) for c in df["context"])
    print(f"[ragtruth] rows={n}  skipped={skipped}")
    print(f"[ragtruth] hallucinated={pos} ({pos / n:.1%})  -- note: NOT the 15-20% often assumed")
    print(f"[ragtruth] task types: {dict(Counter(df['domain']))}")
    print(f"[ragtruth] official split: {dict(Counter(df['split_official']))}")
    print(f"[ragtruth] spans widened to word boundaries: {widened}")
    print(
        f"[ragtruth] answer words  median={statistics.median(ans_len)} "
        f"p95={ans_len[int(0.95 * (len(ans_len) - 1))]} max={ans_len[-1]}"
    )
    print(
        f"[ragtruth] context words median={statistics.median(ctx_len)} "
        f"p95={ctx_len[int(0.95 * (len(ctx_len) - 1))]} max={ctx_len[-1]}"
    )
    print(
        "[ragtruth] WARNING: a 512-position encoder cannot hold these. Use a "
        "long-context backbone (answerdotai/ModernBERT-base, 8192) for this arm."
    )

    # The official split must not scatter one source across sides.
    base = df["id"].str.replace(r"_m\d+$", "", regex=True)
    grp = df.assign(base=base).groupby("base")["split_official"].nunique()
    bad = int((grp > 1).sum())
    print(f"[ragtruth] official split is source-disjoint: {bad == 0} ({bad} sources straddle)")
    print(r'[ragtruth] set ID_SUFFIX_PATTERN = r"_m\d+$" in config.py before using src/data.py')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/ragtruth", help="directory holding the two .jsonl files")
    ap.add_argument("--out", default=None, help="write a CSV that src/data.py can load directly")
    ap.add_argument("--task", default=None, choices=["QA", "Summary", "Data2txt"])
    ap.add_argument("--split", default=None, choices=["train", "test"])
    ap.add_argument("--keep-bad-quality", action="store_true")
    args = ap.parse_args()

    df = load_ragtruth(args.dir, task=args.task, split=args.split, drop_bad_quality=not args.keep_bad_quality)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"[ragtruth] wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
