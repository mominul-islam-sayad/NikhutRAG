"""LLM-as-a-judge reference point, run locally, scored exactly like the SLM.

The paper's argument is that a small token classifier is cheaper than asking an
LLM to judge. That needs a measured LLM row in the same table, so this module
asks a local instruction-tuned LLM (served by llama.cpp's llama-server) to quote
the unsupported parts of each answer, maps those quotes back onto the answer's
words, and scores them with the same src/evaluate.py code as every other system.

Serve the model first (Vulkan build, RX 6600), one request at a time:

    llama-server.exe -m gemma-4-E4B-it-Q4_0.gguf -ngl 99 -c 8192 -np 1 --jinja ^
        --reasoning off --reasoning-budget 0 --port 8080

Reasoning must be off at the server: Gemma 4 otherwise sometimes thinks out
loud despite ``enable_thinking: false`` in the request, spends the whole token
budget there and returns empty content. The judge measured here is the
non-thinking mode, which is the fast and cheap one.

then:

    python -m src.llm_judge --name gemma-4-E4B-q4 --split test

Responses are cached in outputs/judge/<name>.jsonl, one line per example, so an
interrupted run resumes where it stopped. Latency is the wall-clock time of each
request at batch size 1, the same unit as the SLM latency.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

from . import config as cfg
from . import evaluate as ev
from .baselines import LexicalOverlapBaseline
from .data import group_split, load_dataframe, tokenize_answer

JUDGE_DIR = cfg.OUTPUT_DIR / "judge"

SYSTEM_PROMPT = (
    "You are a strict fact-checker for a Bangla question-answering system that answers "
    "from a retrieved context. The context is the only source of truth: do not use "
    "outside knowledge. Your job is to find every part of the answer that is not "
    "supported by the context or that contradicts it: wrong names, numbers, dates, "
    "facts added that the context does not state, or content that misstates what the "
    "context says."
)

INSTRUCTION = (
    "Copy each unsupported part exactly as it appears in the answer, character for "
    "character, and keep each part as short as possible (usually one to four words). "
    "Do not quote parts of the answer that the context supports. If the whole answer is "
    "supported by the context, return an empty list. Respond with JSON only, in the "
    'form {"hallucinated_spans": ["...", "..."]}.'
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"hallucinated_spans": {"type": "array", "items": {"type": "string"}}},
    "required": ["hallucinated_spans"],
}

_PUNCT_ONLY = re.compile(r"^\W+$")


def _user_message(context: str, question: str, answer: str) -> str:
    return (
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n{INSTRUCTION}"
    )


def pick_fewshot(train, schema: cfg.ColumnSchema = cfg.COLUMNS, seed: int = cfg.SPLIT_SEED) -> list[dict]:
    """One faithful and one hallucinated demonstration, from the training split.

    Taken from train only, so the judge sees no test data; short contexts keep
    the prompt cheap. The hallucinated one shows the expected quoting.
    """
    short = train[train[schema.context].str.len() < train[schema.context].str.len().quantile(0.3)]
    demos = []
    for label in (1, 0):
        r = short[short[schema.label] == label].sample(1, random_state=seed).iloc[0]
        spans = [t for _, _, t in ev.extract_spans(list(r["_words"]), [l for _, l in r["_tokens"]])]
        demos.append(
            {
                "user": _user_message(r[schema.context], r[schema.question], r[schema.answer]),
                "assistant": json.dumps({"hallucinated_spans": spans}, ensure_ascii=False),
            }
        )
    return demos


def build_messages(context: str, question: str, answer: str, fewshot: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for d in fewshot:
        msgs.append({"role": "user", "content": d["user"]})
        msgs.append({"role": "assistant", "content": d["assistant"]})
    msgs.append({"role": "user", "content": _user_message(context, question, answer)})
    return msgs


def call_server(server: str, messages: list[dict], max_tokens: int = 256, timeout: int = 600) -> dict:
    body = {
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": {"schema": RESPONSE_SCHEMA}},
        # Reasoning models would otherwise spend the budget thinking out loud.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{server.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    out["_latency_s"] = time.perf_counter() - t0
    return out


def parse_spans(content: str) -> tuple[list[str], bool]:
    """Spans from the model's JSON. Returns (spans, parsed_ok)."""
    try:
        obj = json.loads(content)
        spans = obj.get("hallucinated_spans", [])
        return [str(s) for s in spans if str(s).strip()], True
    except (json.JSONDecodeError, AttributeError):
        # Fall back to anything quoted, rather than scoring a format slip as "clean".
        return re.findall(r'"([^"]+)"', content or "")[1:], False


def spans_to_word_labels(words: list[str], spans: list[str], scheme: str = "regex") -> tuple[list[int], int]:
    """Mark the answer words each quoted span covers.

    First tries an exact contiguous match of the span's words; failing that,
    marks the answer words that appear in the span, so a quote with a small
    slip still lands on the right words. Punctuation-only words are never
    marked on their own. Returns (labels, spans that matched nothing).
    """
    labels = [0] * len(words)
    unmatched = 0
    for span in spans:
        sw = [w for w in tokenize_answer(span, scheme) if not _PUNCT_ONLY.match(w)]
        if not sw:
            continue
        hit = False
        for i in range(len(words) - len(sw) + 1):
            window = [w for w in words[i : i + len(sw)]]
            if window == sw:
                for j in range(i, i + len(sw)):
                    labels[j] = 1
                hit = True
        if not hit:
            wanted = set(sw)
            for j, w in enumerate(words):
                if w in wanted and not _PUNCT_ONLY.match(w):
                    labels[j] = 1
                    hit = True
        unmatched += 0 if hit else 1
    return labels, unmatched


def run(args: argparse.Namespace) -> dict:
    schema = cfg.COLUMNS
    df, report = load_dataframe(verbose=False)
    splits = group_split(df, verbose=False)
    target = splits[args.split]
    if args.limit:
        target = target.head(args.limit)
    fewshot = pick_fewshot(splits["train"]) if args.fewshot else []

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = JUDGE_DIR / f"{args.name}.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["id"]] = row

    todo = [r for _, r in target.iterrows() if r[schema.id] not in cache]
    print(f"[judge] {args.name}: {len(target)} rows, {len(cache)} cached, {len(todo)} to query", flush=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        for k, r in enumerate(todo, start=1):
            msgs = build_messages(r[schema.context], r[schema.question], r[schema.answer], fewshot)
            try:
                out = call_server(args.server, msgs, max_tokens=args.max_tokens)
                content = out["choices"][0]["message"].get("content") or ""
                usage = out.get("usage", {})
                row = {
                    "id": r[schema.id],
                    "content": content,
                    "latency_s": out["_latency_s"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "finish_reason": out["choices"][0].get("finish_reason"),
                }
            except Exception as exc:  # noqa: BLE001
                row = {"id": r[schema.id], "content": "", "error": str(exc)[:300], "latency_s": None}
            cache[row["id"]] = row
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if k % 25 == 0 or k == len(todo):
                lat = [c["latency_s"] for c in cache.values() if c.get("latency_s")]
                print(
                    f"[judge] {k}/{len(todo)}  median {statistics.median(lat):.2f}s/example",
                    flush=True,
                )

    records, n_bad_json, n_unmatched, n_errors = [], 0, 0, 0
    for _, r in target.iterrows():
        row = cache[r[schema.id]]
        words = list(r["_words"])
        if row.get("error"):
            n_errors += 1
        spans, ok = parse_spans(row.get("content", ""))
        n_bad_json += 0 if ok else 1
        labels, unmatched = spans_to_word_labels(words, spans, report.word_scheme)
        n_unmatched += unmatched
        records.append(
            ev.Record(
                id=r[schema.id],
                words=words,
                gold_word_labels=[l for _, l in r["_tokens"]],
                pred_word_labels=labels,
                gold_label=int(r[schema.label]),
                # The judge gives a verdict, not a probability, so AUROC is
                # computed on 0/1 scores and equals balanced accuracy.
                score=float(any(labels)),
                gold_span=r.get(schema.hallucinated_span),
                hallucination_type=r.get(schema.hallucination_type),
                domain=r.get(schema.domain),
            )
        )

    judge_report = ev.full_report(records, f"judge:{args.name}")
    baselines = [
        ev.full_report(LexicalOverlapBaseline(mode=m).run(target), LexicalOverlapBaseline(mode=m).name)
        for m in ("exact", "morph")
    ]
    print(f"\n=== {args.split} ({len(target)} rows) ===")
    print(ev.format_table(baselines + [judge_report]))
    print("\nper hallucination type:")
    for t, m in judge_report["per_type"].items():
        rec = m["example_recall"]
        print(f"  {t:<22} n={m['n']:<5} ex_recall={f'{rec:.3f}' if rec is not None else ' n/a'}  word_F1={m['word_f1']:.3f}")

    lat = sorted(c["latency_s"] for c in (cache[r[schema.id]] for _, r in target.iterrows()) if c.get("latency_s"))
    ptoks = [c["prompt_tokens"] for c in (cache[r[schema.id]] for _, r in target.iterrows()) if c.get("prompt_tokens")]
    latency = {
        "median_s": statistics.median(lat) if lat else None,
        "p95_s": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "mean_s": statistics.fmean(lat) if lat else None,
        "total_s": sum(lat),
        "mean_prompt_tokens": statistics.fmean(ptoks) if ptoks else None,
    }
    print(
        f"\n[judge] latency median {latency['median_s']:.2f}s  p95 {latency['p95_s']:.2f}s  "
        f"total {latency['total_s'] / 60:.1f} min  prompt {latency['mean_prompt_tokens']:.0f} tokens avg"
    )
    print(f"[judge] unparseable JSON {n_bad_json}, spans matching no answer word {n_unmatched}, errors {n_errors}")

    result = {
        "run": f"judge-{args.name}",
        "judge": args.name,
        "server": args.server,
        "split": args.split,
        "n": len(target),
        "fewshot": len(fewshot),
        "system_prompt": SYSTEM_PROMPT,
        "instruction": INSTRUCTION,
        "test": judge_report,
        "baselines": baselines,
        "latency": latency,
        "quality": {"bad_json": n_bad_json, "unmatched_spans": n_unmatched, "errors": n_errors},
    }
    out = cfg.METRICS_DIR / f"judge-{args.name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[judge] metrics -> {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="label for this judge, e.g. gemma-4-E4B-q4")
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-fewshot", dest="fewshot", action="store_false")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
