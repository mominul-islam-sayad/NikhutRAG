"""Human-verified gold test set: sample, track, measure agreement, evaluate.

The dataset's labels come from a generator and none were checked by a person.
This builds a ~200-item gold set from the TEST split, has it annotated blind by
two team members per item, and scores every system against the human labels.

    python -m src.gold sample      # draw the items and assign annotators (once)
    streamlit run app/annotate.py  # each teammate annotates their share
    python -m src.gold status      # progress per annotator
    python -m src.gold agree       # inter-annotator agreement, adjudication queue
    python -m src.gold evaluate    # every system vs the human labels

Design choices, each for a stated reason:

- Test split only, so no system was trained or tuned on a gold item.
- Balanced 100 faithful / 100 hallucinated, hallucinated spread evenly over the
  six types, faithful spread over domains.
- At most one row per (context, question) pair. The faithful and hallucinated
  answers to the same question differ in a word or two; an annotator who saw
  both would find the hallucination by comparison instead of by reading.
- Blind: annotators see context, question and answer only, never the
  generator's label or type.
- Two annotators per item, rotating over all six pairs of the four-person team,
  so agreement (Cohen's kappa) is measurable for every pair.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics

import numpy as np
import pandas as pd

from . import config as cfg
from . import evaluate as ev
from . import gold_io as gio

N_FAITHFUL = 100
N_HALLUCINATED = 100
SEED = 2026


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------


def sample(force: bool = False) -> pd.DataFrame:
    from .data import group_split, load_dataframe

    if gio.ITEMS_CSV.exists() and not force:
        existing = gio.load_all_annotations()
        if any(existing.values()):
            raise SystemExit("gold_items.csv already has annotations; refusing to resample (use --force)")
    schema = cfg.COLUMNS
    df, _ = load_dataframe(verbose=False)
    test = group_split(df, verbose=False)["test"].copy()
    rng = np.random.default_rng(SEED)
    test["_pair"] = test["group_id"].astype(str) + "||" + test[schema.question].astype(str)

    used_pairs: set[str] = set()
    chosen: list[pd.Series] = []

    # Hallucinated: even quota per type, remainder to the first types.
    halluc = test[test[schema.label] == 1]
    types = sorted(halluc[schema.hallucination_type].unique())
    base, extra = divmod(N_HALLUCINATED, len(types))
    for i, t in enumerate(types):
        quota = base + (1 if i < extra else 0)
        pool = halluc[halluc[schema.hallucination_type] == t]
        pool = pool.iloc[rng.permutation(len(pool))]
        taken = 0
        for _, r in pool.iterrows():
            if taken == quota:
                break
            if r["_pair"] in used_pairs:
                continue
            chosen.append(r)
            used_pairs.add(r["_pair"])
            taken += 1
        if taken < quota:
            raise SystemExit(f"only {taken} usable {t} rows in test, need {quota}")

    # Faithful: from pairs not used above, round-robin over domains.
    faithful = test[(test[schema.label] == 0) & (~test["_pair"].isin(used_pairs))]
    by_domain = {
        d: list(g.iloc[rng.permutation(len(g))].iterrows())
        for d, g in faithful.groupby(schema.domain)
    }
    taken = 0
    while taken < N_FAITHFUL and any(by_domain.values()):
        for d in sorted(by_domain):
            if taken == N_FAITHFUL or not by_domain[d]:
                continue
            _, r = by_domain[d].pop()
            chosen.append(r)
            used_pairs.add(r["_pair"])
            taken += 1

    items = pd.DataFrame(chosen).reset_index(drop=True)
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)

    pairs = list(itertools.combinations(gio.ANNOTATORS, 2))
    out = pd.DataFrame(
        {
            "id": items[schema.id],
            "context_id": items["group_id"],
            "domain": items[schema.domain],
            "context": items[schema.context],
            "question": items[schema.question],
            "answer": items[schema.answer],
            "words": items["_words"].map(lambda w: json.dumps(list(w), ensure_ascii=False)),
            "annotators": [json.dumps(list(pairs[i % len(pairs)])) for i in range(len(items))],
            "position": range(len(items)),
        }
    )
    gio.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(gio.ITEMS_CSV, index=False, encoding="utf-8")

    labels = items[schema.label].value_counts().to_dict()
    load = {a: sum(a in json.loads(x) for x in out["annotators"]) for a in gio.ANNOTATORS}
    print(f"[gold] {len(out)} items -> {gio.ITEMS_CSV}")
    print(f"[gold] labels {labels}  types {items[schema.hallucination_type].value_counts().to_dict()}")
    print(f"[gold] domains {items[schema.domain].value_counts().to_dict()}")
    print(f"[gold] items per annotator {load}  (two annotators per item, {len(pairs)} rotating pairs)")
    return out


# --------------------------------------------------------------------------
# status and agreement
# --------------------------------------------------------------------------


def status() -> None:
    items = gio.load_items()
    ann = gio.load_all_annotations()
    print(f"[gold] {len(items)} items")
    for a in gio.ANNOTATORS:
        mine = items[items["annotators"].map(lambda x: a in x)]
        done = sum(1 for i in mine["id"] if i in ann[a])
        print(f"  {a:<10} {done:>4} / {len(mine)}")
    both = sum(1 for _, it in items.iterrows() if all(it["id"] in ann[a] for a in it["annotators"]))
    adj = gio.needs_adjudication(items, ann)
    settled = gio.load_adjudicated()
    print(f"  both annotators done: {both} / {len(items)}")
    print(f"  disagreements: {len(adj)}  adjudicated: {sum(1 for i in adj if i in settled)}")


def _kappa(a: list, b: list) -> float | None:
    from sklearn.metrics import cohen_kappa_score

    if len(a) < 2 or len(set(a) | set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b))


def _word_vector(item_words: list[str], rec: dict) -> list[int]:
    flagged = set(rec["flagged"]) if rec["verdict"] == "hallucinated" else set()
    return [int(i in flagged) for i in range(len(item_words))]


def agreement() -> dict:
    items = gio.load_items()
    ann = gio.load_all_annotations()
    verdict_a, verdict_b, word_a, word_b = [], [], [], []
    per_pair: dict[str, dict[str, list]] = {}
    for _, it in items.iterrows():
        x, y = it["annotators"]
        ra, rb = ann[x].get(it["id"]), ann[y].get(it["id"])
        if not (ra and rb):
            continue
        key = f"{x}+{y}"
        pp = per_pair.setdefault(key, {"a": [], "b": []})
        pp["a"].append(ra["verdict"])
        pp["b"].append(rb["verdict"])
        verdict_a.append(ra["verdict"])
        verdict_b.append(rb["verdict"])
        word_a += _word_vector(it["words"], ra)
        word_b += _word_vector(it["words"], rb)

    n = len(verdict_a)
    if n == 0:
        print("[gold] no item has both annotations yet")
        return {}
    raw = sum(p == q for p, q in zip(verdict_a, verdict_b)) / n
    binary = [(p, q) for p, q in zip(verdict_a, verdict_b) if "unsure" not in (p, q)]
    res = {
        "n_items": n,
        "verdict_raw_agreement": raw,
        "verdict_kappa_3way": _kappa(verdict_a, verdict_b),
        "verdict_kappa_binary": _kappa([p for p, _ in binary], [q for _, q in binary]),
        "word_kappa": _kappa(word_a, word_b),
        "word_f1_between_annotators": ev.word_level_metrics(
            [ev.Record("all", ["w"] * len(word_a), word_a, word_b, 0)]
        ).f1
        if word_a
        else None,
        "per_pair": {
            k: {"n": len(v["a"]), "kappa": _kappa(v["a"], v["b"])} for k, v in per_pair.items()
        },
        "needs_adjudication": len(gio.needs_adjudication(items, ann)),
    }
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    print(f"[gold] items with both annotations: {n}")
    print(f"  verdict agreement {fmt(raw)}   kappa 3-way {fmt(res['verdict_kappa_3way'])}   "
          f"kappa supported/hallucinated {fmt(res['verdict_kappa_binary'])}")
    print(f"  word-level kappa {fmt(res['word_kappa'])}   word F1 between annotators {fmt(res['word_f1_between_annotators'])}")
    for k, v in res["per_pair"].items():
        print(f"    {k:<18} n={v['n']:<4} kappa {fmt(v['kappa'])}")
    print(f"  disagreements awaiting adjudication: {res['needs_adjudication']}")
    return res


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def final_labels() -> tuple[dict[str, dict], dict]:
    """Human gold per item: the pair's shared answer, else the adjudicated one.

    Items still pending (missing annotations or unadjudicated disagreement)
    and items judged "unsure" are left out and counted.
    """
    items = gio.load_items()
    ann = gio.load_all_annotations()
    settled = gio.load_adjudicated()
    gold, counts = {}, {"agreed": 0, "adjudicated": 0, "pending": 0, "unsure": 0}
    for _, it in items.iterrows():
        recs = [ann[a].get(it["id"]) for a in it["annotators"]]
        if it["id"] in settled:
            rec, how = settled[it["id"]], "adjudicated"
        elif all(recs) and gio.pair_agrees(recs[0], recs[1]):
            rec, how = recs[0], "agreed"
        else:
            counts["pending"] += 1
            continue
        if rec["verdict"] == "unsure":
            counts["unsure"] += 1
            continue
        counts[how] += 1
        gold[it["id"]] = {
            "label": int(rec["verdict"] == "hallucinated"),
            "word_labels": _word_vector(it["words"], rec),
        }
    return gold, counts


def evaluate(checkpoints: list[str], judges: list[str]) -> dict:
    from .baselines import LexicalOverlapBaseline
    from .data import load_dataframe
    from .llm_judge import JUDGE_DIR, parse_spans, spans_to_word_labels

    schema = cfg.COLUMNS
    gold, counts = final_labels()
    print(f"[gold] usable items {len(gold)}  {counts}")
    if not gold:
        raise SystemExit("no finished gold items yet")
    df, report = load_dataframe(verbose=False)
    rows = df[df[schema.id].isin(gold)].set_index(schema.id)

    def records(pred_fn, score_fn=None) -> list[ev.Record]:
        out = []
        for item_id, g in gold.items():
            r = rows.loc[item_id]
            words = list(r["_words"])
            preds = pred_fn(item_id, r, words)
            out.append(
                ev.Record(
                    id=item_id,
                    words=words,
                    gold_word_labels=g["word_labels"],
                    pred_word_labels=preds,
                    gold_label=g["label"],
                    score=score_fn(item_id, preds) if score_fn else float(any(preds)),
                    hallucination_type=r[schema.hallucination_type],
                    domain=r[schema.domain],
                )
            )
        return out

    reports = []
    # The generator's own labels, scored as if they were a system.
    reports.append(
        ev.full_report(records(lambda i, r, w: [l for _, l in r["_tokens"]]), "generator labels")
    )
    bl = LexicalOverlapBaseline(mode="morph")
    reports.append(
        ev.full_report(records(lambda i, r, w: bl.predict_words(r[schema.context], w)), bl.name)
    )

    for name in judges:
        path = JUDGE_DIR / f"{name}.jsonl"
        if not path.exists():
            print(f"[gold] skip judge {name}: no cache at {path}")
            continue
        cache = {json.loads(l)["id"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}

        def judge_pred(i, r, w, cache=cache):
            spans, _ = parse_spans(cache.get(i, {}).get("content", ""))
            return spans_to_word_labels(w, spans, report.word_scheme)[0]

        reports.append(ev.full_report(records(judge_pred), f"judge:{name}"))

    if checkpoints:
        import sys

        from app._loader import load_bundle
        from .model import predict_one

        if str(gio.ROOT) not in sys.path:
            sys.path.insert(0, str(gio.ROOT))
        groups: dict[str, list[dict]] = {}
        for ck in checkpoints:
            path = cfg.CHECKPOINT_DIR / ck
            if not path.exists():
                print(f"[gold] skip checkpoint {ck}: not found")
                continue
            b = load_bundle(path, device="cpu")
            probs: dict[str, list[float]] = {}

            def model_pred(i, r, w, b=b, probs=probs):
                out = predict_one(
                    b.model, b.tokenizer, r[schema.question], r[schema.context], r[schema.answer],
                    max_length=b.max_length, word_scheme=b.word_scheme,
                )
                if out["words"] != w:
                    raise RuntimeError(f"{i}: inference words differ from the gold words")
                probs[i] = out["word_probs"]
                return out["word_labels"]

            rep = ev.full_report(records(model_pred, lambda i, p, probs=probs: max(probs[i])), ck)
            reports.append(rep)
            groups.setdefault(ck.rsplit("-s", 1)[0], []).append(rep)

    print()
    print(ev.format_table(reports))
    print("\nper hallucination type, word F1 (types from the generator):")
    for rep in reports:
        print(f"  {rep['name']:<34} " + "  ".join(f"{t} {m['word_f1']:.2f}" for t, m in rep["per_type"].items() if t != "none"))

    summary = {}
    if checkpoints:
        print("\nmean ± std over seeds:")
        for g, reps in groups.items():
            w = [r["word_level"]["f1"] for r in reps]
            s = [r["span_level"]["f1"] for r in reps]
            fmt = lambda v: f"{statistics.fmean(v):.3f}" + (f" ± {statistics.stdev(v):.3f}" if len(v) > 1 else "")  # noqa: E731
            summary[g] = {"word_f1": w, "span_f1": s}
            print(f"  {g:<28} word F1 {fmt(w)}   span F1 {fmt(s)}   (n={len(reps)})")

    result = {"counts": counts, "n": len(gold), "reports": reports, "seed_summary": summary}
    out = cfg.METRICS_DIR / "gold_eval.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[gold] -> {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--force", action="store_true")
    sub.add_parser("status")
    sub.add_parser("agree")
    se = sub.add_parser("evaluate")
    se.add_argument(
        "--checkpoints",
        nargs="*",
        default=[f"banglabert-{m}-e10-s{s}" for m in ("full", "lora") for s in (42, 43, 44)],
    )
    se.add_argument("--judges", nargs="*", default=["gemma-4-E4B-q4", "gemma-4-12B-qat-q4"])
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.force)
    elif args.cmd == "status":
        status()
    elif args.cmd == "agree":
        agreement()
    else:
        evaluate(args.checkpoints, args.judges)


if __name__ == "__main__":
    main()
