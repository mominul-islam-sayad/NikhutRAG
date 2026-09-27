"""Mean ± std across seeds, from the metrics JSONs that src/train.py writes.

    python -m src.summarize banglabert-full-e10 banglabert-lora-e10

Each argument is a run-name prefix; every outputs/metrics/<prefix>-s<seed>.json
is one seed. Reports the default threshold (0.5) and the threshold tuned on the
validation split, which was chosen on val and applied to test unseen.

A single seed is one draw. Quoting mean ± std over seeds is what makes a
difference between two systems defensible.
"""

from __future__ import annotations

import argparse
import json
import statistics

from . import config as cfg

METRICS = [
    ("word F1", ("word_level", "f1")),
    ("word P", ("word_level", "precision")),
    ("word R", ("word_level", "recall")),
    ("span F1", ("span_level", "f1")),
    ("span exact", ("span_level", "exact")),
    ("ex F1", ("example_level", "f1")),
    ("AUROC", ("example_level", "auroc")),
]


def _get(report: dict, path: tuple[str, str]) -> float:
    return float(report[path[0]][path[1]])


def _fmt(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:.3f}" if values else "n/a"
    return f"{statistics.fmean(values):.3f} ± {statistics.stdev(values):.3f}"


def summarize(prefix: str) -> dict:
    runs = sorted(cfg.METRICS_DIR.glob(f"{prefix}-s*.json"))
    runs = [r for r in runs if "smoke" not in r.name]
    if not runs:
        raise SystemExit(f"no metrics matching {prefix}-s*.json in {cfg.METRICS_DIR}")
    data = [json.loads(r.read_text(encoding="utf-8")) for r in runs]

    out: dict = {"prefix": prefix, "seeds": [d["seed"] for d in data], "n": len(data)}
    for key, label in (("test", "t=0.5"), ("test_tuned", "tuned")):
        out[label] = {
            name: [_get(d[key], path) for d in data if d.get(key)] for name, path in METRICS
        }
    out["thresholds"] = [d.get("threshold_tuned") for d in data]
    out["best_epochs"] = [d["best_epoch"] for d in data]
    out["train_seconds"] = [d["train_seconds"] for d in data]
    out["peak_memory_mb"] = [d["peak_memory_mb"] for d in data]
    out["per_type_word_f1"] = {
        t: [d["test"]["per_type"][t]["word_f1"] for d in data]
        for t in data[0]["test"]["per_type"]
        if t != "none"
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefixes", nargs="+")
    args = ap.parse_args()

    summaries = [summarize(p) for p in args.prefixes]
    names = [name for name, _ in METRICS]
    for label in ("t=0.5", "tuned"):
        print(f"\n=== test, threshold {label} (mean ± std over seeds) ===")
        print(f"{'run':<24}{'n':>3}  " + "".join(f"{n:>17}" for n in names))
        for s in summaries:
            print(f"{s['prefix']:<24}{s['n']:>3}  " + "".join(f"{_fmt(s[label][n]):>17}" for n in names))

    for s in summaries:
        print(f"\n{s['prefix']}: seeds {s['seeds']}  best epochs {s['best_epochs']}  thresholds {s['thresholds']}")
        print(
            f"  train {_fmt([x / 60 for x in s['train_seconds']])} min   "
            f"peak memory {_fmt(s['peak_memory_mb'])} MB (host RAM)"
        )
        print("  per type word F1 (t=0.5): " + ", ".join(f"{t} {_fmt(v)}" for t, v in s["per_type_word_f1"].items()))

    out = cfg.METRICS_DIR / "summary.json"
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[summarize] -> {out}")


if __name__ == "__main__":
    main()
