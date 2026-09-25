"""Re-measure inference latency of saved checkpoints, CPU and GPU separately.

The paper's claim is <200ms per example on consumer hardware, so latency is
measured here on this machine only, at batch size 1, after warmup, cycling
through the same real test examples for every model. Sequence length drives the
cost, so a single example is not a fair stand-in.

    python -m src.latency banglabert-full banglabert-lora --device cpu
    $env:HF_HUB_OFFLINE=1
    .venv-dml\\Scripts\\python.exe -m src.latency mmBERT-base-lora --device dml

Results merge into outputs/metrics/latency.json keyed by run and device. Run it
with nothing else using the CPU or GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

from . import config as cfg
from . import evaluate as ev
from .data import build_datasets

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Same count src/train.py uses, so the two agree.
N_EXAMPLES = 30
OUT = cfg.METRICS_DIR / "latency.json"


def measure(run: str, device_name: str, n_examples: int = N_EXAMPLES) -> dict:
    from app._loader import load_bundle

    from .train import resolve_device

    ckpt = cfg.CHECKPOINT_DIR / run
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}; train with --save-model first")

    bundle = load_bundle(ckpt, device="cpu")
    device = resolve_device(device_name)
    model = bundle.model.to(device).eval()

    datasets, _, _, max_length = build_datasets(backbone=bundle.backbone, verbose=False)
    test = datasets["test"]
    n = min(n_examples, len(test))
    inputs = [
        {k: torch.tensor([v], device=device) for k, v in test[j].items() if k != "labels"}
        for j in range(n)
    ]
    cursor = [0]

    def _call():
        one = inputs[cursor[0] % n]
        cursor[0] += 1
        with torch.no_grad():
            out = model(**one).logits
            if device_name == "dml":
                out.cpu()  # DirectML is async; wait for the result

    lat = ev.latency_benchmark(_call, device=device_name)
    lat.update(
        {
            "run": run,
            "backbone": bundle.backbone,
            "is_lora": bundle.is_lora,
            "max_length": max_length,
            "n_examples": n,
            "mean_input_tokens": sum(x["input_ids"].shape[1] for x in inputs) / n,
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "machine": platform.processor() or platform.machine(),
        }
    )
    return lat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="checkpoint names under outputs/checkpoints")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | dml")
    args = ap.parse_args()

    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for run in args.runs:
        lat = measure(run, args.device)
        results.setdefault(run, {})[args.device] = lat
        print(
            f"[latency] {run:<22} {args.device:<4} median {lat['median_ms']:6.1f} ms  "
            f"p95 {lat['p95_ms']:6.1f} ms  ({lat['n_examples']} examples, "
            f"{lat['mean_input_tokens']:.0f} tokens avg)  under 200ms: {lat['meets_200ms']}",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[latency] -> {OUT}")


if __name__ == "__main__":
    main()
