"""Fine-tune the token classifier, full or LoRA, and report against the baseline.

A plain PyTorch loop rather than HF Trainer: the job is ~175 steps an epoch, the
loss needs a class weight over answer tokens only, and a visible loop is easier
to defend than Trainer's defaults.

Every run ends by printing the results table with the lexical baseline in it.
CLAUDE.md requires that, and for good reason -- the morphology-aware baseline
scored example-level F1 ~0.94 on the first sample, so a model row without a
baseline row next to it says nothing about whether the model learned anything.

    python -m src.train                      # full fine-tuning
    python -m src.train --lora               # LoRA
    python -m src.train --backbone jhu-clsp/mmBERT-base --lora
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from . import config as cfg
from . import evaluate as ev
from .baselines import LexicalOverlapBaseline
from .data import Collator, build_datasets
from .model import WeightedTokenLoss, attach_lora, build_model, decompose_layernorms, predict_records


#: Latency cycles through this many test examples at batch size 1.
LATENCY_EXAMPLES = 30


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def peak_memory_mb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / (1024**2)
    try:
        import psutil

        info = psutil.Process().memory_info()
        # rss is the *current* footprint at the end of the run, not the peak.
        # Windows tracks the true peak working set; Linux exposes it via
        # resource.getrusage, which reports kilobytes.
        if hasattr(info, "peak_wset"):
            return info.peak_wset / (1024**2)
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:  # noqa: BLE001
        return float("nan")


def resolve_device(name: str):
    """Map a device name to what ``.to()`` accepts.

    "dml" is DirectML (torch-directml), the only PyTorch GPU route for an AMD
    RDNA2 card such as the RX 6600 on Windows -- ROCm does not support it. It
    needs its own venv (.venv-dml) because torch-directml pins an older torch.
    """
    if name == "dml":
        try:
            import torch_directml
        except ImportError as exc:
            raise SystemExit("--device dml needs torch-directml; run it with the .venv-dml python") from exc
        return torch_directml.device()
    return torch.device(name)


def evaluate_split(model, dataset, tokenizer, device, name: str, batch_size: int = 32) -> dict:
    records = predict_records(model, dataset, tokenizer, device=device, batch_size=batch_size)
    return ev.full_report(records, name)


def train(args: argparse.Namespace) -> dict:
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = resolve_device(device_name)
    train_cfg = cfg.TRAIN
    train_cfg.backbone = args.backbone
    train_cfg.use_lora = args.lora
    train_cfg.epochs = args.epochs
    if args.positive_weight is not None:
        train_cfg.positive_class_weight = args.positive_weight
    if args.batch_size is not None:
        train_cfg.train_batch_size = args.batch_size
    if args.grad_accum is not None:
        train_cfg.grad_accum_steps = args.grad_accum
    if args.grad_checkpointing:
        train_cfg.gradient_checkpointing = True
    if args.eval_batch_size is not None:
        train_cfg.eval_batch_size = args.eval_batch_size

    set_seed(args.seed)
    run_name = f"{args.backbone.split('/')[-1]}-{'lora' if args.lora else 'full'}"
    if args.limit_rows:
        run_name += "-smoke"
    print(f"[train] run={run_name}  device={device_name}  machine={platform.processor() or platform.machine()}")

    datasets, splits, report, max_length = build_datasets(backbone=args.backbone, verbose=True)
    if args.limit_rows:
        # Smoke test only: a code-path check, never a result.
        for name, ds in datasets.items():
            ds.features, ds.meta = ds.features[: args.limit_rows], ds.meta[: args.limit_rows]
            splits[name] = splits[name].head(args.limit_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    model = build_model(args.backbone)
    if device_name == "dml":
        n = decompose_layernorms(model)
        print(f"[train] DirectML: {n} LayerNorms decomposed into plain ops (backward is unsupported)")
    if train_cfg.gradient_checkpointing:
        # Non-reentrant checkpointing works with a frozen (LoRA) backbone
        # without forcing requires_grad onto the inputs.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora:
        model = attach_lora(model, args.backbone, train_cfg)
    model.to(device)

    eff = ev.efficiency_stats(model)
    print(
        f"\n[train] params total={eff['total_params']:,} trainable={eff['trainable_params']:,} "
        f"({eff['trainable_pct']:.2f}%)  under_500M={eff['param_budget_500m_ok']}"
    )

    lr = train_cfg.lora_learning_rate if args.lora else train_cfg.learning_rate
    collate = Collator(tokenizer)
    train_loader = DataLoader(
        datasets["train"], batch_size=train_cfg.train_batch_size, shuffle=True, collate_fn=collate
    )
    accum = max(1, train_cfg.grad_accum_steps)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * train_cfg.epochs
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=train_cfg.weight_decay
    )
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * train_cfg.warmup_ratio), total_steps
    )
    loss_fn = WeightedTokenLoss(train_cfg.positive_class_weight)

    print(
        f"[train] lr={lr}  epochs={train_cfg.epochs}  batch={train_cfg.train_batch_size}x{accum}  "
        f"steps/epoch={steps_per_epoch}  total={total_steps}  grad_ckpt={train_cfg.gradient_checkpointing}",
        flush=True,
    )

    best_score, best_epoch, best_state = -1.0, -1, None
    history: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        n_loader = len(train_loader)
        for i, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            # The last window of an epoch may hold fewer than `accum` batches.
            window = accum if i <= (n_loader // accum) * accum else n_loader % accum
            (loss / window).backward()
            running += loss.item()
            n_batches += 1
            if i % accum and i != n_loader:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], train_cfg.max_grad_norm
            )
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        val = evaluate_split(
            model, datasets["val"], tokenizer, device, f"{run_name}/val", train_cfg.eval_batch_size
        )
        score = val["word_level"]["f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(n_batches, 1),
                "val_word_f1": score,
                "val_example_f1": val["example_level"]["f1"],
                "val_auroc": val["example_level"]["auroc"],
            }
        )
        print(
            f"[train] epoch {epoch}  loss={running / max(n_batches, 1):.4f}  "
            f"val word_F1={score:.4f}  val ex_F1={val['example_level']['f1']:.4f}  "
            f"elapsed={time.perf_counter() - t0:.0f}s",
            flush=True,
        )

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= train_cfg.early_stopping_patience:
            print(f"[train] early stopping at epoch {epoch} (best was {best_epoch})")
            break

    wall = time.perf_counter() - t0
    mem = peak_memory_mb(device_name)
    print(f"[train] done in {wall:.1f}s  peak_mem={mem:.0f}MB  best epoch={best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final evaluation, always alongside the baseline --------------------
    test_report = evaluate_split(
        model, datasets["test"], tokenizer, device, run_name, train_cfg.eval_batch_size
    )
    baseline_reports = []
    for mode in ("exact", "morph"):
        bl = LexicalOverlapBaseline(mode=mode)
        baseline_reports.append(ev.full_report(bl.run(splits["test"]), bl.name))

    print("\n=== test results ===")
    print(ev.format_table(baseline_reports + [test_report]))

    print("\nper hallucination type:")
    for t, m in test_report["per_type"].items():
        r = m["example_recall"]
        print(
            f"  {t:<22} n={m['n']:<5} ex_recall={f'{r:.3f}' if r is not None else ' n/a'}  "
            f"word_F1={m['word_f1']:.3f}"
        )

    # ---- latency on this machine, never on cloud hardware -------------------
    lat = None
    if not args.skip_latency:
        # Cycle through real test examples: one example's length is not a fair
        # stand-in, since cost grows with sequence length.
        n_lat = min(LATENCY_EXAMPLES, len(datasets["test"]))
        inputs = [
            {k: torch.tensor([v], device=device) for k, v in datasets["test"][j].items() if k != "labels"}
            for j in range(n_lat)
        ]
        cursor = [0]
        model.eval()

        def _call():
            one = inputs[cursor[0] % n_lat]
            cursor[0] += 1
            with torch.no_grad():
                out = model(**one).logits
                if device_name == "dml":
                    # DirectML runs asynchronously; reading the result back is
                    # the only way to wait for it, or the timing is fiction.
                    out.cpu()

        lat = ev.latency_benchmark(_call, device=device_name)
        lat["n_examples"] = n_lat
        lat["mean_input_tokens"] = sum(x["input_ids"].shape[1] for x in inputs) / n_lat
        print(
            f"\n[latency] {lat['median_ms']:.1f} ms median, {lat['p95_ms']:.1f} ms p95 "
            f"on {device_name} at batch size 1 -- under 200ms: {lat['meets_200ms']}"
        )

    result = {
        "run": run_name,
        "backbone": args.backbone,
        "use_lora": args.lora,
        "device": device_name,
        "max_length": max_length,
        "word_scheme": report.word_scheme,
        "dataset": {
            "csv": cfg.CSV_PATH.name,
            "rows": report.n_rows,
            "groups": report.n_groups,
            "group_key": report.group_key,
            "split_rows": {k: len(v) for k, v in splits.items()},
        },
        "seed": args.seed,
        "config": asdict(train_cfg),
        "efficiency": eff,
        "train_seconds": wall,
        "peak_memory_mb": mem,
        "best_epoch": best_epoch,
        "history": history,
        "test": test_report,
        "baselines": baseline_reports,
        "latency": lat,
        "dataset_caveats": {
            "unlearnable_rows": len(report.label_says_hallucinated_tokens_say_clean),
            "span_token_mismatch": len(report.span_token_mismatch),
            "non_bangla_rows": len(report.non_bangla),
            "foreign_script_rows": len(report.foreign_script),
            "context_truncated_rows": len(report.context_truncated),
        },
    }

    out = cfg.METRICS_DIR / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[train] metrics -> {out}")

    if args.save_model:
        ckpt = cfg.CHECKPOINT_DIR / run_name
        ckpt.mkdir(parents=True, exist_ok=True)
        # DirectML tensors cannot be serialized (no accessible storage), so
        # save from the CPU. Harmless everywhere else.
        model.to("cpu")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"[train] checkpoint -> {ckpt}")

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default=cfg.BACKBONE)
    ap.add_argument("--lora", action="store_true", help="PEFT/LoRA instead of full fine-tuning")
    ap.add_argument("--epochs", type=int, default=cfg.TRAIN.epochs)
    ap.add_argument("--seed", type=int, default=cfg.TRAIN.seed)
    ap.add_argument("--device", default=None, help="cpu | cuda | dml (default: auto)")
    ap.add_argument("--positive-weight", type=float, default=None)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None, help="per-step batch (default: config)")
    ap.add_argument("--grad-accum", type=int, default=None, help="batches per optimizer step")
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--grad-checkpointing", action="store_true", help="trade ~30%% speed for memory")
    ap.add_argument(
        "--limit-rows", type=int, default=0, help="smoke test: N rows per split, run name gets -smoke"
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
