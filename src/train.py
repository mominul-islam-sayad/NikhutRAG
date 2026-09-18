"""Fine-tune the token classifier, full or LoRA, and report against the baseline.

A plain PyTorch loop rather than HF Trainer: the job is ~130 steps an epoch, the
loss needs a class weight over answer tokens only, and a visible loop is easier
to defend than Trainer's defaults.

Every run ends by printing the results table with the lexical baseline in it.
CLAUDE.md requires that, and for good reason -- the morphology-aware baseline
scores example-level F1 ~0.94 on the sample, so a model row without a baseline
row next to it says nothing about whether the model learned anything.

    python -m src.train                      # full fine-tuning
    python -m src.train --lora               # LoRA
    python -m src.train --backbone jhu-clsp/mmBERT-base --lora
"""

from __future__ import annotations

import argparse
import json
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
from .model import WeightedTokenLoss, attach_lora, build_model, predict_records


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

        return psutil.Process().memory_info().rss / (1024**2)
    except Exception:  # noqa: BLE001
        return float("nan")


def evaluate_split(model, dataset, tokenizer, device, name: str) -> dict:
    records = predict_records(model, dataset, tokenizer, device=device)
    return ev.full_report(records, name)


def train(args: argparse.Namespace) -> dict:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = cfg.TRAIN
    train_cfg.backbone = args.backbone
    train_cfg.use_lora = args.lora
    train_cfg.epochs = args.epochs
    if args.positive_weight is not None:
        train_cfg.positive_class_weight = args.positive_weight

    set_seed(args.seed)
    run_name = f"{args.backbone.split('/')[-1]}-{'lora' if args.lora else 'full'}"
    print(f"[train] run={run_name}  device={device}  machine={platform.processor() or platform.machine()}")

    datasets, splits, report, max_length = build_datasets(backbone=args.backbone, verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    model = build_model(args.backbone)
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
    total_steps = len(train_loader) * train_cfg.epochs
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=train_cfg.weight_decay
    )
    sched = get_linear_schedule_with_warmup(
        optim, int(total_steps * train_cfg.warmup_ratio), total_steps
    )
    loss_fn = WeightedTokenLoss(train_cfg.positive_class_weight)

    print(f"[train] lr={lr}  epochs={train_cfg.epochs}  steps/epoch={len(train_loader)}  total={total_steps}")

    best_score, best_epoch, best_state = -1.0, -1, None
    history: list[dict] = []
    t0 = time.perf_counter()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], train_cfg.max_grad_norm
            )
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.item()
            n_batches += 1

        val = evaluate_split(model, datasets["val"], tokenizer, device, f"{run_name}/val")
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
            f"val word_F1={score:.4f}  val ex_F1={val['example_level']['f1']:.4f}"
        )

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= train_cfg.early_stopping_patience:
            print(f"[train] early stopping at epoch {epoch} (best was {best_epoch})")
            break

    wall = time.perf_counter() - t0
    mem = peak_memory_mb(device)
    print(f"[train] done in {wall:.1f}s  peak_mem={mem:.0f}MB  best epoch={best_epoch}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final evaluation, always alongside the baseline --------------------
    test_report = evaluate_split(model, datasets["test"], tokenizer, device, run_name)
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
        one = datasets["test"][0]
        one = {k: torch.tensor([v], device=device) for k, v in one.items() if k != "labels"}
        model.eval()

        def _call():
            with torch.no_grad():
                model(**one)

        lat = ev.latency_benchmark(_call, device=device)
        print(
            f"\n[latency] {lat['median_ms']:.1f} ms median, {lat['p95_ms']:.1f} ms p95 "
            f"on {device} at batch size 1 -- under 200ms: {lat['meets_200ms']}"
        )

    result = {
        "run": run_name,
        "backbone": args.backbone,
        "use_lora": args.lora,
        "device": device,
        "max_length": max_length,
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
        },
    }

    out = cfg.METRICS_DIR / f"{run_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[train] metrics -> {out}")

    if args.save_model:
        ckpt = cfg.CHECKPOINT_DIR / run_name
        ckpt.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--positive-weight", type=float, default=None)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
