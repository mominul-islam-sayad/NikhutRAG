"""Backbone + token classification head, LoRA wiring, and inference.

Two labels per answer token: 0 grounded, 1 hallucinated. Everything that is
not an answer token carries IGNORE_INDEX and contributes nothing to the loss.

The LoRA target module names in config.BACKBONES are architecture-specific and
easy to get silently wrong -- peft will happily attach to nothing and report a
tiny trainable count that looks like a success. `attach_lora` therefore
verifies the targets actually matched and fails loudly with the real module
names if they did not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from . import config as cfg
from . import evaluate as ev

LABEL_NAMES = {0: "grounded", 1: "hallucinated"}


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def build_model(backbone: str | None = None, num_labels: int = 2):
    backbone = backbone or cfg.BACKBONE
    conf = AutoConfig.from_pretrained(backbone, num_labels=num_labels)
    conf.id2label = LABEL_NAMES
    conf.label2id = {v: k for k, v in LABEL_NAMES.items()}
    model = AutoModelForTokenClassification.from_pretrained(backbone, config=conf)
    return model


class DecomposedLayerNorm(nn.Module):
    """LayerNorm written as plain tensor ops, numerically the same function.

    DirectML (the only GPU route for an AMD RX 6600 on Windows) cannot run the
    backward pass of a bias-free ``nn.LayerNorm`` -- the op fails to build or
    takes the device down with it. ModernBERT/mmBERT uses bias-free norms
    throughout (``norm_bias: false``), so it cannot be trained there without
    this. Parameters keep their names, so checkpoints still load into a stock
    model with ``nn.LayerNorm``.
    """

    def __init__(self, ln: nn.LayerNorm):
        super().__init__()
        self.weight, self.bias, self.eps = ln.weight, ln.bias, ln.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - x.mean(-1, keepdim=True)
        y = xc * torch.rsqrt((xc * xc).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight
        return y + self.bias if self.bias is not None else y


def decompose_layernorms(module: nn.Module) -> int:
    """Swap every nn.LayerNorm under ``module`` for DecomposedLayerNorm."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, DecomposedLayerNorm(child))
            n += 1
        else:
            n += decompose_layernorms(child)
    return n


def _module_suffixes(model) -> set[str]:
    return {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}


def attach_lora(model, backbone: str | None = None, train: cfg.TrainConfig = cfg.TRAIN):
    """Wrap the model in a LoRA adapter, verifying the targets exist.

    At ~110M parameters full fine-tuning is already cheap, so LoRA is not
    strictly required here -- but it is a claim in the FYDP-I paper, so both
    are run and the delta in F1, trainable params and wall-clock is reported.
    If LoRA loses accuracy for no meaningful saving at this scale, that is a
    legitimate finding and should be stated.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    backbone = backbone or cfg.BACKBONE
    spec = cfg.backbone_spec(backbone)
    targets = list(spec.lora_targets) if spec.lora_targets else None

    if targets:
        available = _module_suffixes(model)
        missing = [t for t in targets if t not in available]
        if missing:
            attn_like = sorted(
                n for n in available if any(k in n.lower() for k in ("quer", "key", "val", "qkv", "attn", "proj"))
            )
            raise ValueError(
                f"LoRA targets {missing} do not exist in {backbone}. "
                f"Attention-like modules actually present: {attn_like}. "
                f"Fix lora_targets in config.BACKBONES."
            )

    peft_conf = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=train.lora_r,
        lora_alpha=train.lora_alpha,
        lora_dropout=train.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    peft_model = get_peft_model(model, peft_conf)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable == 0:
        raise ValueError(f"LoRA attached but nothing is trainable for {backbone}")
    return peft_model


@dataclass
class WeightedTokenLoss:
    """Cross-entropy over answer tokens only, optionally upweighting positives.

    Even in a hallucinated answer most words are grounded, so positives are the
    minority at token level even when the dataset is balanced at example level.
    """

    positive_weight: float = 1.0

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.positive_weight != 1.0:
            weight = torch.tensor([1.0, self.positive_weight], device=logits.device, dtype=logits.dtype)
        fn = nn.CrossEntropyLoss(weight=weight, ignore_index=cfg.IGNORE_INDEX)
        return fn(logits.view(-1, logits.size(-1)), labels.view(-1))


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


@torch.no_grad()
def predict_records(
    model,
    dataset,
    tokenizer,
    device: str = "cpu",
    threshold: float = cfg.SPAN_THRESHOLD,
    batch_size: int = 32,
) -> list[ev.Record]:
    """Run the model over an encoded dataset and produce evaluable Records.

    Labelled positions are exactly the first subword of each answer word, in
    order, so the probabilities read off those positions map one-to-one onto
    the answer's words. data.py asserts that correspondence holds (delta=0).
    """
    from torch.utils.data import DataLoader

    from .data import Collator

    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer))

    probs_per_example: list[list[float]] = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits.detach().cpu()
        p_halluc = torch.softmax(logits.float(), dim=-1)[..., 1]
        for i in range(p_halluc.size(0)):
            mask = labels[i] != cfg.IGNORE_INDEX
            probs_per_example.append(p_halluc[i][mask].tolist())

    records: list[ev.Record] = []
    for meta, probs in zip(dataset.meta, probs_per_example):
        words = meta["words"]
        if len(probs) != len(words):
            # Only possible if a word lost all subwords or was truncated away;
            # pad with 0.0 so the record stays aligned rather than crashing.
            probs = (probs + [0.0] * len(words))[: len(words)]
        preds = [int(p > threshold) for p in probs]
        records.append(
            ev.Record(
                id=meta["id"],
                words=words,
                gold_word_labels=meta["word_labels"],
                pred_word_labels=preds,
                gold_label=meta["label"],
                score=max(probs) if probs else 0.0,  # example score = max over tokens
                gold_span=meta.get("hallucinated_span"),
                hallucination_type=meta.get("hallucination_type"),
                domain=meta.get("domain"),
            )
        )
    return records


@torch.no_grad()
def predict_one(
    model,
    tokenizer,
    question: str,
    context: str,
    answer: str,
    device: str = "cpu",
    max_length: int = 256,
    threshold: float = cfg.SPAN_THRESHOLD,
    word_scheme: str = "regex",
) -> dict:
    """Single-example path used by the UI and the latency benchmark.

    ``word_scheme`` must be the one the checkpoint was trained with (saved in
    its metrics JSON), so the answer is cut into the same words as training.
    """
    from .data import encode_example, tokenize_answer

    words = tokenize_answer(answer, word_scheme)
    feat, _ = encode_example(question, context, words, [0] * len(words), tokenizer, max_length)
    labels = feat.pop("labels")
    batch = {k: torch.tensor([v], device=device) for k, v in feat.items()}

    model.eval().to(device)
    logits = model(**batch).logits.detach().cpu()[0]
    p = torch.softmax(logits.float(), dim=-1)[:, 1]

    mask = torch.tensor([l != cfg.IGNORE_INDEX for l in labels])
    probs = p[mask].tolist()
    probs = (probs + [0.0] * len(words))[: len(words)]
    preds = [int(x > threshold) for x in probs]

    return {
        "words": words,
        "word_probs": probs,
        "word_labels": preds,
        "spans": ev.extract_spans(words, preds),
        "score": max(probs) if probs else 0.0,
        "hallucinated": bool(max(probs) > threshold) if probs else False,
    }


def describe(model) -> dict:
    stats = ev.efficiency_stats(model)
    return stats


if __name__ == "__main__":
    import json

    backbone = cfg.BACKBONE
    print(f"[model] building {backbone}")
    m = build_model(backbone)
    print("[model] full fine-tuning:", json.dumps(describe(m), indent=2))

    print(f"\n[model] attaching LoRA (targets={cfg.backbone_spec(backbone).lora_targets})")
    lm = attach_lora(build_model(backbone), backbone)
    print("[model] LoRA:", json.dumps(describe(lm), indent=2))

    tok = AutoTokenizer.from_pretrained(backbone)
    out = predict_one(
        m,
        tok,
        question="বাংলাদেশের রাজধানী কোথায়?",
        context="বাংলাদেশের রাজধানী ঢাকা। এটি দেশের বৃহত্তম শহর।",
        answer="বাংলাদেশের রাজধানী চট্টগ্রাম।",
        max_length=128,
    )
    print("\n[model] untrained spot check (random head, values meaningless):")
    print(json.dumps({k: v for k, v in out.items() if k != "word_probs"}, ensure_ascii=False, indent=2))
