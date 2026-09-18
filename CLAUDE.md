# FYDP-II — Bangla RAG Hallucination Detection

Context file for Claude Code. Read this before writing any code.

## What this project is

**"An Efficient and Lightweight Hallucination Detection Framework for RAG Using SLMs"**
United International University, CSE, Group B2.
Team: Mehrin Sultana, Mominul Islam, Nazmul Alam, Md. Mainul Hossain Fahim Chowdhury.
Supervisor: Khushnur Binte Jahangir. Co-supervisor: Nahid Hossain.

FYDP-I delivered the paper, poster and design. FYDP-II is the implementation.

**The method:** encode `context + question + answer` jointly in an encoder-only small
language model, run **token-level binary classification over the answer tokens only**
(0 = grounded in context, 1 = hallucinated), then aggregate consecutive positive tokens
into highlighted spans shown to the user. This is deliberately *not* LLM-as-a-judge —
the whole contribution is that it's small, local, fast and free.

## Hard constraints (these are commitments made in FYDP-I, not preferences)

- Model **under 500M parameters**
- Inference **under 200ms** per example
- **LoRA / PEFT** fine-tuning must be used and measured
- Runs **locally** on consumer hardware — no API calls, no cloud inference
- Token-level output, not just a sequence-level true/false
- Target **F1 > 75%** (see the baseline warning below — this target is too low)
- Final deliverable includes a **Streamlit UI** demo

## The dataset

`data/bangla_rag_hallucination_8k.csv` — the filename says 8k, the reality is smaller.

| Property | Value |
|---|---|
| Rows | 2,652 |
| Unique context+question pairs | **1,326** |
| Structure | every pair has a faithful `_F` row and a hallucinated `_H` row |
| Language | 2,648 Bangla, 4 English (in `Bangladesh Affairs`) |
| Class balance | exactly 50/50 |
| Context length | median 25 words, max 56 |
| Answer length | median 3 words, mean 4.7, max 29 |
| Human verified | **0%** — every row is `auto_generated` |

Domains: Agriculture 1000, History 970, Government Services 638, Bangladesh Affairs 44.

Hallucination types: `number_error` 480, `entity_replacement` 389, `date_error` 251,
`fabricated_fact` 173, `contradiction` 33, `none` 1326.

Columns: `id, domain, context, question, answer, label, hallucination_type,
hallucinated_span, token_labels, is_human_verified, verification_status`.

`token_labels` is a JSON list of `[word, 0|1]` pairs, one per whitespace token of
`answer`. It parses cleanly and the length matches the answer's whitespace token count
in **all 2,652 rows**. `hallucinated_span` is a literal substring of `answer` in 1,325
of 1,326 hallucinated rows.

## Two things that must not be gotten wrong

### 1. Split leakage

Each context appears exactly twice — once faithful, once hallucinated. A random row-level
train/test split puts the same context on both sides, and the reported F1 becomes
meaningless.

**Always group-split on the base ID.** Derive it by stripping the `_F` / `_H` suffix
(`BD_HIS_01519_F` → `BD_HIS_01519`) and use `GroupShuffleSplit` or `StratifiedGroupKFold`
on that. Both twins of a pair land on the same side, always. Suggested 70/15/15.

Also consider stratifying by `domain` — Bangladesh Affairs has only 44 rows and will
otherwise vanish from a split.

### 2. A trivial baseline already scores F1 = 0.900

Flagging an answer as hallucinated whenever any of its words is absent from the context —
no model, no training — gives:

```
Accuracy 0.893   Precision 0.840   Recall 0.971   F1 0.900
```

This means the stated target of F1 > 75% sits *below* a five-line string comparison, and
a fine-tuned SLM will score ~0.95 without proving anything. This is the single biggest
threat to the project at defense.

Required response:
- Implement this lexical baseline in `src/baselines.py` and **report it in every results
  table**. Hiding it is worse than the weakness itself.
- Report token-level and span-level metrics too, where the baseline is much weaker than
  it is at example-level.
- Fix the dataset (below).

## Known dataset weaknesses and the planned fix

- **Answers are too short** (median 3 words) for "token-level" to be meaningful. RAGTruth
  answers are paragraphs; that's what makes span localization a real task.
- **Corruptions are lexically obvious** — swapped entities and numbers that don't appear
  in the context, which is exactly what the naive baseline catches.
- **Nothing is human verified.** The FYDP-I presentation explicitly promises expert
  annotation as a contribution.

Planned regeneration, to be done alongside model development:
1. Multi-sentence answers (3–5 sentences) where only **one clause** is corrupted, so the
   rest of the answer is grounded and the model must localize.
2. Corruptions that **reuse vocabulary present elsewhere in the context**, so lexical
   overlap cannot detect them.
3. A **human-verified gold test set of ~200 examples**, annotated by the team.
4. A second test set at a **realistic class ratio** (~15–20% hallucinated, matching
   RAGTruth) alongside the balanced one. Report both.

## Model choice

**ModernBERT is English-only.** Its tokenizer fragments Bangla badly. The FYDP-I paper
names ModernBERT and DeBERTa; that plan does not transfer to Bengali as written, and the
paper text needs updating to say so.

Backbones, in priority order:

1. `csebuetnlp/banglabert` — ELECTRA-base, ~110M params. Bangla-specific, strongest on
   Bangla benchmarks, comfortably inside the size and latency budget. **Primary.**
2. `microsoft/mdeberta-v3-base` — ~278M. Multilingual, and keeps the DeBERTa thread from
   the FYDP-I paper intact. **Main comparison.**
3. `xlm-roberta-base` (278M) and `google/muril-base-cased` (~238M) — additional points for
   the comparison table.
4. `answerdotai/ModernBERT-base` — **only** for an optional English RAGTruth arm, never
   for Bangla.

Write **one training script driven by a config**, so the backbone is a string in
`src/config.py` and all four can be run without editing code.

## Input encoding and labels

- Sequence: `[CLS] question [SEP] context [SEP] answer [SEP]`, `max_length=256`
  (median context is 25 words — 256 is generous, do not use 512 and pay for padding).
- Tokenize the **answer with `is_split_into_words=True`** on its whitespace tokens, so
  `word_ids()` maps subwords straight back to the `token_labels` list. This is the clean
  way to align; do not try to reconstruct alignment from character offsets.
- Label **only answer subword tokens**. Question, context, and all special tokens get
  `-100` so they're ignored by the loss.
- For a word split into several subwords, label the first subword and set the rest to
  `-100` (report the choice; the alternative of labelling all subwords is also defensible
  but changes the token-level metrics).
- Example-level score = **max** over answer token hallucination probabilities.
- Span extraction = merge runs of consecutive tokens above threshold 0.5.

## Evaluation protocol

Every results table must report:

- **Token-level**: precision / recall / F1 on the hallucinated class, aggregated to
  *word* level (that's what the UI highlights, so it's the honest unit)
- **Example-level**: precision / recall / F1, accuracy, **AUROC**
- **Span-level**: exact and partial match against `hallucinated_span`
- **Per hallucination type**: breakdown across the five types — `contradiction` has only
  33 examples and will likely be the weak spot, which is worth saying out loud
- **Latency**: ms per example at batch size 1, CPU and GPU separately, after ~20 warmup
  runs, measured on the actual local machine — not on Colab. The sub-200ms claim is about
  consumer hardware, so it has to be measured on consumer hardware.
- **Efficiency**: parameter count, trainable parameter count under LoRA, peak memory,
  training wall-clock

Compare against: the lexical baseline, full fine-tuning vs LoRA, and at least one
LLM-as-a-judge reference point for the cost argument.

Note on LoRA: at 110M parameters full fine-tuning is cheap, so LoRA is not strictly
necessary here — but it's a claim in the paper, so run both and report the delta in F1,
trainable params and training time. If LoRA loses accuracy for no meaningful saving at
this scale, say that; it's a legitimate finding.

## Repo layout

```
.
├── CLAUDE.md
├── requirements.txt
├── data/
│   └── bangla_rag_hallucination_8k.csv
├── src/
│   ├── config.py        # backbone, hyperparams, paths — the only file to edit per run
│   ├── data.py          # load, group split, tokenize + label alignment
│   ├── model.py         # backbone + token classification head, LoRA wiring
│   ├── train.py
│   ├── evaluate.py      # token / example / span metrics, AUROC, latency bench
│   └── baselines.py     # lexical string-match baseline
├── app/
│   └── streamlit_app.py
└── outputs/             # checkpoints, metrics JSON, plots
```

## Environment

Local, VS Code, CPU or laptop GPU. The job is small: ~2,100 training rows, seq len 256,
110M params, ~130 steps per epoch. Minutes on a GPU, well under an hour on CPU. No cloud
compute is needed for the primary model. Colab is an escape hatch for mDeBERTa sweeps
only.

## Don't

- Don't random-split the rows. Group on base ID.
- Don't report example-level F1 alone.
- Don't omit the lexical baseline from results tables.
- Don't use ModernBERT for the Bangla model.
- Don't benchmark latency on cloud hardware.
- Don't silently drop the 4 English rows — either keep them and note it, or drop them and
  note it.
- Don't pad to 512 "just in case".
