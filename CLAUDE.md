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

## The paper says RAGTruth, not Bangla — resolved 2026-09-18

`docs/Fydp1_final_paper.pdf` commits to the **English RAGTruth corpus** and does not
contain the words Bangla, Bengali, Bangladesh, Indic or multilingual anywhere in its 26
pages. RAGTruth is named 14+ times, and task allocation marks "Dataset Acquisition
(RAGTruth)" and "Preprocessing RAGTruth Data" as 100% complete.

**Decision: Bangla is primary, plus a small English RAGTruth arm.**

Reasons, in the order they matter:

- The paper names **Luna as "the main baseline"** and **LettuceDetect (F1 79.22% on
  RAGTruth)** as the system to beat. Neither exists in Bangla. Without a RAGTruth arm,
  every comparison point in the literature review becomes incomparable.
- The paper's own **gap analysis row 4** already states: *"Absence of bilingual illusion
  identification abilities. The majority of current datasets and detectors are
  exclusively in English."* Bangla delivers that stated gap. **Update that row to cite
  Bangla rather than Chinese** — it converts the pivot from a deviation into the
  contribution.
- The paper text must be revised to say all of this. It currently does not.

## Hard constraints (commitments from FYDP-I, not preferences)

- Model **under 500M parameters** — BanglaBERT measured at 110,028,290. Satisfied.
- Inference **under 200ms** per example. The paper's wording is stricter: "within
  100-200 milliseconds for each inference" (p.15). Measured 34.8ms median on CPU.
- **LoRA / PEFT** fine-tuning must be used and measured
- Runs **locally** on consumer hardware — no API calls, no cloud inference
- Token-level output, not just a sequence-level true/false
- **Streamlit UI** demo, **and a FastAPI + Uvicorn backend** — the paper's software
  stack (p.17) commits to both. Streamlit alone does not satisfy the paper.

## Headline metric: word-level and span-level F1 — decided 2026-09-18

The paper sets F1 > 75%. **That target is 19 points below a five-line string
comparison** (see below) and must be raised, and the unit must change.

- **Primary metrics: word-level F1 and span-level exact match.** These are what the UI
  highlights and what the paper's "Span Aggregation" requirement is about.
- **Example-level F1 is reported only with the lexical baseline beside it.** Alone it is
  misleading: on a 1-epoch smoke test the model beat the fair baseline by just +0.036 at
  example level but +0.241 at word level and +0.236 at span level.
- Set the new target against word/span F1 once a tuned run exists. Do not carry >75%
  forward as an example-level claim.

## The dataset

`data/bangla_rag_hallucination_8k.csv` is **a sample, not the final dataset.** The real
one is being built outside this repo. Everything below describes the sample and will
change — do not hardcode any of it. `src/data.py` validates and measures instead.

| Property | Value (sample) |
|---|---|
| Rows | 2,652 |
| Unique context+question pairs | 1,326 |
| Structure | every pair has a faithful `_F` row and a hallucinated `_H` row |
| Class balance | exactly 50/50 |
| Rows lacking Bengali script | **10**, not the 4 previously recorded |
| Human verified | 0% |

Domains: Agriculture 1000, History 970, Government Services 638, Bangladesh Affairs 44.

Hallucination types: `number_error` 480, `entity_replacement` 389, `date_error` 251,
`fabricated_fact` 173, `contradiction` 33, `none` 1326.

The 10 non-Bangla rows are 4 fully-English plus 6 with Bangla questions and Latin or
numeric answers (`1764`, `Dell Inspiron 15।`, `Ministry of Foreign Affairs`).
`config.DROP_NON_BANGLA` controls them and the count is printed on every load.

## Three things that must not be gotten wrong

### 1. Split leakage

Each context appears exactly twice. A random row-level split puts the same context on
both sides and the reported F1 becomes meaningless.

**Always group-split on the base ID** (`BD_HIS_01519_F` → `BD_HIS_01519`). Implemented
in `src/data.py:group_split`, which groups *and* stratifies by domain in one pass and
asserts the splits are disjoint. Plain `GroupShuffleSplit` would not keep Bangladesh
Affairs (44 rows) in every split; this does — 30/6/8.

### 2. The lexical baseline is stronger than the target

Flagging any answer word absent from the context — no model, no training — measured on
the full sample:

| mode | ex_P | ex_R | ex_F1 | ex_Acc | AUROC | word_F1 | span_exact |
|---|---|---|---|---|---|---|---|
| `exact` | 0.838 | 0.986 | **0.906** | 0.898 | 0.896 | 0.541 | 0.387 |
| `morph` | 0.915 | 0.964 | **0.939** | 0.937 | 0.942 | 0.581 | 0.391 |

`exact` is the variant previously recorded here as F1 0.900. It is **handicapped by
Bangla morphology** — `গ্রিসে` does not string-match `গ্রিস`. `morph` absorbs case
suffixes and the danda and is **the honest bar**. Report `morph`.

Both live in `src/baselines.py` and **must appear in every results table**;
`src/train.py` prints them automatically beside the model row.

Note where the baseline is weak: word F1 0.581 and span exact 0.391. That gap is the
contribution.

### 3. The generator mislabels inflected spans

**66 of 1,326 hallucinated rows (5.0%) carry `label=1` with every token label 0.** They
are unlearnable under max-over-tokens scoring and cap achievable recall. A further
**316 (23.8%)** have flagged words disagreeing with `hallucinated_span`.

Cause: the generator matches `hallucinated_span` to answer words by exact whitespace
equality, which Bangla agglutination and the attached danda defeat — span `বাংলাদেশ` vs
word `বাংলাদেশের`, span `দিল্লি` vs word `দিল্লিতে।`. 56 of the 66 are
`entity_replacement`, because entities take case endings.

**Fix this in the generator, not here:** match by substring/prefix per token, strip `।`
first, and assert every `label=1` row ends with ≥1 positive token. `src/data.py` reports
both counts on every load. `src/evaluate.py` excludes unusable rows from span scoring so
a data defect is not charged to the model.

## Planned dataset regeneration

1. Multi-sentence answers (3–5 sentences) where only **one clause** is corrupted.
2. Corruptions that **reuse vocabulary present elsewhere in the context**, so lexical
   overlap cannot detect them.
3. A **human-verified gold test set of ~200 examples**, annotated by the team.
4. A second test set at a **realistic class ratio**. Note: RAGTruth is **not** 15–20% —
   measured, **43.1%** of its 17,790 responses carry at least one hallucination span.
   Pick the target ratio from the deployment story being argued, not from RAGTruth, and
   report it alongside the balanced set.
5. Fix the span-matching bug above before regenerating.

Note: longer answers will push sequence length past the sample's p99 of 86. `max_length`
is measured automatically, but check `AUTO_LENGTH_CAP` and the backbone's position limit.

## Model choice

**English ModernBERT is not for Bangla** — its tokenizer fragments Bangla badly. Use it
**only** for the English RAGTruth arm.

Measured Bangla tokenizer fertility (subwords per word, lower is better), on 300 rows of
the actual dataset:

| backbone | params | vocab | fertility | positions |
|---|---|---|---|---|
| `csebuetnlp/banglabert` | 110M | 32k | **1.40** | 512 |
| `xlm-roberta-base` | 278M | 250k | 2.20 | 512 |
| `microsoft/mdeberta-v3-base` | 278M | 250k | 2.82 | 512 |
| `jhu-clsp/mmBERT-base` | 307M | 256k | 3.99 | **8192** |

1. `csebuetnlp/banglabert` — **primary.** Best fertility by ~2.9×. Its 512-position
   ceiling is the risk once contexts lengthen.
2. `jhu-clsp/mmBERT-base` — **main comparison.** This is the *multilingual* ModernBERT,
   not the English one, so it keeps the paper's ModernBERT thread intact honestly.
   Higher fertility costs sequence length and latency, but it cannot run out of context.
   Latency headroom exists: BanglaBERT measured 34.8ms against a 200ms budget.
   `AUTO_LENGTH_CAP` currently caps it at 512 — raise it to use its long context.
3. `microsoft/mdeberta-v3-base`, `xlm-roberta-base`, `google/muril-base-cased` —
   additional comparison points.
4. `answerdotai/ModernBERT-base` — English RAGTruth arm only, never for Bangla.

One training script driven by config: the backbone is a string in `src/config.py`.

## Input encoding and labels

- Sequence: `[CLS] question [SEP] context [SEP] answer [SEP]`
- **`max_length` is measured, not pinned.** `config.MAX_LENGTH = "auto"` covers the 99th
  percentile, rounded to a multiple of 32, capped by `AUTO_LENGTH_CAP` and the backbone's
  position limit. On the sample this selects **96** (p99 = 86). The previously pinned 256
  was ~2.7× over-padded and would have become wrong anyway once answers lengthen.
- Tokenize the **answer with `is_split_into_words=True`** on its whitespace tokens, so
  `word_ids()` maps subwords back to `token_labels`. Never reconstruct from char offsets.
- Label **only answer subword tokens**; question, context and specials get `-100`.
- Label the **first subword** of each word, rest `-100` (`LABEL_ALL_SUBWORDS=False`).
- **Truncation is `only_first`** — sacrifice context before the answer. Truncating the
  answer silently discards labels, which is unrecoverable.
- Example-level score = **max** over answer token hallucination probabilities.
- Span extraction = merge runs of consecutive tokens above threshold 0.5.

## Evaluation protocol

All implemented in `src/evaluate.py`, which scores every system through one code path.

- **Word-level** P/R/F1 on the hallucinated class — *headline*
- **Span-level** exact and partial match against `hallucinated_span` — *headline*
- **Example-level** P/R/F1, accuracy, AUROC — only with the baseline beside it
- **Per hallucination type** — `contradiction` (33 rows) is the expected weak spot and
  that should be said out loud
- **Latency**: ms per example at batch size 1, CPU and GPU separately, after ~20 warmups,
  on the actual local machine. Never on Colab — the claim is about consumer hardware.
- **Efficiency**: parameter count, trainable count under LoRA, peak memory, wall-clock

Compare against: the lexical baseline (both modes), full fine-tuning vs LoRA, and at
least one LLM-as-a-judge reference point for the cost argument. For the RAGTruth arm,
compare to Luna and LettuceDetect's published numbers.

### LoRA vs full fine-tuning — measured 2026-09-18

Both run, 4 epochs, BanglaBERT, CPU, identical splits and seed:

| | full | LoRA | delta |
|---|---|---|---|
| **word F1** | **0.857** | 0.831 | −0.026 |
| **span exact** | **0.667** | 0.661 | −0.006 |
| example F1 | 0.967 | **0.972** | +0.005 |
| AUROC | **0.997** | 0.992 | −0.005 |
| trainable params | 110,028,290 | **886,274** | 124× fewer |
| peak memory | 2,749 MB | **2,025 MB** | −26% |
| wall-clock | 1,123.9 s | **1,084.3 s** | −3.5% |
| latency (median) | **33.8 ms** | 37.2 ms | +3.4 ms |

**This is the legitimate finding the paper's LoRA claim needs.** At 110M, LoRA buys a
124× reduction in trainable parameters and 26% less memory, but **almost no training
time** — on CPU the backward pass still traverses the frozen backbone, so the saving
LoRA is famous for does not materialise at this scale. It costs 0.026 word F1, which is
the headline metric, while nudging example F1 up by 0.005 — a reminder that example-level
numbers are too coarse to rank these two.

Report this honestly rather than implying LoRA was necessary. It becomes genuinely
worthwhile on the larger backbones (mmBERT 307M, mDeBERTa 278M) and on GPU; re-measure
there before generalising from this row.

## Repo layout

```
.
├── CLAUDE.md
├── requirements.txt          # torch installed separately, CPU-only
├── .gitattributes
├── data/
├── docs/Fydp1_final_paper.pdf
├── src/
│   ├── config.py        # backbone, schema, hyperparams — the only file to edit per run
│   ├── data.py          # load, validate, group split, tokenize + label alignment
│   ├── model.py         # backbone + token classification head, LoRA wiring, inference
│   ├── train.py         # full vs LoRA; always prints the baseline beside the model
│   ├── evaluate.py      # word/example/span metrics, AUROC, latency, efficiency
│   └── baselines.py     # lexical string-match baseline, exact + morph
├── app/
│   ├── streamlit_app.py # not yet written
│   └── api.py           # FastAPI + Uvicorn, promised by the paper; not yet written
└── outputs/             # checkpoints, metrics JSON, plots (the paper's "Data Store D1")
```

## Environment

Local, VS Code, CPU. `.venv` holds CPU-only torch 2.14. ~2,100 training rows, 116 steps
per epoch, ~250s per epoch on CPU at 2.5GB peak. No cloud compute needed for the primary
model. Colab is an escape hatch for mDeBERTa/mmBERT sweeps only — never for latency.

## Don't

- Don't random-split the rows. Group on base ID.
- Don't report example-level F1 as the headline, or without the baseline beside it.
- Don't omit the lexical baseline from results tables, and report the `morph` mode.
- Don't use English ModernBERT for Bangla. mmBERT-base is a different model.
- Don't benchmark latency on cloud hardware, or while another job is using the CPU.
- Don't silently drop the 10 non-Bangla rows — keep and note, or drop and note.
- Don't hardcode the sample's dimensions; the real dataset is being built elsewhere.
- Don't quote smoke-test numbers as results.
