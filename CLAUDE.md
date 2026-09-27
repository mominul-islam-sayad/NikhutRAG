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
  100-200 milliseconds for each inference" (p.15). BanglaBERT measured **57.1 ms median on
  CPU and 11.5 ms on the RX 6600 GPU** on the 4,000-row dataset (30 test examples; see
  Results). mmBERT **misses the budget on CPU (210 ms)**.
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

`data/bangla_rag_halu_4000.csv` (committed 2026-09-25) replaced the first 2,652-row
sample `bangla_rag_hallucination_8k.csv`. It is still being built outside this repo and
may change again — do not hardcode any of it. `src/data.py` validates and measures
instead, and prints all of the checks below on every load.

| Property | Value (4,000-row dataset) |
|---|---|
| Rows | 4,000 (2,000 faithful, 2,000 hallucinated — exactly 50/50) |
| Contexts (`context_id`) | 943, with **2, 4 or 6 rows each** (230 / 369 / 344 contexts) |
| Structure | each context+question has one faithful and one hallucinated answer; every context has both labels |
| IDs | `BDA_0001`, `HIS_0050`, … — **no `_F`/`_H` suffix any more** |
| `token_labels` format | `[{"token": w, "label": l}, …]`, punctuation split off as its own token |
| Answer length | 15 words median, p95 27, max 51; **96% are a single sentence** |
| Context length | 71 words median, max 219 |
| Rows lacking Bengali script | 0 |
| Rows with foreign-script letters | 23 — 3 are real corruption inside an answer word (`কবিতայ`, `আওत`, `सार्वजनिक`), 20 are in contexts |
| Human verified | 0% |
| Source | Bangla Wikipedia 3,658, DGHS 162, National Portal 126, Ministry of Education 48, BANBEIS 6 |

Domains: 500 each of Bangladesh Affairs, History, Healthcare, Agriculture, Government
Services, Education, Universities, Science & Technology.

Hallucination types: `fabricated_info` 334, `entity_replacement` 334, `omission` 333,
`contradiction` 333, `number_error` 333, `date_error` 333, `none` 2000. Note `omission`
is new and `fabricated_fact` was renamed `fabricated_info`.

`config.DROP_NON_BANGLA` still exists but currently affects no rows.

## Three things that must not be gotten wrong

### 1. Split leakage

Each context appears 2, 4 or 6 times, under different questions and with faithful and
hallucinated answers. A random row-level split puts the same context on both sides and
the reported F1 becomes meaningless.

**Always group-split on `context_id`** (`ColumnSchema.group`). Grouping on the ID alone
would now be **wrong**: IDs are unique per row, so the old `_F`/`_H` suffix rule would
leak. The loader falls back to the suffix rule only when the column is missing (old
sample, RAGTruth adapter), and it refuses to load if one context text appears under two
`context_id`s or one `context_id` holds two texts.

Implemented in `src/data.py:group_split`, which groups *and* stratifies by domain in one
pass and asserts the splits are disjoint. On the 4,000-row dataset: train 2,824 rows /
659 contexts, val 588 / 142, test 588 / 142, all eight domains in every split,
positive rate 0.500 in each.

### 2. The lexical baseline — always in the table

Flagging any answer word absent from the context — no model, no training. On the first
sample it scored example F1 0.939 (`morph`), above the paper's >75% target. On the
**4,000-row dataset it collapses**, because in 24% of hallucinated rows every flagged
word also appears verbatim somewhere in the context:

| mode | split | ex_P | ex_R | ex_F1 | ex_Acc | AUROC | word_F1 | span_F1 | span_exact | span_partial |
|---|---|---|---|---|---|---|---|---|---|---|
| `exact` | test (588) | 0.523 | 0.952 | 0.676 | 0.543 | 0.641 | 0.333 | 0.072 | 0.207 | 0.820 |
| `morph` | test (588) | 0.549 | 0.915 | **0.686** | 0.582 | 0.659 | **0.370** | 0.089 | 0.211 | 0.731 |
| `morph` | all (4,000) | 0.553 | 0.907 | 0.687 | 0.586 | 0.674 | 0.396 | 0.094 | 0.216 | 0.744 |

For comparison, the first sample (morph, all rows): ex_F1 0.939, word_F1 0.581.

`exact` is **handicapped by Bangla morphology** — `গ্রিসে` does not string-match `গ্রিস`.
`morph` absorbs case suffixes and is **the honest bar**. Report `morph`.

Both live in `src/baselines.py` and **must appear in every results table**;
`src/train.py` prints them automatically beside the model row.

Per type (morph, test): `omission` is nearly invisible to it (ex recall 0.587, word F1
0.113), as is `entity_replacement` at word level (0.198). `fabricated_info` is its easiest
case (word F1 0.756).

### 3. Words at inference must be cut exactly as the labels were

The labelled words are the tokens in `token_labels`, **not** `answer.split()`. The
4,000-row dataset splits punctuation (`।`, `,`, `(`, `-`, `%`) into its own tokens but
keeps numbers whole (`১০,০০০`, `২.৫`, `1.4.1.2`), and treats only Bengali and ASCII as
word characters. `config.ANSWER_TOKEN_PATTERN` reproduces its tokens for **all 4,000
rows**. `src/data.py:tokenize_answer` applies it.

The loader detects which scheme a dataset uses (`regex` or `whitespace`), refuses to load
if neither fits every row, and `src/train.py` saves it as `word_scheme` in the metrics
JSON. `app/_loader.py` reads it back, so the UI and API cut unseen answers the same way
the checkpoint was trained. Runs with no `word_scheme` are treated as `whitespace`.

**The first sample's generator bug is fixed in this dataset.** The sample had 66
hallucinated rows (5.0%) with no flagged token and 316 (23.8%) whose flagged words
disagreed with `hallucinated_span` — exact whitespace matching broke on inflection and
the attached danda. The 4,000-row dataset has **0** of either: flagged tokens match
`hallucinated_span` on every row once whitespace is ignored. `src/data.py` still reports
both counts on every load, in case a regeneration brings the bug back.

## Planned dataset regeneration

Status against the 4,000-row dataset:

1. Multi-sentence answers (3–5 sentences) where only **one clause** is corrupted —
   **not yet**: 96% of answers are still one sentence (15 words median).
2. Corruptions that **reuse vocabulary present elsewhere in the context**, so lexical
   overlap cannot detect them — **partly**: in 24% of hallucinated rows every flagged
   word appears verbatim in the context, and the baseline fell from 0.939 to 0.686 ex_F1.
3. A **human-verified gold test set of ~200 examples**, annotated by the team — **not yet**.
4. A second test set at a **realistic class ratio**. Note: RAGTruth is **not** 15–20% —
   measured, **43.1%** of its 17,790 responses carry at least one hallucination span.
   Pick the target ratio from the deployment story being argued, not from RAGTruth, and
   report it alongside the balanced set — **not yet**; the dataset is again exactly 50/50.
5. Fix the span-matching bug — **done** (0 unlearnable rows, 0 span mismatches).
6. Clean the 3 answers with foreign-script letters inside Bangla words (reported on load).

Sequence length already grew: p99 is 288 subwords (was 86), max 365, so `max_length`
auto-selects 288. That still fits BanglaBERT's 512 positions, but 3–5 sentence answers
could push it past; the loader reports truncated contexts when it happens.

## Model choice

**English ModernBERT is not for Bangla** — its tokenizer fragments Bangla badly. Use it
**only** for the English RAGTruth arm.

Measured Bangla tokenizer fertility (subwords per word, lower is better), on 300 rows of
the first sample (see mmBERT below for the 4,000-row figures):

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
   It turned out slower and not more accurate — see Results.

   **Prepared 2026-09-25 on the 4,000-row dataset.** Fertility there: answers 3.51 vs
   BanglaBERT 1.14, contexts 3.94 vs 1.36. Full sequences: median 383, p99 ~830, max
   977 subwords. At 512 it would truncate context on ~19% of rows, so `AUTO_LENGTH_CAP`
   was raised to 1024 (BanglaBERT is still held to 512 by its own position limit) and
   `max_length` auto-selects **832**. Word alignment checked on all 4,000 rows: 0
   misaligned. LoRA targets `Wqkv` verified: 1,082,882 trainable of 308.6M (0.35%).
   Weights cached locally: `pytorch_model.bin` (1.2GB), plus a converted `model.safetensors` for `.venv-dml`.

   **Run it on the GPU** (see Environment for the DirectML quirks). LoRA, batch 1 with
   accumulation to an effective 16:

   ```
   $env:HF_HUB_OFFLINE=1
   .venv-dml\Scripts\python.exe -m src.train --device dml --backbone jhu-clsp/mmBERT-base --lora --batch-size 1 --grad-accum 16 --eval-batch-size 1 --save-model
   ```

   On the CPU the same run would take several hours: batch 4 × accumulation 4 with
   `--grad-checkpointing`, since batch 16 at ~830 subwords does not fit 16GB RAM. Full
   fine-tuning is unlikely to fit the GPU's 8GB. AdamW state alone is ~4.9GB for 308M
   params, most of it for the 256k-token embedding matrix.

   Measured: 30 min on the GPU (7.5 min per epoch), reproducible to the last digit across
   two runs. Latency **210 ms on CPU (over budget)** and 53 ms on the GPU.
3. `microsoft/mdeberta-v3-base`, `xlm-roberta-base`, `google/muril-base-cased` —
   additional comparison points.
4. `answerdotai/ModernBERT-base` — English RAGTruth arm only, never for Bangla.

One training script driven by config: the backbone is a string in `src/config.py`.

## Input encoding and labels

- Sequence: `[CLS] question [SEP] context [SEP] answer [SEP]`
- **`max_length` is measured, not pinned.** `config.MAX_LENGTH = "auto"` covers the 99th
  percentile, rounded to a multiple of 32, capped by `AUTO_LENGTH_CAP` and the backbone's
  position limit. On the 4,000-row dataset this selects **288** (p99 = 288, median 135);
  on the first sample it selected 96.
- Tokenize the **answer with `is_split_into_words=True`** on the `token_labels` words
  (see "Words at inference" above), so `word_ids()` maps subwords back to the labels.
  Never reconstruct from char offsets.
- Label **only answer subword tokens**; question, context and specials get `-100`.
- Label the **first subword** of each word, rest `-100` (`LABEL_ALL_SUBWORDS=False`).
- **Truncation is `only_first`** — sacrifice context before the answer. Truncating the
  answer silently discards labels, which is unrecoverable.
- Example-level score = **max** over answer token hallucination probabilities.
- Span extraction = merge runs of consecutive tokens above threshold 0.5.

## Evaluation protocol

All implemented in `src/evaluate.py`, which scores every system through one code path.

- **Word-level** P/R/F1 on the hallucinated class — *headline*
- **Span-level** — *headline*. Gold spans are runs of flagged words in the gold token
  labels; spans are compared by word position, not text. Reported as span F1 (exact
  boundaries, pooled over all rows so spans on faithful answers count as false
  positives), span exact (share of hallucinated rows whose every gold span is predicted
  exactly) and span partial (any overlap). Text comparison against `hallucinated_span`
  was dropped once punctuation became its own word; the two agree on every row anyway.
- **Example-level** P/R/F1, accuracy, AUROC — only with the baseline beside it
- **Per hallucination type** — types are now balanced (~333 each, 41–59 per type in
  test). `omission` is the new one and the baseline's weakest; say out loud whichever
  type the model is weakest on.
- **Latency**: ms per example at batch size 1, CPU and GPU separately, after ~20 warmups,
  on the actual local machine. Never on Colab — the claim is about consumer hardware.
- **Efficiency**: parameter count, trainable count under LoRA, peak memory, wall-clock

Compare against: the lexical baseline (both modes), full fine-tuning vs LoRA, and at
least one LLM-as-a-judge reference point for the cost argument. For the RAGTruth arm,
compare to Luna and LettuceDetect's published numbers.

### Headline results — BanglaBERT, 3 seeds, measured 2026-09-27

**Quote these numbers.** Seeds 42/43/44, up to 10 epochs with early stopping on val word
F1 (patience 2), effective batch 16 (8 × 2 accumulation), trained on the RX 6600 GPU.
Splits are identical across seeds. Test split (588 rows), mean ± std. Produced by
`python -m src.summarize banglabert-full-e10 banglabert-lora-e10`, written to
`outputs/metrics/summary.json`.

| system | **word F1** | word P | word R | **span F1** | span exact | ex F1 | AUROC |
|---|---|---|---|---|---|---|---|
| lexical baseline (`morph`) | 0.370 | 0.295 | 0.497 | 0.089 | 0.211 | 0.686 | 0.659 |
| **BanglaBERT full** | **0.835 ± 0.007** | 0.883 ± 0.016 | 0.793 ± 0.024 | **0.583 ± 0.016** | 0.580 ± 0.014 | 0.924 ± 0.004 | 0.977 ± 0.001 |
| BanglaBERT LoRA | 0.822 ± 0.016 | 0.862 ± 0.037 | 0.786 ± 0.031 | 0.513 ± 0.017 | 0.524 ± 0.015 | 0.925 ± 0.011 | 0.974 ± 0.005 |

Per type, word F1 (full / LoRA): `fabricated_info` 0.976 / 0.980, `number_error` 0.882 /
0.837, `contradiction` 0.844 / 0.829, `date_error` 0.827 / 0.793, **`entity_replacement`
0.643 / 0.666, `omission` 0.598 / 0.598** (± 0.01–0.04).

What the tuning run established:

- **More epochs do not help.** The best epoch was 3, 9 and 4 (full) and 3, 3 and 7
  (LoRA). Val word F1 plateaus around epoch 3. The 4-epoch CPU run (0.833) was already at
  the ceiling for this setup, and the 3-seed mean is 0.835.
- **Threshold tuning does not help.** The val-tuned thresholds are unstable (0.30–0.65),
  and test word F1 is identical at 0.5 and tuned (0.835 vs 0.835). Span F1 is slightly
  worse when tuned. **Keep 0.5** in the UI and API. `train.py` still reports both.
- **Full vs LoRA.** The word F1 gap (−0.013) is within one LoRA std, so it is **not
  distinguishable from seed noise**. The span F1 gap (**−0.070**, stds ~0.017) is real:
  LoRA finds hallucinations about as well but draws their boundaries worse.
- **LoRA saves real time on the GPU:** 168 ± 21 s per epoch vs 304 ± 22 s for full,
  **1.8× faster**. On the CPU it saved only 9%. The saved adapter is **5 MB vs 422 MB**
  (84× smaller), which is the deployment argument for LoRA.
- **Seed variance is small** (word F1 std 0.007 full). A difference between systems
  below ~0.02 word F1 should not be claimed from one seed.

**Not reportable from these runs.** "Peak memory" on DirectML grows with the number of
epochs trained (full: 3.1 GB at 5 epochs, 6.4 GB at 10 epochs), so it measures run
length, not the model. It is host RAM, and VRAM is not measured. The in-training
latencies are noisy on DirectML (one run read 52.6 ms, the others 13–20 ms). Use
`src/latency.py` for latency.

### Results on the 4,000-row dataset — measured 2026-09-25 (single seed, superseded by the headline above)

Test split (588 rows), 4 epochs, identical splits and seed, best epoch 4 for all.
BanglaBERT trained on the CPU, mmBERT on the RX 6600 GPU:

| system | word P | word R | **word F1** | **span F1** | span exact | span partial | ex F1 | AUROC |
|---|---|---|---|---|---|---|---|---|
| lexical baseline (`exact`) | 0.238 | 0.556 | 0.333 | 0.072 | 0.207 | 0.820 | 0.676 | 0.641 |
| lexical baseline (`morph`) | 0.295 | 0.497 | 0.370 | 0.089 | 0.211 | 0.731 | 0.686 | 0.659 |
| BanglaBERT full | 0.885 | 0.787 | **0.833** | **0.581** | 0.575 | 0.888 | 0.919 | 0.979 |
| BanglaBERT LoRA | 0.838 | 0.796 | 0.817 | 0.488 | 0.500 | 0.908 | 0.929 | 0.975 |
| mmBERT LoRA | 0.817 | 0.800 | 0.809 | 0.527 | 0.551 | 0.878 | 0.894 | 0.962 |

The model beats the honest baseline by **+0.463 word F1** and **+0.492 span F1**.

Per type (word F1, full / LoRA): `fabricated_info` 0.976 / 0.985, `contradiction` 0.851 /
0.804, `number_error` 0.845 / 0.838, `date_error` 0.824 / 0.786, **`omission` 0.638 /
0.580, `entity_replacement` 0.627 / 0.663**. The last two are the weak spots: only ~74%
of those rows are caught at all (example recall 0.674–0.739). They are also what the
lexical baseline cannot see, so this is where the paper should look for the model's
added value and its limits.

Full vs LoRA:

| | full | LoRA | delta |
|---|---|---|---|
| **word F1** | **0.833** | 0.817 | −0.016 |
| **span F1** | **0.581** | 0.488 | −0.093 |
| example F1 | 0.919 | **0.929** | +0.010 |
| trainable params | 110,028,290 | **886,274** | 124× fewer |
| wall-clock | 5,658 s | **5,144 s** | −9% |

The shape matches the first sample: LoRA costs headline accuracy — noticeably so at span
level, where exact boundaries matter — nudges example F1 up, and saves little training
time on CPU. Report it honestly rather than implying LoRA was necessary at 110M.

**mmBERT does not beat BanglaBERT.** It has 2.8× the parameters and ~2.9× longer inputs
(fertility), yet word F1 is 0.809, below both BanglaBERT runs. Its span F1 (0.527) sits
between BanglaBERT LoRA and full. It shares the same weak types: `omission` 0.626 and
`entity_replacement` 0.625 word F1. This backs BanglaBERT as the primary model:
a Bangla-specific tokenizer beats a bigger multilingual model here.

Latency, re-measured on saved checkpoints with `src/latency.py`, batch size 1, 20
warmups, 100 timed calls cycling through the same 30 test examples, nothing else
running. Results are in `outputs/metrics/latency.json`:

| model | tokens (avg) | CPU median | CPU p95 | GPU median | GPU p95 |
|---|---|---|---|---|---|
| BanglaBERT full | 145 | **57.1 ms** | 99.7 ms | **11.5 ms** | 20.8 ms |
| BanglaBERT LoRA | 145 | 66.3 ms | 118.9 ms | 12.4 ms | 22.2 ms |
| mmBERT LoRA | 417 | **210.2 ms ✗** | 369.9 ms | 52.7 ms | 103.9 ms |

CPU is a Ryzen (AMD64 Family 25 Model 97), 6 cores, torch 2.14. GPU is the RX 6600 via
DirectML, torch 2.4.1. LoRA checkpoints are timed with the adapter unmerged, which
explains their few extra ms. Merging it (`merge_and_unload`) would match full
fine-tuning. **mmBERT breaks the paper's 200 ms budget on CPU.** It is usable only on
the GPU.

**Two measurements recorded during training are not reportable. Use the table above:**

- **Latency in the BanglaBERT metrics JSONs** (full 153.2 ms, LoRA 108.8 ms) was timed
  on **one** test example, and cost grows with length. `src/train.py` now cycles through
  30 examples, as the mmBERT run did.
- **"Peak memory"** in the BanglaBERT JSONs (1,794 / 957 MB, and 2,749 / 2,025 MB on the
  first sample) was the process's memory at the *end* of the run, not the peak.
  `src/train.py` now reads the true peak working set. The mmBERT figure (6,158 MB, host
  RAM, not VRAM) uses the fix. Rerun BanglaBERT to get true peaks before quoting memory.

### LoRA vs full fine-tuning — first sample, measured 2026-09-18

**Superseded by the results above; kept for the record.** Measured on the first
2,652-row sample, whose baseline was far stronger, so none of these numbers transfer.
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

The latency and "peak memory" rows above suffer the same measurement flaws described
for the 4,000-row runs.

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
│   ├── baselines.py     # lexical string-match baseline, exact + morph
│   ├── latency.py       # re-time saved checkpoints, CPU and GPU, same 30 examples
│   └── data_ragtruth.py # RAGTruth adapter for the English arm
├── app/
│   ├── _loader.py       # checkpoint loading + inference shared by UI and API
│   ├── streamlit_app.py # Streamlit demo
│   └── api.py           # FastAPI + Uvicorn backend, promised by the paper
└── outputs/             # checkpoints, metrics JSON, plots (the paper's "Data Store D1")
```

## Environment

Local, VS Code. Two venvs:

- **`.venv`** — CPU-only torch 2.14. BanglaBERT at `max_length` 288: 2,824 training rows,
  177 steps per epoch, ~22 min per epoch, ~95 min per 4-epoch run.
- **`.venv-dml`** — torch 2.4.1 + `torch-directml`, for the **AMD RX 6600** (8GB) via
  DirectML. ROCm does not support this card on Windows or WSL, so DirectML is the only
  GPU route. Same `transformers`/`peft` versions as `.venv`. Run with `--device dml` and
  `HF_HUB_OFFLINE=1`.

DirectML quirks, all measured 2026-09-25:

- **~4× faster than CPU.** BanglaBERT full fine-tuning: 0.10 s per example vs 0.40 s.
  Inference: 11.2 ms vs 53.3 ms median.
- **Bias-free LayerNorm backward is unsupported**, and crashes the device. mmBERT uses
  bias-free norms throughout. `src/model.py:decompose_layernorms` rewrites them as plain
  ops (identical output to ~2e-6, same checkpoint keys). `src/train.py` applies it
  automatically for `--device dml`.
- **Padded batches run out of VRAM on mmBERT**, even at batch 2: padded attention masks
  are expensive under DirectML. Batch 1 has no padding and is stable. So run mmBERT with
  `--batch-size 1 --grad-accum 16 --eval-batch-size 1` (~0.14 s per step, ~7 min per
  epoch of training). BanglaBERT fits at batch 8 but not 16.
- **Torch 2.4.1 cannot load `.bin` weights** under current `transformers` (CVE-2025-32434
  guard). Both backbones therefore need a `model.safetensors` in the local HF cache
  snapshot that `refs/main` points to, with no `.no_exist/<rev>/model.safetensors`
  marker. For mmBERT it was converted locally from `pytorch_model.bin`. For BanglaBERT it
  was copied from the other cached snapshot. `HF_HUB_OFFLINE=1` makes `transformers` use
  the cache instead of asking the Hub, which has no safetensors for these repos.

No cloud compute is needed. Latency is always measured on this machine, CPU and GPU
separately, and never on Colab.

## Don't

- Don't random-split the rows. Group on base ID.
- Don't report example-level F1 as the headline, or without the baseline beside it.
- Don't omit the lexical baseline from results tables, and report the `morph` mode.
- Don't use English ModernBERT for Bangla. mmBERT-base is a different model.
- Don't benchmark latency on cloud hardware, or while another job is using the CPU.
- Don't silently drop the 10 non-Bangla rows — keep and note, or drop and note.
- Don't hardcode the sample's dimensions; the real dataset is being built elsewhere.
- Don't quote smoke-test numbers as results.
