# Gold test set: annotation guide

We need **200 examples checked by people** so the paper can say its results hold on
human-verified labels, not only on the generator's labels. Each example is checked by
**two of us, independently**. You have about **100 examples**, which takes roughly
**1.5–2 hours**. You can stop and continue at any time; your work is saved after every
example.

## 1. Get the app running

You don't need the GPU or PyTorch, only Python with Streamlit and pandas.

```powershell
git pull
python -m pip install streamlit pandas     # skip if you already use the project .venv
streamlit run app/annotate.py
```

A browser tab opens. Choose **your name** in the sidebar and read the **Guidelines**
there before you start.

## 2. For each example

1. Read the **context**. It is the only source of truth for this task.
2. Read the **question** and the **answer**.
3. Click every answer word that the context does **not** support. Click again to unmark.
4. Choose a verdict:
   - **Fully supported by the context**: no words marked.
   - **Contains a hallucination**: at least one word marked.
   - **Cannot decide from the context**: only when the context really doesn't let you
     decide. Add a note saying why.
5. Press **Save and go to next**.

The rules in short:

- **Ignore your own knowledge.** A true fact that the context doesn't state is still
  *unsupported*.
- **Mark the smallest wrong part**, usually 1–4 words: the wrong name, number or date,
  or the added claim.
- **Paraphrases are fine.** Synonyms, `ও` vs `এবং`, and small grammar changes count as
  *supported* when the meaning matches.
- **An incomplete list presented as complete is a hallucination.** Mark the list.
- **Never mark punctuation** (`।`, `,`) on its own.

## 3. Work alone

Don't discuss examples and don't open the dataset CSV until **everyone** has finished.
The paper reports how often two of us agree. Talking about it first would inflate that
number and make it worthless.

## 4. Hand in your work

Your answers are saved in `data/gold/annotations/<YourName>.jsonl`. Commit and push
**only your own file**:

```powershell
git add data/gold/annotations/<YourName>.jsonl
git commit -m "Gold annotations: <YourName>"
git pull --rebase
git push
```

Each person has their own file, so pushes never conflict.

## 5. After everyone has finished

- `python -m src.gold status`: who has finished what.
- `python -m src.gold agree`: how often we agreed (Cohen's kappa), and how many
  examples we disagree on.
- **Settle the disagreements together.** Open the app, switch the sidebar to
  **Adjudicate**, go through each disagreement, and save the answer the guidelines
  support. The person saving is recorded.
- `python -m src.gold evaluate`: scores BanglaBERT, the baseline and the LLM judges
  against our labels, and checks how often the dataset generator's own labels were right.
