# Phase 2 — Weaknesses of the v1 Colab (`notebook.ipynb`)

> Written 2026-11-24, before building v2. Each finding is labelled **Verified** (checked against the
> notebook code/outputs or a live source), **Inferred** (read from the code, not run) or
> **Assumed** (a guess plus what would prove it). Severity: **High** = can change the article's
> conclusions; **Medium** = weakens the claims; **Low** = polish.
> Cell numbers are code cells, counted from 0.

## Summary

| # | Weakness | Area | Severity |
|---|---|---|---|
| 1 | Evaluates on mMARCO **train** queries, and the reranker was trained on mMARCO | Evaluation | High |
| 2 | The corpus has almost no hard negatives: ~12 of 5,312 passages | Data | High |
| 3 | No fine-tuning: the objective says *implementar*, but v1 uses a ready-made reranker | Scope | High |
| 4 | 300 queries, no confidence intervals or significance tests | Evaluation | Medium |
| 5 | The queries are not a random sample | Data | Medium |
| 6 | The qualitative analysis only looks for wins | Evaluation | Medium |
| 7 | The first-stage ceiling (Recall@K) is never reported | Evaluation | Medium |
| 8 | Weak lexical baselines: no Portuguese stemming/stopwords, `rank_bm25` defaults | First stage | Medium |
| 9 | The reranker throws away the first-stage score | Reranking | Medium |
| 10 | A single reranker, no comparison or ablation | Reranking | Medium |
| 11 | Latency methodology: no warm-up, CPU vs GPU, mean only | Evaluation | Medium |
| 12 | Garbled encoding (mojibake) and `errors="ignore"` decoding, no text normalization | Data | Low–Medium |
| 13 | Tie-breaking in `argsort` is correlated with the label | First stage | Low |
| 14 | MRR@10 and nDCG@10 are presented as two results, but with one relevant passage they are the same signal | Evaluation | Low |
| 15 | Not reproducible: no version pins, `[0]` file picking, no dataset revision | Reproducibility | Low |
| 16 | Wasteful compute: the same pairs are re-scored for each K | Efficiency | Low |

---

## Details

### 1. Train-split queries + a reranker trained on mMARCO (High) — Verified
- **Code:** cell 1 reads `portuguese_queries.train.tsv` and `triples.train.ids.small.tsv`.
- **What we know:** the model card for `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` says it was trained on
  mMARCO. It doesn't say which split (Verified). The standard practice is to train on the
  train triples, so the reranker has probably **seen these exact (query, positive) pairs** (Inferred).
- **Why it matters:** the gain from 0.80 to 0.91 MRR@10 may be partly memorization. Every paper in our
  review evaluates on the **dev** set.
- **Fix:** use the official dev split. It exists in the dataset repo (Verified 2026-11-24):
  - `data/google/queries/dev/portuguese_queries.dev.small.tsv`
  - `data/qrels.dev.small.tsv`
  - `data/google/runs/run.bm25_portuguese-msmarco.txt`, an official Portuguese BM25 run (133 MB)

### 2. The corpus has almost no hard negatives (High) — Verified (arithmetic on the output)
- **How the corpus is built (cell 1):**
  - 300 positives
  - the triple's negative, **only if it is in the pool**
  - 5,000 "distractors", which are simply the **first** passages of the pool
- **The numbers:**
  - The output reports 5,312 passages, so positives + negatives = 312, which means only **about 12
    hard negatives** made it in.
  - The pool is the first 250k of 8.8M passages (~2.8%), so most negatives are missing.
- **Consequence:** for most queries the relevant passage is the only on-topic text in a pile of random
  passages. The task becomes "find the passage about the same subject", which is easy for BM25 and
  trivial for a cross-encoder.
  - This is why our absolute scores (0.80 → 0.91) are about 3× the literature's (0.15 → 0.29 on the
    full collection). The two sets of numbers are **not comparable**, and the article must say so.
- **Fix:**
  - Build the corpus from the official BM25 top-100/1000 of each dev query. Those passages are hard by
    construction.
  - Or include each query's BM25 top candidates from the full collection.

### 3. "Implementar" vs using a pre-trained model (High) — Verified
- The objective is to *"implementar e avaliar um modelo baseado na arquitetura Transformer, especificamente
  um cross-encoder"*. v1 loads a ready-made checkpoint and only runs inference (cell 4).
- A reviewer may ask what was *implemented*.
- Nogueira & Cho, ptt5-v2 and mMARCO all **fine-tune their own reranker**.

### 4. No significance testing (Medium) — Verified
- There are 300 queries, and each reported number is a single mean.
- Differences such as BM25+CE K=50 vs K=100 (+0.013 MRR) amount to a few queries.
- **Fix:**
  - A paired test over per-query scores (paired t-test or Wilcoxon).
  - A bootstrap 95% CI for each mean.

### 5. Non-random queries (Medium) — Inferred
- The query selection takes the **first** 300 triples whose positive is among the first 250k passage IDs.
  It is a sample ordered by file position, not a random draw.
- Whether that biases topic or difficulty is unknown (Assumed; comparing against a random sample would show it).

### 6. Cherry-picked qualitative analysis (Medium) — Verified
- Cell 5 keeps only the queries where both lexical methods missed and the CE put the relevant passage
  at rank 1 (11 cases). Losses are never shown: queries where the reranker pushed the relevant passage
  **down**.
- **Fix:** a wins/ties/losses table, plus some losses read by hand. Some losses may be false negatives,
  i.e. unlabelled relevant passages.

### 7. The first-stage ceiling is not reported (Medium) — Verified
- Only Recall@10 is computed.
- Recall@25/50/100 of the first stage are never reported, yet they are the maximum any reranker can
  reach. That is the key number for explaining the K sweep.

### 8. Weak lexical baselines (Medium) — Verified (code) / Assumed (values)
- `tokenize_pt` is `re.findall(r'\w+', text.lower())`: no Portuguese stemming, no stopword removal.
- `BM25Okapi` runs with its library defaults.
  - Assumed: k1=1.5, b=0.75. Not re-checked against the installed version.
  - The papers use k1=0.9 and b=0.4 (Pyserini).
- A weak baseline inflates the reranker's apparent gain.

### 9. The first-stage score is discarded (Medium) — Verified
- Cell 4 sorts by `ce_scores` only.
- Interpolating with BM25, or using RRF, is a standard and cheap baseline (Lin/Nogueira/Yates survey).
  It is not tested.

### 10. A single reranker, no comparison (Medium) — Verified
- There is no second reranker, no ablation of size or architecture, and no model specific to Portuguese.
- **Candidates that exist on Hugging Face** (Verified, HTTP 200, 2026-11-24):
  - `unicamp-dl/mMiniLM-L6-v2-pt-v2` (the mMARCO authors' Portuguese mMiniLM)
  - `unicamp-dl/monoptt5-base` (ptt5-v2, seq2seq)

### 11. Latency methodology (Medium) — Verified
- **No warm-up:** the first `predict` call includes CUDA and kernel initialization.
- **Different hardware:** stage 1 runs on CPU, the CE on GPU.
- **Not comparable across pipelines:** the reported total adds the *mean* stage-1 time to the *mean* CE
  time, measured in separate loops.
- **Mean only:** no p50 or p95.
- **No tokenization cost:** tokenization happens inside `predict`, so it is counted, but
  `rank_bm25` brute-force scoring is not representative of an inverted index.

### 12. Encoding (Low–Medium) — Verified
- Every line is decoded with `errors="ignore"`, which silently drops bytes.
- The collection already contains mojibake. Verified: `PlazaÃ®` tokenizes to `plazaã`.
- Nothing normalizes the text before either stage. A fixer such as `ftfy` would repair it.
- This hurts BM25 more than the CE, so it **inflates the reranker's gain**.

### 13. Tie-breaking correlated with labels (Low) — Inferred
- Positives are added to `corpus` first, so they get the lowest doc ids. Distractors come later.
- `np.argsort(scores)[::-1]` breaks ties by putting the **higher** id first.
- So, among passages tied with a positive (for example score 0 for queries with few matches), the
  positive is systematically ranked **last**.
- The effect is small, but doc id order should never correlate with the label. The fix is to shuffle
  the corpus ids.

### 14. MRR@10 ≈ nDCG@10 (Low) — Verified
- With exactly one relevant passage, nDCG@10 = 1/log2(rank+1). It is a monotone function of the same
  rank as MRR.
- Present them as one result, not two pieces of evidence.

### 15. Reproducibility (Low) — Verified
- `pip install -q datasets sentence-transformers rank-bm25` installs unpinned versions.
- Files are found with `[f for f in repo_files if ...][0]` and no dataset revision.
- No `requirements`/versions are printed in the output.

### 16. Redundant scoring (Low) — Verified
- For each K the CE re-scores `top[:K]`, so the top 10 is scored 4 times.
- Scoring the top 100 once and slicing would cut CE compute by ~45% (10+25+50+100 = 185 vs 100 pair
  scorings). That frees time budget for v2 experiments.
- Latency per K can still be reported as K × the per-pair cost, or timed separately on a small subset.

---

## What this means for v2

- **Weaknesses 1 and 2 have to be fixed in any v2:** otherwise every new technique is measured on a
  leaked, easy benchmark, and we can't tell whether it helps.
- **Weaknesses 4, 6, 7 and 11 are cheap evaluation fixes:** they belong in v2 regardless of which
  technique we add.
- **The "new idea" should come from weaknesses 3, 8, 9 or 10.** Candidates (from the Very High / High papers only):

| Idea | Source paper | Fixes | Effort on a Colab T4 |
|---|---|---|---|
| **A. Fine-tune our own cross-encoder** on Portuguese mMARCO train triples (start from `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`), evaluate on dev | Nogueira & Cho 2019; mMARCO; ptt5-v2 | 3 (and 1, since evaluation moves to dev) | High (training run; time not measured yet) |
| **B. Compare a seq2seq reranker** (`unicamp-dl/monoptt5-base`, true/false generation) against our CE | ptt5-v2; Lin/Nogueira/Yates | 10 | Low–Medium |
| **C. Score fusion** BM25 + CE (interpolation / RRF) | Lin/Nogueira/Yates | 9 | Low |
| **D. Stronger first stage**: Portuguese stemming + stopwords, BM25 k1=0.9 / b=0.4 | ptt5-v2; BEIR | 8 (raises the ceiling) | Low |
| **E. Third, pairwise stage** (mono → duo) | Multi-Stage (monoBERT/duoBERT) | 10 | High (no Portuguese duo checkpoint verified) |

Decision pending (see `../STATUS.md`).
