# v2 results — analysis (run of 2026-11-25/26, Colab T4)

> Source: `phase2/results/*.csv` from `v2_completo.ipynb`.
> - **Setup:** all **6,980** mMARCO-pt dev-small queries; official BM25 top-100 over the full 8.8M collection.
> - **Environment:** torch 2.14.1, transformers 5.22.1, Tesla T4, CE in fp16, monoPTT5 in fp32.
> - **Evidence labels:** every number is read from the CSVs or recomputed from
>   `ranks_por_pergunta_com_cascata.csv` with `analysis/analyze_v2.py` (`python analysis/analyze_v2.py results`).
>   Interpretations not directly measured are marked *Inferred* or *Assumed*.

## 1. Headline numbers

| System | MRR@10 [95% CI] | nDCG@10 | Recall@10 |
|---|---|---|---|
| BM25 (official run) | 0.153 [0.146, 0.160] | 0.189 | 0.311 |
| CE nosso (K=100) | 0.255 [0.246, 0.263] | 0.299 | 0.449 |
| CE pronto (K=100) | 0.285 [0.276, 0.294] | 0.330 | 0.478 |
| CE nosso → monoPTT5 (N=20) | 0.285 [0.276, 0.294] | 0.328 | 0.472 |
| CE pronto → monoPTT5 (N=20) | 0.286 [0.277, 0.295] | 0.330 | 0.479 |

**Sanity checks against the literature:**
- **BM25:** our 0.1529 vs the mMARCO paper's **0.152** for Portuguese. The data pipeline reproduces the
  published setup.
- **CE pronto:** our 0.2854 (reranking BM25 top-100) vs the paper's mMiniLM **0.289** (top-1000). The small gap
  fits our shallower depth (*Inferred*).

## 2. v1 was inflated about 3.2×

The same ready-made cross-encoder at K=100 scored:

| Setup | MRR@10 |
|---|---|
| v1: training queries, 5k mostly-random passages | 0.909 (TF-IDF) / 0.913 (BM25) |
| v2: dev queries, BM25 top-100 over the full collection | **0.285** |

The v1 numbers measured an easy, possibly leaked benchmark. This is the main methodological finding.

## 3. The first stage is the main limit

| | Value |
|---|---|
| Queries with any labelled relevant passage in BM25 top-100 | **56.6%** (3,947 / 6,980) |
| Perfect-reranker MRR@10 (relevant always moved to rank 1 when reachable) | **0.566** |
| BM25 Recall@10 / @25 / @50 / @100 / @1000 | 0.311 / 0.411 / 0.484 / 0.557 / 0.744 |

**How much of that ceiling each system uses:**

| System | % of the perfect reranker | Share of the BM25 → perfect gap closed | Rank-1 hit rate on reachable queries |
|---|---|---|---|
| BM25 | 27.0% | 0% | 15.6% |
| CE nosso (K=100) | 45.0% | 24.6% | 29.3% |
| CE pronto (K=100) | 50.5% | 32.1% | 34.2% |
| CE nosso → monoPTT5 (N=20) | 50.4% | 32.0% | 34.4% |

Two limits stack up:
- **43% of queries are lost before reranking starts.**
- **On the queries that remain, even the best system ranks the relevant passage first only about a third of the time.**

Missing labels make both look worse than they are. Dev small has about one labelled relevant passage per query
(*Inferred*: the share of unlabelled correct answers is unmeasured).

## 4. K matters a lot on real candidates (unlike v1)

| K | CE pronto MRR@10 | CE nosso MRR@10 | CE pronto Recall@10 / ceiling |
|---|---|---|---|
| 10 | 0.220 | 0.208 | 100% (by construction) |
| 25 | 0.255 | 0.235 | 96.4% |
| 50 | 0.270 | 0.245 | 90.7% |
| 100 | 0.285 | 0.255 | 85.8% |

- **Every step of K still pays.** In v1, MRR was nearly flat after K=25.
- **Deeper candidates get used less efficiently:** the reranker converts 96% of the reachable relevants into the
  top 10 at K=25, but only 86% at K=100.
- **Rescues by how deep BM25 had buried the relevant passage** (CE pronto):

  | BM25 rank of the relevant | Ends in the top 10 | Ends at rank 1 |
  |---|---|---|
  | 1 | 97% | 64% |
  | 11–25 | 84% | 22% |
  | 51–100 | 69% | 19% |

  The reranker really does recover deep passages.
- **Recall@1000 is 0.744, but the paper's top-1000 rerank reaches only 0.289 vs our 0.285.** A higher ceiling
  doesn't turn into MRR (*Inferred* from the two numbers).

## 5. Our fine-tuned cross-encoder

- **Gap to the ready-made model:** 0.255 vs 0.285, Δ = **−0.031 [−0.036, −0.026]**, significant (Holm p ≈ 1e-36).
- **Share of the gain:** it gets **77%** of the ready-made model's gain over BM25 (+0.102 vs +0.132), with
  300k training pairs and ~45 min on a T4.
- **Per query:** wins/ties/losses against BM25 are 2,562 / 3,548 / 870 for ours and 2,733 / 3,602 / 645 for the
  ready-made model. Ours rescues 1,186 queries into the top 10, and pushes 209 out of it.

**Learning curve (1,000-query subset; BM25 = 0.142 and CE pronto = 0.281 on the same subset):**

| Pairs seen | Pair accuracy (validation) | MRR@10 |
|---|---|---|
| 15k | 0.917 | 0.130 (**below BM25**) |
| 30k | 0.941 | 0.195 |
| 75k | 0.956 | 0.236 |
| 150k | 0.965 | 0.242 |
| 300k | 0.967 | 0.255 |

- **Still rising at 300k, with diminishing returns:** +0.04 from 30k to 75k, +0.019 from 75k to 300k.
  More data would probably keep helping (*Assumed*).
- **Validation pair accuracy saturates at about 0.96 while MRR keeps improving.** Pair accuracy on (positive,
  negative) triples is a poor stand-in for ranking quality.
- **Caveat:** the snapshots were taken mid-schedule, so this is training progress, not data scaling.

## 6. The cascade (third stage, monoPTT5-base)

| Path | + monoPTT5 N=5 | N=10 | N=20 |
|---|---|---|---|
| CE nosso (0.255) | +0.021 * | +0.025 * | **+0.030 *** → 0.285 |
| CE pronto (0.285) | +0.002 | +0.002 | +0.001 (all n.s., Holm p = 1.0) |

\* significant after Holm correction.

- **On our CE, the cascade closes the whole gap to the ready-made CE:** 0.2848 vs 0.2854. A weaker first
  reranker plus a strong third stage matches a stronger reranker alone.
- **On the ready-made CE it adds nothing on average, but it changes a lot.**
  - At N=20: 1,083 wins vs 1,082 losses; 363 queries gain rank 1 and 354 lose it.
  - The two models are about equally strong, but they disagree on many queries. That makes **score fusion**
    (combining their scores instead of replacing one with the other) a promising, nearly free next experiment (done in §6b).
    The scores are already cached.
- **Cost per pair:** monoPTT5 ~13 ms (fp32), the CE ~1.4–2.4 ms (fp16), so roughly 7× per pair. Part of the
  difference is precision: T5 can't safely run in fp16.

**Cascade vs monoPTT5 applied directly to the BM25 top-100 (1,000-query subset):**

| System | MRR@10 | Δ vs direct | p (unadjusted) | Rerank latency |
|---|---|---|---|---|
| monoPTT5 direct (K=100) | 0.291 | 0 | – | 1,390 ms |
| CE nosso → monoPTT5 (N=20) | 0.288 | −0.003 | 0.077 | 502 ms |
| CE nosso → monoPTT5 (N=5) | 0.279 | −0.012 | 0.005 | 311 ms |
| CE pronto alone | 0.281 | | | 173 ms |

- **The cascade at N=20 reaches the direct model's quality with no detectable loss, using about 36% of its
  latency.** That's the funnel argument from the multi-stage paper, now measured.
- **Subset noise:**
  - On this subset, monoPTT5 looks better than CE pronto (0.291 vs 0.281).
  - On the full set, CE pronto → monoPTT5 barely moves CE pronto (0.285 → 0.286).
  - With 1,000 queries the CI half-width is about ±0.025, so small differences there aren't reliable.

## 6b. Score fusion (stage 5, run 2026-11-26)

Source: `results/fusao_resultados.csv`, `fusao_curva_alpha.csv`. All 6,980 queries.
- **"Best single source"** is the best member of each combination used on its own.
- **Interpolation:** α was chosen by 2-fold cross-validation.
- **p-values:** Holm-corrected across all fusion tests.

| Combination | Best single source | Interpolation (CV α) | RRF (k=60) |
|---|---|---|---|
| **CE pronto + monoPTT5** (top-20) | 0.2863 (monoPTT5 cascade) | 0.2899 (+0.004, p=0.13) | **0.2920 (+0.006, p=0.015)** ✅ |
| CE nosso + monoPTT5 (top-20) | 0.2848 (monoPTT5 cascade) | 0.2848 (α=0 chosen: CE adds nothing) | 0.2752 (−0.010, sig. worse) |
| CE pronto + CE nosso (top-100) | 0.2854 (CE pronto) | 0.2855 (+0.000, n.s.) | 0.2793 (−0.006, sig. worse) |
| CE pronto + BM25 (top-100) | 0.2854 (CE pronto) | 0.2844 (α≈1 chosen) | 0.2319 (−0.054, sig. worse) |
| CE nosso + BM25 (top-100) | 0.2545 (CE nosso) | 0.2517 (α≈1 chosen) | 0.2197 (−0.035, sig. worse) |
| All three (top-20 of CE pronto) | 0.2863 (monoPTT5) | 0.2866 (equal-weight z-sum, n.s.) | 0.2862 (n.s.) |

**Best system so far:** CE pronto → top-20 → RRF with monoPTT5.
- **MRR@10 0.2920 [0.2831, 0.3011]:** +0.0066 over CE pronto alone, +0.0057 over the cascade.
- **Per query:** wins/losses against the cascade are 933 / 794.
- **Cost:** the same latency as the cascade at N=20, since the same models run.
- **Size:** a small gain (≈2% relative), but significant, and it's the only fusion that helps.

**Why it works there and nowhere else.** Fusion helps only when the sources are **(a) about equally strong and
(b) different** (*Inferred* from the pattern below).

| Combination | Strength | Different? | Outcome |
|---|---|---|---|
| CE pronto + monoPTT5 | 0.285 vs 0.286 | Yes: multilingual MiniLM classifier vs Portuguese T5 generator | Helps |
| CE pronto + CE nosso | Unequal | No: same architecture and data | Redundant, nothing to add |
| Any + BM25 | Far weaker, and BM25 enters as ranks only | Yes | Hurts |
| CE nosso + monoPTT5 | Unequal | Yes | CV picks α = 0 (monoPTT5 alone) |

**RRF vs interpolation:**
- **RRF has no weight to tune:** it trusts every source equally.
- **That is good when the sources are equal** (the best result) **and harmful when they're not** (every RRF with a
  weaker source was significantly worse).
- **Cross-validated interpolation degrades gracefully:** it sets α to 0 or 1 when one source is useless, so it
  never fell significantly below the best source.
- **The best α is stable:** the CE pronto + monoPTT5 interpolation curve stays flat (0.287–0.290) for α between
  0.1 and 0.6.
- **Takeaway:** with an unknown source, use interpolation with validation; with two known-equal rerankers, RRF is
  simpler and slightly better here.

**BM25 adds nothing once a cross-encoder is present.** Caveat: the official run gives only ranks, not BM25
scores, so this tests BM25's *ordering*, not its scores.

## 7. Where reranking helps most

**By question word** (MRR@10; n = number of queries; no CIs, so treat small groups with care):

| Type | n | BM25 | CE pronto | Gain |
|---|---|---|---|---|
| quando | 187 | 0.099 | 0.262 | **+0.162** |
| onde | 283 | 0.173 | 0.330 | +0.158 |
| definição / significado | 607 | 0.136 | 0.290 | +0.154 |
| o que | 1,410 | 0.137 | 0.278 | +0.141 |
| quem | 254 | 0.135 | 0.274 | +0.140 |
| keyword-style (no question word) | 2,580 | 0.185 | 0.314 | +0.129 |
| quanto | 528 | 0.109 | 0.233 | +0.124 |
| qual / quais | 828 | 0.137 | 0.243 | **+0.107** |

- **The biggest gains are on questions where BM25 is weakest** (*quando*, *quanto*): the answer is a date or
  number that doesn't repeat the query's words.
- **Keyword-style queries already work best with BM25** (0.185), so they gain less.
- **Shorter queries gain more:** +0.169 for 1–3 words, +0.114 for 7+.
- This is the most direct evidence for the objective's claim that the cross-encoder **resolves what keyword
  matching misses**. The examples need checking by hand before going in the paper.

## 8. Latency: use it as an order of magnitude only

- **Two identical architectures got different numbers.** CE pronto and CE nosso are the same model size, yet
  they measured **173 ms vs 243 ms** at K=100 (and 79 vs 70 ms at K=50). p95 is about 2× the mean.
- **The cause is the environment:** on a shared Colab T4, measurement noise is ±40%.
- **Report it this way:** per-pair costs (CE ~2 ms, monoPTT5 ~13 ms) and latencies rounded, with the noise
  stated. Don't claim that one CE is faster than the other.

## 9. Other observations

- `ftfy` changed **61,895 passages**, so mojibake is common in mMARCO-pt. Queries were unaffected.
  - **Its repair is only partial** (`results/ftfy_tipos.csv`, `ftfy_exemplos.csv`: top-50 substitutions in a
    300k-passage sample, 149k occurrences).
  - ~9% are accent fixes (`estÃ¡` → `está`) and ~21% quote normalization.
  - ~12% delete mojibake fragments.
  - **~52% turn one kind of garbage into another:** spaced mojibake such as `â € œ` becomes U+2800 (Braille
    blank) or U+08A0 (Arabic letter).
  - *Inferred:* spaces were inserted inside the mojibake (probably by the machine translation), which breaks
    ftfy's decoding.
  - The effect on MRR was not measured, and the official BM25 run uses the original text.

## 10. Main findings

1. **Main claim:** we implemented and trained a Portuguese cross-encoder on free hardware. It reaches 77% of a
   published reranker's gain over BM25, and the training curve hasn't flattened yet.
2. **Methodological claim:** v1's protocol inflated MRR@10 about 3.2× (0.91 → 0.285). The leakage-free protocol
   reproduces the published BM25 (0.153 vs 0.152) and mMiniLM (0.285 vs 0.289) numbers.
3. **Pipeline claim:** the first stage loses 43% of queries before reranking. Deeper K keeps helping. A third
   stage lifts a weak reranker to the strong one's level, and a cascade matches the expensive model at about a
   third of its latency.
4. **Linguistic claim:** gains are largest for *quando*/*onde*/definition questions and short queries, and
   smallest for *qual*/*quais* and long queries.

## 11. Future work

1. ~~Score fusion~~: done (§6b). CE pronto + monoPTT5 with RRF is the best system, 0.292.
2. **Per-group bootstrap CIs for §7,** so the question-type claims carry error bars.
3. **Manual check of ~20 "losses",** to estimate how many are unlabelled correct passages.
4. **Optional, needs GPU:** train for longer (e.g. 300k triples = 600k pairs), since the curve is still rising.
