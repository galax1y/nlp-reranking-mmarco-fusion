# Phase 2 — v2: fine-tune our own cross-encoder

**Idea (option A):** train our own Portuguese cross-encoder instead of only using a ready-made one.
- **Sources:** Nogueira & Cho 2019; mMARCO (Bonifacio et al.); ptt5-v2. All are Very High / High relevance
  in `research/literature_review.xlsx`.
- **What it closes:** the "implementar" gap in the objective. See `WEAKNESSES.md` #3.

**Evaluation fixes built in:**
- **Real evaluation split:** all **6,980** official mMARCO-pt **dev small** queries, as in the papers, no longer train queries (#1). The snapshots for the learning curve use a fixed 1,000-query subset.
- **Hard candidates:** the official BM25 top-100 over the full 8.8M collection (#2).
- **Statistics:** the full dev set, so no sampling bias (#5), bootstrap 95% CIs, and paired Wilcoxon tests with Holm
  correction (#4).
- **Per-query analysis:** wins, ties and losses, plus examples of losses (#6).
- **Ceiling:** BM25 Recall@K reported as the first-stage ceiling (#7).
- **Other fixes:**
  - warmed-up latency on the same GPU, with p50 and p95 (#11)
  - `ftfy` text repair (#12)
  - stable tie-breaking (#13)
  - versions recorded (#15)
  - each pair scored once (#16)

## Notebook: `v2_completo.ipynb` (one notebook, free Colab, GPU = T4 only)

How to run: **Runtime → Run all**. If the session drops, reconnect and **Run all** again: each stage
checks Drive (`MyDrive/AssignmentPLN_v2/`) and skips or resumes.

| Stage | What it does | GPU | Est. time* | Checkpoint on Drive | On re-run |
|---|---|---|---|---|---|
| 1. Data | All 6,980 dev queries + their BM25 top-100; 300k + 2k training triples; texts from the 3.4 GB collection | No | 20–40 min | `data/` | Skipped if the data exists with the same config. The training triples are kept even if the dev config changes |
| 2. Training | Hand-written PyTorch loop (BCE, AdamW, warmup, fp16) from `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`, with snapshots for the learning curve | Yes | ~45 min for 150k triples | `checkpoints/<RUN_NAME>/` (resume point every 15 min, `final/`) | Resumes from the last point; skipped once `final/` exists |
| 3. Evaluation | BM25 vs ready-made CE vs our CE (K = 10/25/50/100), CIs, significance, wins/losses, learning curve, latency, examples | Yes | 45–75 min | `results/` (scores cached per model) | Models already scored are not re-scored |
| 4. Cascade + step-by-step benchmark | `unicamp-dl/monoptt5-base` (T5, fp32) reranks the top-N (5/10/20) of each CE's top-100. Table per pipeline step: MRR@10 with CI, Δ and paired test against the previous step, the ceiling handed to the next step, cumulative latency. Also cascade vs monoPTT5 applied directly to BM25 top-100 (1,000-query subset) | Yes | 1–2 h | `results/t5_pair_scores.pkl` (cache per (query, passage) pair) | Pairs already scored are not re-scored |
| 5. Score fusion | Combines sources instead of one replacing the other: CE pronto + monoPTT5, CE nosso + monoPTT5, CE pronto + CE nosso, CE + BM25, and all three. Interpolation of per-query z-scores with α chosen by 2-fold cross-validation, plus RRF (k=60); paired tests against the best single source | No | a few minutes | `results/fusao_*.csv/png` | Runs standalone after the Drive + Configuração cells (reads stage 3/4 caches) |

\* Time estimates are **Assumed** (extrapolated from v1's 2.9 ms per pair). The notebook prints the real speed.

- **All settings are in the single "Configuração" cell.** `RUN_NAME` is derived from the training settings, so
  stage 3 always evaluates the model stage 2 trained.
- **GPU-quota tip:** stage 1 needs no GPU. You can run it on a CPU runtime: the notebook stops at stage 2 with a
  message. Then switch to a T4 and Run all again.
- **The earlier three notebooks** (`01`/`02`/`03`) are in `archive/`. They use the same Drive folders, so
  anything they already produced is picked up by `v2_completo.ipynb`.

**Drive space:** about 4–5 GB in total.
- Each snapshot is ~240 MB, and there are 5.
- The final model is ~470 MB.
- The resume file is ~1.4 GB and is deleted at the end.
- Data files are ~0.5 GB.

## Verification status
- **Verified:**
  - All file paths and formats in the mMARCO repo (checked on Hugging Face, 2026-11-24).
  - The base model exists and has its tokenizer files.
  - Every code cell parses, and a static check found no undefined names across the merged notebook.
  - The metric, reranking, Holm-correction and cascade functions were tested on toy inputs.
  - Stage 5 was run end to end on synthetic data with known answers: BM25-only = BM25 order; α=1 = CE alone; α=0 = cascade;
    fusing two independent noisy signals beats each; the cross-validated α never beats the optimistic one.
  - monoPTT5: the files exist, `▁Sim` = 4295 and `▁Não` = 852 are single tokens (`spiece.vocab`), decoder start id = 0.
    The notebook self-tests against the model card's example scores (0.8187 / 0.0080) before scoring anything.
- **Not verified:** nothing was run end to end. There is no GPU or `transformers` here, so the first Colab
  run is the real test.

The notebooks were generated with a script; edit the `.ipynb` files directly from now on.
