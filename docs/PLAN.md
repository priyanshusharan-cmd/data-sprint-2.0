# Plan, status and findings — NLP track

Last updated: 23 Nov 2026. Leaderboard at that time: top public score 0.837 (5 teams); organiser baseline (char TF-IDF + LinearSVC, no augmentation) 0.520 public.

## 1. Data (measured, `analysis/eda.py`)
| | train | test |
|---|---|---|
| rows | 16,936 | 4,912 |
| Devanagari / Roman / mixed | **61% / 33% / 6%** | **33% / 64% / 3%** |
| tokens per query | mean 7.6, p95 14 | mean 7.9, p95 14 |
| rows with digits | 0% | 1.3% |
| fillers (bhai, please, na, jaldi, hey) | ≈0% | ~3% each |

- 56 classes, all present; smallest class 36 (84 rows), largest 48 (1,224). Class 55 = 249 rows.
- Train class 55 looks like a few held-out intents merged (alarms, coffee, podcasts, games). Test OOS also contains **novel** topics (coding, homework, definitions), so class 55 needs a confidence rule, not only a learned class.
- Many low-similarity test rows are written in **English-heavy colloquial Hinglish** (English content words, digits, fillers) that train lacks; pretrained multilingual encoders should handle this style better than TF-IDF. (Style observation on unlabeled test text only. An earlier draft of this note named intent IDs for two test queries; those were never used in any code, rule, label or submission, and no test row is ever labelled by hand.)
- Test nearest-train cosine (char TF-IDF): mean 0.58 vs 0.73 for a held-out train split; 32% of test rows below 0.5 vs 11%. This is why train-CV over-estimates the leaderboard.
- Inferred class meanings (top keywords per class): `analysis/classes.py` output.

## 2. Experiments so far (CPU)
| experiment | macro-F1 |
|---|---|
| fit Devanagari only → score real Roman rows, raw text | 0.061 |
| same + romanized copies (`augment.romanize`) | 0.887 |
| phonetic features + romanized copies | 0.922 |
| TF-IDF baseline, 5-fold CV | 0.8733 |
| TF-IDF + augmentation (2 views/row), 5-fold CV | **0.8859** (Roman rows 0.904 → 0.931) |
Submission files: `submissions/v001_tfidf.csv`, `submissions/v002_tfidf-aug.csv`. All scores live in `submissions/registry.csv`.

## 2b. GPU results + leaderboard (23 Nov)
| run / version | val_clean | val_shift | OOF tuned | public LB |
|---|---|---|---|---|
| v002 TF-IDF + aug (5-fold) | 0.886 | — | — | 0.81230 |
| MuRIL defaults (lr 3e-5, llrd 0.9, 6 ep), fold 0 | 0.738 | 0.717 | — | not submitted (under-trained) |
| **MuRIL lr 6e-5, llrd 0.95, 10 ep** (`--run muril_lr6`), fold 0 | 0.885 | 0.874 | 0.888 | **v003 = 0.81917** |
| blend MuRIL fold0 0.4 + TF-IDF 0.6 (`--search`) | — | — | 0.9115 (fold-0 rows) | v004, **not submitted** |
| **v005 = full MuRIL + full TF-IDF 0.5/0.5** (compliant 246.9M, tuned on 5-fold OOF) | — | — | 0.9150 (test-like 0.9086) | **0.84464** (#2 on 23 Nov, #1 0.84686) |
| MuRIL + clean pseudo-labels (fold 0, leak-free teacher) | 0.886 | 0.878 | test-like 0.8745 (worse) | dropped |
| HingRoBERTa `--lr 5e-5 --llrd 0.95 --epochs 10` (`hing_lr5`), 5 folds | folds 0.889/0.895/0.902/0.896/0.885 | | OOF 0.8962, test-like 0.9041 | |
| **v007 = full HingRoBERTa + full TF-IDF 0.5/0.5** (compliant 287,378,288) | — | — | 0.9169 (test-like 0.9191) | **0.86786 (#1 on 23 Nov)** |
- **Final picks (24 Nov): v007 + v009.** Demo and `src/infer.py` default to v007 (end-to-end p50 ~20 ms with ONNX Runtime, plugged in).

## 2c. Last-day experiments (24 Nov) — all lost to v007 on public; full story in `docs/JOURNEY.md`
| version | change | local check | public |
|---|---|---|---|
| — | third linear model in the 12.6M spare params (`analysis/linear2.py`) | practice exam lower on every weight | not uploaded |
| v008 | trimmed HingRoBERTa + trimmed MuRIL + TF-IDF 0.35/0.3/0.35 (`tools/trim_vocab.py`, 299,498,920) | OOF 0.9210, test-like 0.9226 | 0.85792 |
| v009 | DAPT: MLM on train+test text, 6 ep (`kaggle/mlm_pretrain.py`, `--init`), + TF-IDF, v007 rule (287,378,288) | fold 0 blend 0.9085 / practice 0.8999 (v007 0.9057 / 0.8935) | 0.86359 |
| v010 | v007 HingRoBERTa + DAPT HingRoBERTa, both trimmed, + TF-IDF 1/1/1 (299,498,920) | fold 0 0.9083 / practice 0.9122 | 0.86215 |
Takeaway: after v007, local gains (+0.004 to +0.019) never transferred to public; MuRIL never helps on public.
- **MuRIL 5-fold (`muril_lr6`)**: folds 0.879 / 0.894 / 0.893 / 0.888 / 0.876; OOF 0.8909 (tuned 0.8927, test-like est. LB ~0.827).
- **MuRIL + TF-IDF 0.5/0.5, 5-fold OOF**: raw 0.9115 → tuned 0.9150; test-like est. LB **~0.848** (best real 0.819). `src/assemble.py` builds the compliant version from the full-data models (246,890,864 params).
- MuRIL needs a higher LR than the defaults; ~63 s/epoch on T4, still creeping up at epoch 10.
- TF-IDF and MuRIL make different errors: blending added +0.02–0.03 OOF. `baseline.py --save_run runs/tfidf_aug` writes TF-IDF folds (same splits, temperature-scaled SVM margins).
- **Param cap applies to what produces the submission**: 5-fold MuRIL average = 1.19B (not allowed). Allowed final: 1 full-data MuRIL (237,599,288) + 1 TF-IDF SVM (9,291,576 = 165,920 features × 56) = 246.9M. Fold runs and HingRoBERTa are training-time tools only (OOF tuning, pseudo-label teachers).

## 3. Model choice (`analysis/model_scan.py` + literature)
| model | params | tokens/word dev / roman | decision |
|---|---|---|---|
| `google/muril-base-cased` | 238M | **1.13 / 1.38** | **primary** |
| `l3cube-pune/hing-roberta-mixed` | 278M | 1.29 / 1.56 | **second** (best Hinglish results in the papers we found) |
| `ai4bharat/IndicBERTv2-MLM-Sam-TLM` | 278M | 1.10 / 1.55 | third, if GPU time allows |
| `microsoft/mdeberta-v3-base` | 278M | 1.90 / 1.65 | low priority (fp16 overflow on T4, no Hinglish evidence) |
| `xlm-roberta-base` | 278M | 1.29 / 1.56 | skip (HingRoBERTa is XLM-R + Hinglish pretraining) |
| `jhu-clsp/mmBERT-base` | 307M | — | over the cap |
Only one 238–278M model fits under 300M, so a multi-model blend must be **distilled** into one student for the final/demo model.

## 4. Pipeline stages
| # | stage | where | status |
|---|---|---|---|
| 1 | EDA + shift-aware validation (clean + `val_shift` = augmented val) | CPU | done |
| 2 | normalisation + test-style augmentation | CPU | done |
| 3 | TF-IDF safety submission | CPU | done, **upload and record LB** |
| 4 | fine-tune MuRIL and HingRoBERTa-mixed, 5 folds | **Kaggle GPU** | **next** |
| 5 | blend, tune class-55 `tau` + class biases | CPU | code ready |
| 6 | pseudo-label round: `postprocess.py --pseudo 0.92` → `train.py --pseudo data/pseudo_*.csv` | GPU | code ready |
| 7 | final single model ≤300M: best model `--full`, or distil blend probs into it | GPU | to build (distillation = soft-label loss, not yet coded) |
| 8 | ONNX export + int8, latency check <50 ms, Gradio demo (top-3 intents, confidence, OOS fallback, latency) | CPU | to build |
| 9 | 2-page write-up + PPT (organiser template) + reproducible training notebook | — | to build |

## 5. Kaggle workflow (reliable, resumable)
1. GitHub fine-grained token, this repo only, Contents read-only → Kaggle notebook **Add-ons → Secrets → `GITHUB_TOKEN`**.
2. Import `kaggle/train_runner.ipynb`; Settings: GPU T4 x2 / P100, Internet On; set `MODEL`.
3. **Save Version → Save & Run All (Commit)**: runs server-side, survives disconnects (~35–45 min per model).
4. Progress: version log shows per-epoch JSON (`val_clean`, `val_shift`). Results: download `runs_all.zip` → unzip into the repo root (it contains `runs/<run>/`).
5. Interrupted? Add the previous version's output as Input and commit again; finished folds are skipped.
6. One model per notebook; run MuRIL and HingRoBERTa in parallel copies.

**Pseudo-labelling approved by the organisers (23 Nov, official WhatsApp group):** allowed as long as labels come from our own models on test.csv and the result is reproducible. `src/make_pseudo.py` builds them (model labels only, no human labelling).

## 5b. Submission versioning (`src/subtrack.py`; Kaggle limit **5 per day**, resets 00:00 UTC = 05:30 IST, no lifetime cap)
Every candidate goes through the registry; never upload a raw `postprocess.py` output (it is overwritten on each run).
**Local LB proxy** (`src/testlike.py`, AUC 0.678 train-vs-test): on fold-0 rows it ranks TF-IDF 0.8724 < MuRIL 0.8802, matching the LB, with a steady offset of ~0.061 (MuRIL tuned estimate 0.822 vs real 0.819). Blend MuRIL+TF-IDF estimate ~0.842. The Roman-only subset ranks the models wrongly, so it is diagnostic only.
**Policy:** no exploratory uploads. Upload only a param-cap-compliant model whose full 5-fold OOF beats the last submitted version by ≥ +0.015. Planned: #3 full MuRIL + TF-IDF (tuned on 5-fold OOF), #4 same with pseudo-labels, #5 reserve.
1. Commit the code first (the registry records the commit hash and flags uncommitted `src/ kaggle/ analysis/` changes as `dirty=1`).
2. `subtrack.py add <csv> --stage <name> --note "..."`: validates ids/targets against `data/test.csv`, refuses duplicates, freezes it as `submissions/vNNN_<stage>.csv` (+ sidecar `.json`, OOF F1 read from it).
3. Upload that `vNNN` file on Kaggle → `subtrack.py submitted vNNN` (checks the daily quota, tags `sub-vNNN` on the commit) → `subtrack.py lb vNNN <score>`.
4. Commit `submissions/` and `git push origin main --tags`. `subtrack.py list` shows everything; pick the 2 final submissions from it.
Only `vNNN_*` files and `registry.csv` are tracked in `submissions/`; everything else there is gitignored scratch.

## 6. Next steps, in order (historical, 22–23 Nov; all done except distillation)
1. MuRIL 5 folds `--lr 6e-5 --llrd 0.95 --epochs 10 --run muril_lr6` (owner's Kaggle). HingRoBERTa 5 folds `--lr 5e-5 --epochs 10 --run hing_lr5` (teammate's Kaggle, branch `exp/<name>`).
2. MuRIL `--full` with the same flags (`--run muril_full`): weights for submission #3 and the demo.
3. Final-assembly script (to build): full MuRIL test logits + one full-data TF-IDF SVM, weights/tau/bias from 5-fold OOF blend, param-count assert → submission #3.
4. Pseudo-labels from the teacher blend (MuRIL folds + HingRoBERTa folds + TF-IDF, prob ≥ 0.92) → commit `data/pseudo_*.csv` → MuRIL `--full --pseudo /tmp/repo/data/pseudo_*.csv` → submission #4.
5. Pick 2 final submissions from #3/#4.
6. Demo track (parallel, needs the `--full` weights): ONNX + Gradio, measure latency on CPU.

## 7. Ideas not yet implemented (ranked by expected gain)
- **Distillation** of MuRIL+HingRoBERTa blend into one ≤300M student on train + test (soft labels) — usually beats either single model.
- ~~Domain-adaptive MLM~~: done as v009 (public 0.86359, below v007).
- **Translate-train**: Hindi→English copies of train rows via an open MT model (IndicTrans2, MIT) to teach English phrasings seen in test. Must be an open model, never a commercial API.
- **kNN / Mahalanobis OOS score** on `test_emb` vs train embeddings, combined with the softmax threshold.
- ~~Vocabulary trimming~~: done (`tools/trim_vocab.py`), used for v008 and v010; both lower than v007 on public.
- R-Drop / adversarial (FGM) training for noise robustness.

## 8. Risks
- Public LB is only 974 rows (~17 per class): noisy. Trust `val_shift` + LB together; don't chase small LB gains.
- `val_shift` uses our own augmenter, so it partly measures what we trained on. The real check is the LB.
- Class-55 threshold tuned on OOF is train-distribution; novel OOS in test is likely under-detected. Watch the predicted 55 share on test (train share 1.5%).
