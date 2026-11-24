# Our journey: from 0.520 to 0.868

Plain-language record of every step: what we tried, why, what it scored, and whether we kept it.
Scores: **validation** = 5-fold cross-validation macro-F1 on all 16,936 training queries (each scored by a model
that never saw it); **practice exam** = the same, on the 4,234 training queries that look most like the test
(`src/testlike.py`); **public** = Kaggle public leaderboard (974 test queries).

## The problem in one paragraph
Classify Hindi / Hinglish voice-assistant queries into 55 intents, or class 55 = out of scope. Unweighted macro-F1,
so every class counts equally. At most 300M parameters at inference. The key difficulty: **train is 61% Devanagari,
test is 64% Roman-script Hinglish**, and test wording is more casual and English-heavy than train.

## Kept steps (each one beat the previous)

| # | Change | Why | Validation | Practice exam | Public |
|---|---|---|---|---|---|
| 0 | Organiser baseline: char TF-IDF + LinearSVC, no augmentation | Reference floor | — | — | 0.520 |
| 1 | **Rule-based Hinglish augmentation** (Devanagari → dataset-style Roman spelling, fillers like *bhai/yaar*, number words → digits, typo noise) + phonetic spelling key. TF-IDF + LinearSVC | Train and test are in different scripts; the baseline never saw Roman Hinglish. Fitting on Devanagari only scored 0.06 on real Roman rows; with romanised copies 0.89 | 0.8859 | 0.8724* | **0.81230** (v002) |
| 2 | **MuRIL** fine-tuned (Google, 238M) with **learning rate 3e-5 → 6e-5**, layer decay 0.9 → 0.95, 10 epochs | A pretrained encoder understands meaning, not just spelling. Default settings stalled at 0.738 (under-trained) | 0.8846 (fold 0) | 0.8831* | **0.81917** (v003) |
| 3 | **MuRIL on all data + TF-IDF, averaged 50/50**, class-55 threshold and per-class biases tuned on 5-fold OOF. One model per component (5-fold averaging would be 1.19B params, over the cap) | The two models make different mistakes; blending fixes many (+0.022 validation) | 0.9150 | 0.9086 | **0.84464** (v005) |
| 4 | **HingRoBERTa-mixed** (L3Cube, 278M, pretrained on code-mixed Hinglish in both scripts) replaces MuRIL, + TF-IDF 50/50, **287,378,288 params** | 1,178 of our 1,390 validation errors were on queries with no close wording in train. HingRoBERTa already knows casual Hinglish; it beat MuRIL on all 5 folds (mean 0.8932 vs 0.8862) | 0.9169 | **0.9191** | **0.86786** (v007, #1) |

\* fold-0 rows only (844 test-like queries), before full 5-fold runs existed.

## Tried and dropped (with the evidence)

| Idea | Result | Decision |
|---|---|---|
| MuRIL with default settings (lr 3e-5, 6 epochs) | 0.738 fold 0, still rising | Fixed the learning rate instead |
| Pseudo-labelling (train on our own confident test predictions; organisers approved) | First test looked like +0.004, but the teacher had seen the validation rows (leak). Leak-free: TF-IDF 0.9149 vs 0.9150, MuRIL practice exam 0.8745 vs 0.8802 | Dropped: no real gain |
| Nearest-neighbour vote over training rows | +0.002 validation, +0.003 practice exam, but active on only 15% of test rows; changed 16 of 4,912 test answers | Kept as v006, not uploaded (gain below leaderboard noise) |
| Separate class-55 detector / bigger per-class bias tuning | Tuned threshold came out at 0; biases add ~0.003 | Kept the simple rule |
| MuRIL + HingRoBERTa + TF-IDF together | Practice exam 0.9220 (+0.003 over v007) but 515.7M params; would need vocabulary trimming to fit 300M | Not worth the complexity and demo risk |
| Three-model "compact challenger" (separate experiment): v007, but when its top-2 gap < 0.25 a 35/35/30 vote with a vocabulary-trimmed MuRIL; 298.9M params | Validation 0.9169 → 0.9197, practice exam 0.9191 → 0.9237, but the routing rule was tuned on the same rows; 44–70 ms per query. **Uploaded once: public 0.85605 (v007: 0.86786)** | Dropped; confirms that gains tuned on the evaluation rows don't transfer |
| **v008**: plain 35/30/35 average of HingRoBERTa + MuRIL + TF-IDF, both encoders vocabulary-trimmed (only tokens never used in train/test and outside Latin/Devanagari removed; outputs bit-identical), 299,498,920 params | Validation 0.9210, practice exam 0.9226 (v007: 0.9169 / 0.9191). **Uploaded: public 0.85792** | Dropped. Second MuRIL mix that won locally and lost ~0.01 on public: MuRIL's gains do not transfer to the test |
| **v009**: domain-adaptive pretraining: HingRoBERTa continues masked-word training on train + test query text (no labels) + augmented train for 6 epochs, then fine-tuned exactly like v007; + TF-IDF 50/50, v007 rule; 287,378,288 params | Fold 0 blended: all 0.9085 vs 0.9057, practice exam 0.8999 vs 0.8935 (gate passed). **Uploaded: public 0.86359** | Not better than v007 on public; kept as second final pick (best public after v007) |
| **v010**: v007's HingRoBERTa + the pretrained HingRoBERTa + TF-IDF 1/1/1, both encoders vocabulary-trimmed (outputs bit-identical), 299,498,920 params | Fold 0: all 0.9083, practice exam 0.9122 (v007 0.8935). **Uploaded: public 0.86215** | Dropped. Third time in a day that a local gain (+0.006 to +0.019 on the practice exam) lost 0.004–0.010 on public: we stopped tuning against local proxies and kept v007 |
| HingRoBERTa trained 14 epochs instead of 10 (best / final / average of epochs 12–14), folds 1–2 | Fold 1 blended practice exam up to +0.008, fold 2 down on every variant (−0.002 to −0.005) even though HingRoBERTa alone improved | Dropped; failed the "must win on both folds" rule |
| HingRoBERTa + TF-IDF with more weight on HingRoBERTa (60/40 … 80/20) | All lower than 50/50 | 50/50 |
| A third small linear model in the 12.6M spare parameters (`analysis/linear2.py`): Complement Naive Bayes (alone 0.803), word-level SVM (0.855), logistic regression on char+word TF-IDF (0.888, 3.7M params); also logistic regression replacing the SVM | Validation up to +0.001, but practice exam lower on every weight tried (0.916–0.919 vs v007's 0.9191) | Dropped; v007's two models already cover what a linear model can add |
| int8 quantisation for demo speed | No speed gain, changed 15% of answers | Rejected; used ONNX Runtime instead (identical answers) |
| Hindi-script-only TF-IDF, E5 reranker (from an external audit) | Lower, or over the parameter cap | Rejected |

## Lessons that shaped decisions
- **Measure on rows that look like the test.** Plain CV over-estimated the leaderboard by ~0.06; our practice exam ranked every upload correctly and predicted v005 within 0.004.
- **Leak-check every "free" gain.** The pseudo-label gain disappeared once the teacher could not see validation rows.
- **Upload only what beats the best on 5-fold evidence.** Four uploads, each better than the last: 0.812 → 0.819 → 0.845 → 0.868.
- **Respect the cap in the deployed model, not just the leaderboard file.** The demo runs the exact v007 pipeline: `src/infer.py --check` reproduces the submission 4,912/4,912.

## Reproducibility
`kaggle/reproduce_v007.ipynb` rebuilds v007 from the competition data alone. Run on a Kaggle T4 (24 Nov, 20.4 min):
TF-IDF CV 0.8859 and temperature 4.5 as in our run, HingRoBERTa losses identical epoch by epoch, parameter check
287,378,288, and the resulting `submission.csv` **100.00% identical to the submitted v007** (fixed seed 42).

## Demo engineering
- Same pipeline as the submission (`src/infer.py`), offline on a laptop CPU.
- ONNX Runtime with 6 threads pinned to the performance cores: p50 ~19 ms, ~0.17% of queries just over 50 ms (PyTorch: p50 ~27–35 ms, ~2% over).
- Note: in this dataset, *setting* an alarm is labelled out of scope (class 55); the model follows the official labels.
