# Open-set intent classification for Hindi and Hinglish voice queries
**Anveshan Data Sprint 2.0, NLP track. Kaggle team: Priyanshu Sharan** · Members: _[add names]_

**Result:** public leaderboard macro-F1 **0.86786** (rank 1 at submission time; organiser baseline 0.520).
**Model:** fine-tuned HingRoBERTa-mixed + character TF-IDF/LinearSVC, averaged 50/50, with a class-55 fallback rule.
**Size:** 287,378,288 parameters (limit 300M). **Latency:** ~19 ms median per query on a laptop CPU.

## 1. Problem and the key observation
Each query must be mapped to one of 55 intents or class 55 (out of scope); the metric is unweighted macro-F1, so
small classes count as much as large ones. Exploratory analysis showed a strong train/test shift:

| | train (16,936) | test (4,912) |
|---|---|---|
| Devanagari / Roman / mixed script | 61% / 33% / 6% | 33% / 64% / 3% |
| Colloquial fillers (*bhai, yaar, please*), digits | ≈0% | ~3% each, 1.3% |

A classifier fitted on Devanagari text alone scored **0.06** macro-F1 on genuine Roman-script rows. The test is also
more English-heavy and casual than train: test queries had a mean nearest-train similarity of 0.58 vs 0.73 inside
train. Handling this shift drove every design decision.

## 2. Approach
**Normalisation and rule-based augmentation (`src/normalize.py`, `src/augment.py`).** A Devanagari→Latin
transliterator in the dataset's own spelling style (aa/ee/oo, schwa deletion), a phonetic key that folds Hinglish
spelling variants (*chahoonga ~ chahunga*), and test-style augmentation: romanisation, fillers, number words → digits,
and ASR-like noise. All rules are hand-written code; no generated sentences and no LLM were used for data.

**Component 1: character TF-IDF + LinearSVC (9.3M parameters).** Char 2–5-grams on cleaned text and on the phonetic
key plus word 1–2-grams on the key, trained on each row and two augmented views. SVM margins are converted to
probabilities with one temperature (4.5) fitted on out-of-fold log-loss. Cross-validated macro-F1 0.886.

**Component 2: fine-tuned HingRoBERTa-mixed (278.1M parameters).** `l3cube-pune/hing-roberta-mixed` (CC-BY-4.0) is
pretrained on code-mixed Hinglish in both scripts, which matches the test better than MuRIL (Indian-language
Wikipedia/web text). Mean-pooled encoder + linear head, multi-sample dropout, square-root inverse class weights,
label smoothing 0.05, layer-wise LR decay 0.95, cosine schedule, fp16, lr 5e-5, 10 epochs; each epoch 60% of rows are
replaced by an augmented view. It beat MuRIL on all five folds (mean 0.893 vs 0.886).

**Blend and decision rule.** The two probability vectors are averaged 50/50 (other weights scored lower). Then a
per-class log-probability bias (tuned on out-of-fold predictions and shrunk ×0.5) is added and the argmax taken;
if no supported intent reaches the confidence threshold τ the query falls back to class 55. On out-of-fold data the
tuned τ was 0, so the explicit class-55 training examples carry the fallback.

**Parameter cap.** The submission is produced by exactly one encoder and one SVM; averaging five fold models (1.19B
parameters) was never used for a submission. Checked with the organisers' assertion: 278,086,712 + 9,291,576 =
**287,378,288**.

## 3. Validation strategy
- **5-fold stratified CV** (seed 42) for every component; all blend weights and rules tuned only on out-of-fold
  predictions.
- **A test-like "practice exam" (`src/testlike.py`).** A train-vs-test classifier on script-neutral text (AUC 0.68)
  selects the 25% of training rows that look most like the test. Plain CV over-estimated the leaderboard by ~0.06;
  this subset ranked every submission correctly and predicted v005 within 0.004.
- **Leak checks.** A pseudo-labelling gain disappeared once the teacher could no longer see the validation rows; we
  dropped the method.
- **Upload rule.** Submit only a model that beats the best one on local evidence. Up to v007 each upload beat the
  last; afterwards local gains of +0.004 to +0.019 stopped transferring (section 4), so we kept v007.

## 4. Results
| Version | Model | 5-fold macro-F1 | Test-like | Public LB |
|---|---|---|---|---|
| baseline | organiser char TF-IDF + LinearSVC | — | — | 0.520 |
| v002 | + augmentation, phonetic key | 0.886 | 0.872* | 0.812 |
| v003 | MuRIL, one fold, tuned learning rate | 0.885* | 0.883* | 0.819 |
| v005 | full-data MuRIL + TF-IDF 50/50 | 0.915 | 0.909 | 0.845 |
| **v007** | **full-data HingRoBERTa + TF-IDF 50/50** | **0.917** | **0.919** | **0.868** |
| v008 | vocabulary-trimmed HingRoBERTa + MuRIL + TF-IDF (299.5M) | 0.921 | 0.923 | 0.858 |
| v009 | domain-adapted HingRoBERTa (MLM on task text) + TF-IDF | 0.909* | 0.900* | 0.864 |
| v010 | two trimmed HingRoBERTas + TF-IDF (299.5M) | 0.908* | 0.912* | 0.862 |

\* fold 0 only (v007 on fold 0: 0.906 / 0.894). Final picks: v007 and v009.

**Tried and dropped:** default MuRIL settings (0.738, under-trained); pseudo-labelling (no gain once leak-free);
nearest-neighbour vote (+0.002 CV, changed 16 test rows); a routed three-model blend (public 0.856); 14-epoch
training (worse on one of two folds); a third linear model in the spare 12.6M parameters (lower test-like);
int8 quantisation (changed 15% of answers). v008–v010 (table) won locally but not on public.

**Vocabulary trimming and domain-adaptive pretraining.** To fit two encoders under 300M we removed embedding rows
for tokens never seen in train, test or augmented train and outside Latin/Devanagari script (HingRoBERTa 278M →
145–151M) with bit-identical test outputs. Domain-adaptive pretraining continued HingRoBERTa's masked-word
training on the unlabelled task text for 6 epochs before fine-tuning; it learned faster and won on fold 0, but
did not beat v007 on public. We kept the simplest model that held up.

## 5. Robustness, fallback and the live system
- **Scripts:** the same model handles Devanagari, Roman Hinglish and mixed queries; augmentation teaches script
  variants explicitly.
- **Out of scope:** class 55 has 249 training examples plus the fallback rule. The labels put some assistant-like
  requests (for example *setting* an alarm) in class 55, and the model follows the official labels.
- **Latency:** the demo (`app/demo.py`) runs the exact submission pipeline offline. ONNX Runtime with six threads on
  the CPU performance cores gives ~19 ms median end to end (tokenisation, both models, rule), with about 0.2% of
  queries just over 50 ms on a Windows laptop. The answers are identical to the PyTorch model.

## 6. Compliance and reproducibility
- Data: only the provided `train.csv` / `test.csv`; no external datasets, no MASSIVE/SLURP, no test-label lookup,
  no hand labelling. Pretrained weights: HingRoBERTa-mixed (CC-BY-4.0); MuRIL (Apache-2.0) was used only in experiments.
- No commercial LLM APIs for inference, labels or synthetic data; augmentation is deterministic code.
- **`kaggle/reproduce_v007.ipynb`** rebuilds the submission from the competition data alone (GPU, ~20 min). A
  verification run on a Kaggle T4 produced a `submission.csv` **100% identical** to the submitted v007 file, with
  identical training losses and the parameter assertion passing.
