# Data Sprint 2.0 — NLP track

Open-set intent classification for Hindi / Hinglish voice queries: 56 classes (class 55 = out of scope),
unweighted macro-F1, model budget <= 300M parameters at inference.

## Key findings (analysis/)
- **Script flip:** train 61% Devanagari / 33% Roman; test 33% Devanagari / 64% Roman.
- **Register shift:** test has colloquial fillers (bhai, please, na, jaldi, hey), English content words and digits; train has almost none.
- Fitting on Devanagari only and scoring genuine Roman rows: raw text 0.06 macro-F1, +romanized copies 0.89 → augmentation is essential.

## Layout
| path | what |
|---|---|
| `data/` | Kaggle CSVs (committed; repo is private) |
| `src/normalize.py` | Devanagari→Latin (reused from our Amazon ML Challenge 2026 pipeline) + Hinglish phonetic folding |
| `src/augment.py` | test-style augmentation: dataset-style romanizer, colloquial fillers, number→digit, ASR noise |
| `src/baseline.py` | CPU TF-IDF + LinearSVC (`--aug` for augmentation) |
| `kaggle/train.py` | GPU fine-tuning, 5-fold, resumable per fold, clean + shifted validation |
| `kaggle/train_runner.ipynb` | Kaggle notebook that clones this repo and runs `train.py` |
| `src/postprocess.py` | blend runs, tune class-55 threshold + per-class bias on OOF, write submission / pseudo-labels |

## Workflow
1. **Kaggle (GPU):** import `kaggle/train_runner.ipynb`, set `MODEL`, Save & Run All (commit). Download `runs_<model>.zip`.
2. **Mac (CPU):** unzip into `runs/`, then `python src/postprocess.py runs/muril runs/hingroberta`.
3. Upload `submissions/*.csv` to Kaggle.

## Results
| model | CV macro-F1 | public LB |
|---|---|---|
| TF-IDF + LinearSVC | 0.8733 | — |
| TF-IDF + LinearSVC + augmentation | 0.8859 | — |
