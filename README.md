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
1. **Kaggle (GPU):** import `kaggle/train_runner.ipynb`, set `MODEL`, Save & Run All (commit). Download `runs_all.zip`.
2. **Mac (CPU):** unzip into `runs/`, then `python src/postprocess.py runs/muril runs/hingroberta`.
3. Upload `submissions/*.csv` to Kaggle.

## Results
| model | CV macro-F1 | public LB |
|---|---|---|
| TF-IDF + LinearSVC | 0.8733 | — |
| TF-IDF + LinearSVC + augmentation | 0.8859 | — |

## Run the demo on another machine (no code changes needed)
1. `git pull` (or clone) this repo.
2. Get `dataspint_model_bundle_v007.zip` (1.2 GB, shared privately inside the team; never public) and unzip it **into the repo root**. It creates `runs/hing_full/` (HingRoBERTa brain, tokenizer, offline config) and `runs/tfidf_aug/full/` (TF-IDF model).
3. Python 3.10+:
   - Windows: `python -m venv .venv` → `.venv\Scripts\python.exe -m pip install -r requirements.txt` → `.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu` → `.venv\Scripts\python.exe -m pip install -r requirements-demo.txt`
   - macOS: `python3 -m venv .venv` → `.venv/bin/pip install -r requirements.txt -r requirements-demo.txt`
4. Check: `python src/infer.py --check` → must print `reproduces v007_final-hingfull+tfidf.csv: 100.0000% identical` and 287,378,288 params.
5. Demo: `python app/demo.py` → http://127.0.0.1:7860 (fonts need internet; the model runs offline).
