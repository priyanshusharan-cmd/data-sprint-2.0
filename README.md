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
| `data/` | Kaggle CSVs + `testlike.csv` (practice-exam rows) (committed; repo is private) |
| `src/normalize.py` | Devanagari→Latin (reused from our Amazon ML Challenge 2026 pipeline) + Hinglish phonetic folding |
| `src/augment.py` | rule-based test-style augmentation: dataset-style romanizer, colloquial fillers, number→digit, ASR noise |
| `src/baseline.py` | TF-IDF + LinearSVC (`--aug`, `--save_run` writes 5-fold + full-data outputs) |
| `kaggle/train.py` | GPU fine-tuning, 5-fold or `--full`, resumable; `--init` starts from a local checkpoint |
| `kaggle/mlm_pretrain.py` | domain-adaptive masked-LM pretraining on the task text (used for v009) |
| `kaggle/*.ipynb` | Kaggle runners: `train_runner`, `dapt_pilot`, and **`reproduce_v007`** (for judges) |
| `src/postprocess.py`, `src/assemble.py` | blend runs, tune class-55 threshold + per-class biases on OOF, param-cap assert, write submission |
| `src/testlike.py` | the "practice exam": training rows that look most like the test |
| `src/subtrack.py` | submission registry (`submissions/registry.csv`) |
| `src/infer.py`, `app/demo.py` | the v007 pipeline on CPU (ONNX Runtime) and the Gradio demo |
| `tools/` | vocabulary trimming, fold-0 gate, notebook builder |
| `docs/` | `JOURNEY` (every experiment), `WRITEUP` (2 pages), `DEFENSE` (demo script + Q&A), `HANDOFF`, `PLAN` |

## Results (public leaderboard = 974 test rows)
| version | model | 5-fold macro-F1 | practice exam | public |
|---|---|---|---|---|
| baseline | organiser TF-IDF + LinearSVC | — | — | 0.520 |
| v002 | TF-IDF + augmentation | 0.8859 | 0.8724* | 0.81230 |
| v005 | MuRIL + TF-IDF | 0.9150 | 0.9086 | 0.84464 |
| **v007** | **HingRoBERTa + TF-IDF (287,378,288 params)** | **0.9169** | **0.9191** | **0.86786** |
| v008 | trimmed HingRoBERTa + MuRIL + TF-IDF | 0.9210 | 0.9226 | 0.85792 |
| v009 | domain-adapted HingRoBERTa + TF-IDF | 0.9105* | 0.9060* | 0.86359 |
| v010 | two trimmed HingRoBERTas + TF-IDF | 0.9083* | 0.9122* | 0.86215 |

\* fold 0 only. **Final picks: v007 + v009.** Details and lessons: [`docs/JOURNEY.md`](docs/JOURNEY.md).

## Final submission and how to reproduce it
- **v007 = HingRoBERTa-mixed (fine-tuned) + character TF-IDF/LinearSVC, 50/50**, 287,378,288 parameters, public LB **0.86786**.
- **`kaggle/reproduce_v007.ipynb`**: one self-contained Kaggle notebook (GPU + Internet on, competition data attached,
  ~25 min Run All). It embeds the exact source files, retrains TF-IDF and HingRoBERTa from `train.csv`, applies the
  frozen 5-fold-tuned rule, writes `submission.csv`, asserts the 300M cap and reports agreement with our v007 file.
  Regenerate it after code changes with `python tools/build_repro_notebook.py`.
- Full story of what we tried, kept and dropped: [`docs/JOURNEY.md`](docs/JOURNEY.md).

## Run the demo on another machine (no code changes needed)
1. `git pull` (or clone) this repo.
2. Get `dataspint_model_bundle_v007.zip` (1.2 GB, shared privately inside the team; never public) and unzip it **into the repo root**. It creates `runs/hing_full/` (HingRoBERTa brain, tokenizer, offline config) and `runs/tfidf_aug/full/` (TF-IDF model).
3. Python 3.10+:
   - Windows: `python -m venv .venv` → `.venv\Scripts\python.exe -m pip install -r requirements.txt` → `.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu` → `.venv\Scripts\python.exe -m pip install -r requirements-demo.txt`
   - macOS: `python3 -m venv .venv` → `.venv/bin/pip install -r requirements.txt -r requirements-demo.txt`
4. Check: `python src/infer.py --check` → must print `reproduces v007_final-hingfull+tfidf.csv: 100.0000% identical` and 287,378,288 params.
5. Demo: `python app/demo.py` → http://127.0.0.1:7860 (fonts need internet; the model runs offline).
