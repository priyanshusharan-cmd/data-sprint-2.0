# Handoff: state on 24 Nov 2026 (evening IST)

For the next person or coding agent (Claude, Codex). Read this, then `docs/JOURNEY.md`, `CLAUDE.md` / `AGENTS.md`.
Everything below was verified on the original laptop. **`main` now contains all work** (the `exp/dapt` branch was
merged in on 24 Nov: pretraining code, trimming tools, v008–v010). Zips sent to teammates are snapshots of `main`
without git history.

## First thing to do (teammate's Claude)
1. The zip has no `.git`. To use branches, clone the private repo
   (`git clone https://github.com/priyanshusharan-cmd/data-sprint-2.0.git`, branch `main`), then copy
   the zip's `runs/` folder into the clone. Without repo access, work in the zip folder and send changes back.
2. Do all your work on **your own branch**: `git checkout -b exp/<your-name>` from `main`.
   Commit and push only to that branch. Never commit to, rebase or force-push `main`.
3. `main` holds the verified v007 pipeline used by the demo and the reproducible notebook. Only the repo owner
   merges a winning change into `main`.
4. Kaggle uploads go through the owner (5 per day, resets 05:30 IST). Prepare the CSV and register it with
   `python src/subtrack.py add <csv> --stage <name> --note "..."`.
5. **Do not repeat what already failed** (table below), and gate every idea on 5-fold or at least fold 0
   + the practice exam (`src/testlike.py`) before asking for an upload.

## Where we stand
| version | what | params | fold-0 / OOF check | public |
|---|---|---|---|---|
| **v007** | HingRoBERTa (full) + TF-IDF 50/50, tuned rule | 287,378,288 | OOF 0.9169, practice 0.9191 | **0.86786** (best) |
| v008 | trimmed HingRoBERTa + trimmed MuRIL + TF-IDF | 299,498,920 | OOF 0.9210, practice 0.9226 | 0.85792 |
| v009 | DAPT HingRoBERTa (MLM on train+test text) + TF-IDF, v007 rule | 287,378,288 | fold0 blend 0.9085, practice 0.8999 (v007 fold0: 0.9057 / 0.8935) | 0.86359 |
| **v010** | trimmed v007 HingRoBERTa + trimmed DAPT HingRoBERTa + TF-IDF 1/1/1, v007 rule | 299,498,920 | fold0 0.9083, practice 0.9122 | 0.86215 |

- Leader on public: **0.87040** (Jihaan Jain). Public = 974 rows (20%); final ranking uses the private 80%.
- **Final picks (ticked by the owner before 9 PM IST 24 Nov): v007 + v009.** Private score = the better of the two ticked picks.
- Lesson so far: local gains of +0.003–0.006 did **not** transfer to public three times (challenger, v008, v009).
  MuRIL mixes never help on public. Demo = v007 (two encoders would break <50 ms).
- Deadlines: Unstop final **24 Nov 9 PM IST**. Live demo + defense + slides **25 Nov**.

## Verified working
- `python src/infer.py --check`: the demo pipeline reproduces v007 **4,912/4,912 identical**; median latency
  ~20 ms, max 35 ms (laptop plugged in). **On battery, latency roughly doubles (40–46 ms)**: always demo plugged
  in, in best-performance power mode, with Chrome closed.
- `python app/demo.py` → http://127.0.0.1:7860 (Gradio, offline, ONNX Runtime). `model.onnx` is not in the zip;
  `infer.py` exports it automatically on first start (about a minute, needs `onnx` installed).
- `kaggle/reproduce_v007.ipynb`: re-run on a Kaggle T4 in 20.4 min gave a submission **100% identical** to v007.
- "Set an alarm" → 55 is **correct**: in train.csv, setting/querying alarms is labelled 55 (61 rows); only
  removing/turning off alarms is class 49 (81 rows). Defense talking point (`docs/DEFENSE.md`).

## Small zip (27 MB): what is and is not included
The teammate zip holds **code, data, docs and every model's prediction files** (`test_logits.npy`,
`val_logits.npy`, `val_idx.npy`, `metrics.json`, `config.json`), which is all that is needed to evaluate and blend
new ideas against v007–v010 (`tools/compare_fold.py`, `src/postprocess.py`, `src/assemble.py`).
**Not included:** model weights (`model.pt`, ~1.1 GB each), tokenizers, `tfidf.joblib`, embeddings. Recreate when needed:
- TF-IDF model: `python src/baseline.py --aug --save_run runs/tfidf_aug` (CPU, a few minutes; same results).
- HingRoBERTa / DAPT weights: retrain on Kaggle (`kaggle/reproduce_v007.ipynb`, `kaggle/dapt_pilot.ipynb`).
- The demo needs `runs/hing_full/full/model.pt` + `runs/tfidf_aug/full/tfidf.joblib`: run it on the owner's laptop.

## Folder contents on the owner's laptop (full set)
| path | what |
|---|---|
| `runs/hing_full/full/` | v007 encoder (`model.pt`, tokenizer, `encoder_config`) — used by the demo |
| `runs/dapt_hing_full/full/` | DAPT HingRoBERTa full-data model (v009) |
| `runs/hing_full_trim/full/`, `runs/dapt_hing_full_trim/full/` | the two trimmed encoders of v010 (`id_map.npy` remaps token ids) |
| `runs/tfidf_aug/` | TF-IDF + SVM: `full/tfidf.joblib` (used by all submissions) + 5-fold OOF |
| `runs/hing_lr5/`, `runs/muril_lr6/` | 5-fold OOF logits of HingRoBERTa / MuRIL (rule tuning, blend checks) |
| `runs/dapt_hing_ft/fold0/` | DAPT HingRoBERTa fold 0 (only fold of the pilot) |
| `data/`, `submissions/` | competition CSVs + practice-exam rows; frozen v001–v010 + `registry.csv` |
| `kaggle/` | `train.py` (`--init` to start from a checkpoint), `mlm_pretrain.py`, `dapt_pilot.ipynb`, `reproduce_v007.ipynb` |
| `tools/` | `trim_vocab.py`, `compare_fold.py` (fold-0 gate), `make_dapt_sub.py`, notebook builder |
| `docs/` | this file, DAPT_PILOT, PLAN, JOURNEY, WRITEUP, DEFENSE |

MuRIL full models (v005/v008) are not included: MuRIL is a dead end on public. **Do not copy `.venv/`**.

## Setup on a new Windows machine
```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-demo.txt
.venv\Scripts\python.exe src\infer.py --check      # must print 100.0000% identical
.venv\Scripts\python.exe app\demo.py               # then open http://127.0.0.1:7860
```
Use `.venv\Scripts\python.exe`, not plain `python` (system Python lacks the packages), and not `python -m <path>`.
`infer.py` pins to the first 8 logical CPUs (P-cores of an i5-13500H); on another CPU it still works.

## Experiments done today (all dropped; details in docs/JOURNEY.md)
- Third small linear model in the 12.6M spare parameters (`analysis/linear2.py`: Naive Bayes, word SVM,
  logistic regression): practice exam lower than v007 on every weight. Dropped.

## v008 RESULT: public 0.85792 (worse than v007 0.86786)
Second time a HingRoBERTa + MuRIL mix improved validation and the practice exam but lost ~0.01 on public
(challenger 0.85605). Conclusion: adding MuRIL does not transfer to the test; do not try more MuRIL mixes.

## v008 details: trimmed three-model candidate
- File: `submissions/v008_3model-trim.csv` (registered in `submissions/registry.csv`).
- Models: `runs/hing_full_trim/full` (150,541,880) + `runs/muril_full_trim/full` (139,665,464) + TF-IDF
  (9,291,576) = **299,498,920 params**, built by `tools/trim_vocab.py`. Each trimmed model's test logits are
  **bit-identical** to the untrimmed model (max |diff| 0.0). Inputs use `id_map.npy` to remap token ids.
- Blend 0.35/0.3/0.35, rule tuned on 5-fold OOF: OOF 0.9210 (v007 0.9169), practice exam 0.9226 (v007 0.9191).
  251 of 4,912 test answers differ from v007.
- Plan: owner uploads v008 → `python src/subtrack.py submitted v008` → `lb v008 <score>`. If public ≥ v007, tick
  **v007 + v008** as finals (replace v005). If lower, keep v007 + v005.
- Not done for v008: demo integration (two encoders ≈ double latency; demo stays on v007) and a reproducible
  notebook (v007's notebook + MuRIL full training + `tools/trim_vocab.py` would be needed if v008 ends up winning).

## Background: the three-model analysis
HingRoBERTa + MuRIL + TF-IDF, scored on 5-fold OOF (`src/postprocess.py runs/hing_lr5 runs/muril_lr6 runs/tfidf_aug --weights ...`):

| weights hing/muril/tfidf | 5-fold tuned | practice exam |
|---|---|---|
| v007 (hing + tfidf) | 0.9169 | 0.9191 |
| 1/1/1 | 0.9200 | 0.9203 |
| 0.4/0.3/0.3 | 0.9200 | 0.9220 |
| 0.35/0.3/0.35 | 0.9210 | 0.9226 |

- Untrimmed it is 525M parameters. **Vocabulary trimming** (keep only Latin, Devanagari and digit tokens):
  HingRoBERTa 278M → 155.7M (keeps 90,585 of 250,002 tokens), MuRIL 238M → 143.6M (74,832 of 197,258). Each loses
  only 1 token that train/test use. Total with TF-IDF = **308.5M, still 8.5M over** → also drop ~11K of the
  rarest tokens never used in train/augmented train/test. Answers on the test set stay identical when every used
  token is kept, so the CSV can be built from the existing full-model test logits.
- Expected gain is small (+0.001 to +0.004 practice exam, about +0.003 public), within leaderboard noise. A
  previous 3-model "challenger" with a tuned routing rule scored **0.85605** public (worse than v007), so be careful.
- Two encoders double latency (the earlier challenger measured 44–70 ms), so **the demo stays on v007**.
- Safe use, if any: upload it and, if its public score is ≥ v007, tick it **instead of v005** (v005 can never
  beat v007, so that pick is otherwise wasted). Must be done before 9 PM IST, needs the trimmed models + the
  parameter assert to show compliance. If time is short, skip it: v007 + v005 is a fine final.

## Still to do
1. Account owner: tick final picks on Kaggle before 9 PM IST.
2. Slides in the organisers' template (source: `docs/WRITEUP.md`, `docs/JOURNEY.md`).
3. Add team member names in `docs/WRITEUP.md` (line 2).
4. Practise the Q&A in `docs/DEFENSE.md`.

## Rules that must not break (also in CLAUDE.md)
- ≤300M parameters at inference, summed over every model used for a submission.
- No commercial LLM APIs for inference, labels or data; Claude must not write training sentences, labels or
  example queries. No MASSIVE/SLURP. No hand-labelling of test rows.
- Commits under your own git identity, **no Claude/AI attribution**, push to `main`. Repo, notebooks and the model
  bundle stay **private** within the team.
- Verify every candidate on 5-fold + the practice exam (`src/testlike.py`) before uploading.
