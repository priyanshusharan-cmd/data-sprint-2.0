# Handoff: state on 24 Nov 2026 (afternoon IST)

For the teammate taking over, and for their Claude. Read this, then `CLAUDE.md` and `docs/PLAN.md`.
Everything below was verified on the original laptop before zipping.

## First thing to do (teammate's Claude)
1. `git checkout main && git pull origin main` to get the latest `main` (the zip may be older than GitHub).
2. Create your own branch for all experiments: `git checkout -b exp/<your-name>`. **Commit and push only to
   that branch** (`git push -u origin exp/<your-name>`). Never commit to, rebase or force-push `main`.
3. Leave `main` exactly as it is: it holds the verified v007 pipeline that the demo and the reproducible notebook use.
   Only the repo owner merges a winning change into `main`, after it beats v007 on 5-fold + the practice exam.
4. Uploads to Kaggle are done by the account owner only; prepare the CSV on your branch and send it.

## Where we stand
- **Public leaderboard #1: v007 = 0.86786.** v007 = full-data HingRoBERTa-mixed + full-data char TF-IDF/LinearSVC,
  probabilities averaged 50/50, then the class-55 rule (tau 0 + per-class biases) from
  `submissions/v007_final-hingfull+tfidf.json`. **287,378,288 parameters** (cap 300M).
- v005 = full MuRIL + TF-IDF, public 0.84464. Registry of all uploads: `submissions/registry.csv`
  (v001 … v007; v004 and v006 were never uploaded).
- **Deadlines:** Unstop final **24 Nov 9 PM IST**. Live demo + defense + slides **25 Nov**.
- **Kaggle final picks: v007 + v005 must be ticked** on Kaggle → Submissions before 9 PM IST (only the
  account owner can do it). Private score = the better of the two ticked picks.
- Kaggle allows 5 uploads per day (resets 05:30 IST).
- 24 Nov afternoon: a competitor moved above us on public at **0.87040** (+0.0025). On 974 public rows that is
  about 2–3 rows' worth of macro-F1, well inside noise; the private board (3,938 rows) decides. This is a reason
  to consider the open decision below, not to chase the public board with untested changes.

## Verified working
- `python src/infer.py --check`: the demo pipeline reproduces v007 **4,912/4,912 identical**; median latency
  ~20 ms, max 35 ms (laptop plugged in). **On battery, latency roughly doubles (40–46 ms)**: always demo plugged
  in, in best-performance power mode, with Chrome closed.
- `python app/demo.py` → http://127.0.0.1:7860 (Gradio, offline, ONNX Runtime, 6 threads on P-cores).
- `kaggle/reproduce_v007.ipynb`: re-run on a Kaggle T4 in 20.4 min gave a submission **100% identical** to v007.
- "Set an alarm" → 55 is **correct**: in train.csv, setting/querying alarms is labelled 55 (61 rows); only
  removing/turning off alarms is class 49 (81 rows). This is a defense talking point (`docs/DEFENSE.md`).

## Folder contents (what is in the zip)
| path | what | in git? |
|---|---|---|
| `runs/hing_full/full/` | v007 encoder: `model.pt` (1.1 GB), `model.onnx`, tokenizer, `encoder_config` | no |
| `runs/tfidf_aug/full/tfidf.joblib` | v007 TF-IDF + SVM (+ fold0–4 OOF outputs) | no |
| `runs/muril_full/full/` | v005 encoder (MuRIL, full data) + test logits | no |
| `runs/hing_lr5/`, `runs/muril_lr6/` | 5-fold OOF logits (used to tune rules / evaluate blends) | no |
| `data/` | competition CSVs + `testlike.csv` (local LB proxy) + old pseudo-label files (unused) | yes |
| `submissions/` | frozen v001–v007 files + registry | yes |
| `docs/` | PLAN, JOURNEY (every experiment with scores), WRITEUP (2-page), DEFENSE (demo script + Q&A) | yes |

Old experiment outputs were moved out of the project to `D:\data-sprint-archive-24nov` on the original laptop
(not needed). **Do not copy `.venv/` between machines**: recreate it (below).

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

## Open decision: three-model blend as the second Kaggle pick (NOT started)
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
