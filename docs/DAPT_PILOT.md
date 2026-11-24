# DAPT pilot runbook (branch `exp/dapt`)

**Idea:** before fine-tuning, let HingRoBERTa read the competition's own queries (train + test text, no labels, plus
rule-augmented train) with its original masked-word training. Then fine-tune exactly like v007. Same model size
(278,086,712), same demo speed, same code path; only the starting weights change. `main` is untouched.

## 1. Run on Kaggle (~55 min, one session)
1. Kaggle → Code → New Notebook → File → Import Notebook → `kaggle/dapt_pilot.ipynb` (from this branch).
2. Settings: Accelerator **GPU**, Internet **On**, Add-ons → Secrets → `GITHUB_TOKEN` attached.
   Add Input → competition data **anveshan-data-sprint-2-nlp**.
3. **Save Version → Save & Run All (Commit)**. Keep notebooks **private**.
4. When it finishes: Output tab → download `results_small.zip` (small) and `full_model.zip` (1.1 GB, needed only
   if the gate passes).

What the log should show: MLM loss falling each epoch (roughly 3 → 2), then fold 0 `best 0.89…`, then the full model.

## 2. Decide on the laptop (2 min)
```powershell
git checkout exp/dapt
# unzip results_small.zip so that runs\dapt_hing_ft\fold0 and runs\dapt_hing_full\full exist
.venv\Scripts\python.exe tools\compare_fold.py runs\dapt_hing_ft\fold0
```
v007's fold 0 (50/50 with TF-IDF): **0.9057** all rows, **0.8935** practice exam.
The script prints `GATE PASSED` only if the new model beats both by more than +0.002. **If it fails, stop**:
keep v007 + v005 and spend the time on the demo.

## 3. If the gate passes: build and upload
```powershell
.venv\Scripts\python.exe tools\make_dapt_sub.py runs\dapt_hing_full\full
.venv\Scripts\python.exe src\subtrack.py add submissions\cand_dapt.csv --stage dapt-hing+tfidf --note "DAPT HingRoBERTa + TF-IDF, v007 rule"
```
Upload the new `submissions\v0NN_dapt-hing+tfidf.csv`. If its public score beats **0.86786**, tick it + v007 as finals
(drop v005). The demo can then switch to the new model (copy `runs\dapt_hing_full\full` over, same format as
`runs\hing_full\full`), but only after `src\infer.py --encoder runs\dapt_hing_full\full --rule submissions\v0NN_....json --check`
passes; otherwise keep the demo on v007.

## Notes for the defense, if used
- Only competition text (train + test queries, without labels) was used for pretraining; no external data, no LLM.
- Test-query text was read unlabelled, like pseudo-labelling, which the organisers allowed; no labels were looked up.
- Sanity check done: `make_dapt_sub.py` applied to v007's own encoder reproduces v007 with 0 differences.
