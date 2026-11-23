# Data Sprint 2.0 — NLP track (Anveshan, NIE Mysuru)

Open-set intent classification of Hindi / Hinglish voice queries. Kaggle: https://www.kaggle.com/competitions/anveshan-data-sprint-2-nlp
Read `docs/PLAN.md` first: it has the pipeline stages, what is done, what is next, and the measured findings.

## Task essentials
- 56 classes (0–54 intents, **55 = out of scope**), metric **unweighted macro-F1** (every class 1/56).
- Public LB 974 rows (19.8%), private 3,938 rows (80.2%). Final = 40% private LB + **60% live demo** (Hinglish robustness, class-55 fallback, **<50 ms inference**, code verification, reproducible training notebook).
- Deadline: Unstop says **24 Nov 2026, 9 PM IST**; Kaggle page shows a later close. Plan for 24 Nov 9 PM.

## Hard rules (disqualification risks)
- **≤300M parameters total at inference**, ensembles summed: `assert sum(p.numel() for m in models for p in m.parameters()) <= 300_000_000`.
- **No commercial LLM APIs** (OpenAI, Anthropic, Gemini, Cohere) for inference, labels or **synthetic data**. Claude Code may write code, but must never write training sentences, labels or templates of example queries. Augmentation stays rule-based (`src/augment.py`).
- **Never use Amazon MASSIVE / SLURP** or any checkpoint fine-tuned on them; never look up test queries externally; no hand-labelling of test rows.
- No code/data sharing outside the team. Keep this repo **private** during the event.
- Pretrained models must be open-source licensed (MuRIL Apache-2.0, HingRoBERTa CC-BY-4.0, IndicBERTv2 MIT are fine).

## Git conventions (team decision)
- Commit with **your own git identity**. **Never add `Co-Authored-By: Claude` or any Claude/AI attribution** to commits or descriptions. Claude must not appear in the contributor list.
- Push **directly to `main`**. No pull requests.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # + torch transformers sentencepiece for kaggle/train.py
# data/ (Kaggle CSVs, 1.8 MB) is committed — repo must stay private
```

## Commands
| what | command | where |
|---|---|---|
| CPU baseline (CV + submission) | `.venv/bin/python src/baseline.py --aug` | Mac |
| GPU fine-tune, 5 folds, resumable | `python kaggle/train.py --model muril` (`hingroberta`, `indicbert`) | Kaggle via `kaggle/train_runner.ipynb` |
| Full-data model for demo | `python kaggle/train.py --model muril --full --epochs N` | Kaggle |
| Blend + tune + submission | `.venv/bin/python src/postprocess.py runs/muril runs/hingroberta [--pseudo 0.92]` | Mac |
| Smoke test trainer on CPU | `.venv/bin/python kaggle/train.py --model e5small --limit 400 --epochs 1 --n_folds 2 --folds 0 --run smoke` | Mac |

## Code map
- `src/normalize.py` — Devanagari→Latin transliteration (from our Amazon ML Challenge 2026 repo) + Hinglish phonetic folding.
- `src/augment.py` — dataset-style romanizer (aa/ee/oo + schwa deletion), colloquial fillers, number→digit, ASR noise. `augment(text, rng)`.
- `kaggle/train.py` — encoder + mean-pool + multi-sample-dropout head; sqrt-inverse class weights, label smoothing, LLRD, cosine schedule, fp16; best epoch by mean of `val_clean` and `val_shift` macro-F1. Writes `runs/<model>/fold<k>/{val_idx,val_logits,val_emb,test_logits,test_emb}.npy` + `metrics.json` (written last = fold complete).
- `src/postprocess.py` — averages softmax over folds/runs, tunes class-55 confidence threshold `tau` + per-class log-prob biases (shrunk ×0.5) on OOF, writes `submissions/<runs>.csv`, `.probs.npy`, `.json`.
- `analysis/` — EDA, class keyword dump, augmentation experiment, model scan.
