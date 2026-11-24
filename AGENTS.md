# Instructions for coding agents (Codex and others)

This repo was built with Claude Code; `CLAUDE.md` holds the same rules for Claude. **Read, in order:**
1. `docs/HANDOFF.md`: current state, files, setup, what is open.
2. `docs/JOURNEY.md`: every experiment with its scores, including what failed (do not repeat those).
3. `CLAUDE.md`: commands, code map and the hard rules (summarised below).
4. `docs/PLAN.md`, `docs/WRITEUP.md`, `docs/DEFENSE.md` as needed.

## Task in one paragraph
Anveshan Data Sprint 2.0, NLP track (Kaggle `anveshan-data-sprint-2-nlp`): classify Hindi / Hinglish voice-assistant
queries into 55 intents or 55 = out of scope. Metric: unweighted macro-F1. Final grade: 40% private leaderboard + 60%
live demo (<50 ms per query, class-55 fallback, Hinglish robustness, code check, reproducible notebook).

## Current state (24 Nov 2026, evening IST)
- **Best: v007** = full-data HingRoBERTa-mixed + char TF-IDF/LinearSVC, 50/50, class-55 rule from
  `submissions/v007_final-hingfull+tfidf.json`. **287,378,288 params. Public 0.86786.** The demo
  (`app/demo.py`, `src/infer.py`) and the reproducible notebook (`kaggle/reproduce_v007.ipynb`) use v007.
- **Final Kaggle picks: v007 + v009** (v009 = domain-adapted HingRoBERTa + TF-IDF, public 0.86359).
- Every attempt after v007 (v008, v009, v010, a routed challenger) won on local validation and lost on public.
  See the "Every upload" table and "Lessons" in `docs/JOURNEY.md` before proposing anything new.
- Submission registry: `submissions/registry.csv`, managed by `src/subtrack.py`.

## Hard rules (disqualification risks)
- **≤300M parameters at inference**, summed over every model that produces a submission. Check with
  `assert sum(p.numel() for m in models for p in m.parameters()) <= 300_000_000` (+ the SVM's `coef_` + `intercept_`).
- **No commercial LLM APIs** (OpenAI, Anthropic, Gemini, Cohere) for inference, labels or synthetic data. **An AI
  agent may write code but must never write training sentences, labels or example queries.** Augmentation stays
  rule-based code (`src/augment.py`).
- Never use Amazon MASSIVE / SLURP or checkpoints fine-tuned on them; never look up test queries externally; no
  hand-labelling of test rows. Pretrained models must be open-source licensed.
- Keep the repo, notebooks and model files **private**. No sharing outside the team.

## Git conventions
- Commit under the human's git identity. **No AI attribution** in commits or descriptions (no `Co-Authored-By`).
- Experiments go on a branch `exp/<name>`; `main` must keep the verified v007 pipeline working
  (`python src/infer.py --check` must report 100% identical to v007). The repo owner merges winners into `main`.
- Never force-push or rewrite `main`.

## Before asking for a Kaggle upload
1. Evaluate on 5-fold OOF, or at least fold 0, against v007 with `tools/compare_fold.py` and the practice exam
   (`src/testlike.py`). Remember local gains have not transferred recently: require a clear margin.
2. Count parameters (assert ≤300M) and build the CSV with `src/assemble.py` or `tools/make_dapt_sub.py`.
3. Register it: `python src/subtrack.py add <csv> --stage <name> --note "..."`. Uploads are done by the owner
   (5 per day, resets 05:30 IST); afterwards `subtrack.py submitted vNNN` and `subtrack.py lb vNNN <score>`.

## Environment
Windows laptop, Python venv in `.venv` (`.venv\Scripts\python.exe`). GPU training runs on Kaggle via notebooks in
`kaggle/` that clone this private repo with a `GITHUB_TOKEN` secret (never print or commit the token).
Model weights live in `runs/` (git-ignored); see `docs/HANDOFF.md` for which are present.
