# Branch `improvements`: squeezing past v007 safely

Goal: beat v007 (public 0.86786, OOF 0.9169, test-like 0.9191) by any honest margin, without ever risking it.
Everything here is opt-in: with default settings the code reproduces v007 exactly (augmentation verified identical to
`main` on 12,000 samples, including the random-number sequence; `kaggle/train.py` defaults unchanged).

## Safety rules
1. v007 stays one of the two final picks no matter what.
2. A candidate is considered only if it beats v007 on 5-fold OOF **and** on the test-like rows (`src/compare_v007.py`,
   `src/phase1.py` print per-fold wins and a paired bootstrap).
3. The demo stays on v007 unless the candidate's pipeline passes `src/infer.py --check` (100% identical) and stays
   under 50 ms on the demo laptop.
4. Nothing here submits to Kaggle.

## Phase 1: models we already have (CPU, minutes)
On the machine with the run folders (`runs/hing_lr5`, `runs/muril_lr6`, `runs/tfidf_aug`, `runs/hing_full`, `runs/muril_full`):
```
python src/phase1.py            # estimates for every option + candidate files submissions/cand_p1_*.csv/.json
python src/phase1.py --trim     # + trims both encoders (writes runs/hing_full/trim, runs/muril_full/trim)
python src/infer.py --check --encoder runs/hing_full/trim runs/muril_full/trim --rule submissions/cand_p1_<option>.json
```
Options measured against v007 on the same OOF rows: (B) neighbour vote, (C) HingRoBERTa + MuRIL + TF-IDF with three
pre-declared weightings (TF-IDF keeps half; MuRIL at most half of the transformer half because it transferred worse to
the real test), (D) best C + neighbour vote. Pseudo-labels are not re-run: leak-free they gave 0.000 / -0.006.

**Careful trimming (`src/trim_vocab.py`).** Keeps every token that occurs in train, test and augmented training views,
all special tokens, and the 60,000 most frequent vocabulary pieces (so demo queries with new words still tokenize
normally); copies the kept embedding rows unchanged; keeps the pad id at its index (RoBERTa position ids depend on it);
then **asserts that the trimmed encoder predicts every test query exactly like the untrimmed one** (max logit
difference must be < 1e-3; measured 0.0). Sizes at `--top 60000`: HingRoBERTa 278.1M -> 135.3M, MuRIL 237.6M -> 136.9M;
both + TF-IDF = **281.5M** (cap 300M). The inference loader (`src/verify_model.py`) detects `keep.npy` and remaps token
ids automatically; `src/infer.py` and `app/demo.py` accept two encoders (`--encoder a b --rule r.json`) and run them in
parallel on ONNX Runtime.

## Phase 2: a second HingRoBERTa (GPU, ~1.5 h on Kaggle T4 x2)
`kaggle/cli_runner/run.py` runs two queues at once, one per GPU:
* GPU 0: v007's HingRoBERTa exactly (seed 42): 5 folds (OOF) + full model, then trim.
* then MuRIL (v005 settings) split across both GPUs (folds 0-2 + full on GPU 0, folds 3-4 on GPU 1), so the
  three-model option can be measured from this run alone if the laptop run folders are not available.
* GPU 1: domain-adaptive masked-LM training on the competition text (train + test queries, no labels; 3 epochs),
  then HingRoBERTa seed 7 from that encoder: 5 folds + full model, then trim.
Then `python src/compare_v007.py --base runs/hing_lr5 runs/tfidf_aug --base_w 0.5 0.5 --cand runs/hing_lr5 runs/hing2 runs/tfidf_aug --cand_w 0.25 0.25 0.5`.
The second model only has to be about as good as the first: averaging two different encoders is what gains.
If a candidate passes: `python tools/build_repro_v008.py --second hing2|muril --rule <cand>.json --csv <cand>.csv` writes
`kaggle/reproduce_v008.ipynb` (adds the neighbour-vote step automatically when the rule uses it).

## Demo latency: measured, and why the demo should stay on v007
Real architectures and tokenizers with random weights (same compute), ONNX Runtime, 300 test queries, Apple M4
(Arsh's i5-13500H measured v007 at p50 ~19 ms, close to the 22 ms here):

| pipeline | p50 | p95 | max | over 50 ms |
|---|---|---|---|---|
| v007: 1 HingRoBERTa + TF-IDF | 22.2 ms | 32.6 | 52.3 | 0.3% |
| 2 trimmed HingRoBERTa + TF-IDF (parallel, 3+3 threads) | 38.9 ms | 63.6 | 116.1 | 18.3% |
| trimmed HingRoBERTa + trimmed MuRIL + TF-IDF (parallel) | 37.9 ms | 60.3 | 101.4 | 13.3% |

Any two-encoder pipeline roughly doubles the query time and breaks the 50 ms target on 1 query in 6.
**Recommendation:** a two-encoder winner is used as the second leaderboard pick only; the live demo stays on v007
(one of our two picks, 100% reproducible, 0.3% over 50 ms). `src/infer.py` / `app/demo.py` can still run a
two-encoder candidate for code verification.

Open question for the organisers before submitting a Phase 2 model: masked-LM training uses the unlabelled test
text (pseudo-labels, which also use test text, were approved on 23 Nov).

## Also on this branch (opt-in, off by default)
`AUG_MODE=mined` (filler words and respellings mined from the competition text instead of fixed lists),
`kaggle/train.py --ema --fgm --importance --sample_weights --logit_adj --teacher/--kd_* --init_from`,
`kaggle/mlm_pretrain.py`, `src/noise_weights.py`, `src/postprocess.py --temps --group_vote --prior_em --stack`.
Checked and not worth it: group voting over filler variants (v007 already answers them consistently: only 5 groups,
17 test rows differ), more folds (the submitted model is the full-data one either way).
