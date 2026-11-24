"""
Gate for any new blend: compare it with v007 (HingRoBERTa + TF-IDF 0.5/0.5) on the same 5-fold OOF rows,
using exactly the tuning procedure of src/assemble.py (per-class biases + class-55 threshold tuned on all OOF rows).

    python src/compare_v007.py --base runs/hing_lr5 runs/tfidf_aug --base_w 0.5 0.5 \
                               --cand runs/hing_lr5 runs/hing2 runs/tfidf_aug --cand_w 0.25 0.25 0.5

Prints OOF and test-like ("practice exam") macro-F1 for both, the test-like score per fold, and a paired bootstrap
over the test-like rows (share of resamples where the candidate beats the base).
PASS only if the candidate is better overall on OOF and on test-like rows AND on test-like rows in every fold.
No test labels are used anywhere (they do not exist); everything is out-of-fold on train.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import postprocess as pp
from testlike import LB_OFFSET, testlike_mask

ROOT = Path(__file__).resolve().parents[1]


def blend(runs, w, n):
    loaded = [pp.load_run(r, n) for r in runs]
    assert np.logical_and.reduce([c for _, _, c in loaded]).all(), "every run needs 5 finished folds"
    return sum(wi * l[0] for wi, l in zip(w, loaded)) / sum(w)


def fold_of(run, n):
    f = np.full(n, -1)
    for k, d in enumerate(sorted(p for p in Path(run).glob("fold*") if (p / "metrics.json").exists())):
        f[np.load(d / "val_idx.npy")] = k
    return f


def evaluate(name, oof, y, tl, folds):
    bias, tau = pp.tune(oof, y)
    pred = pp.apply_rule(oof, bias, tau)
    res = {"raw": f1_score(y, oof.argmax(1), average="macro"), "tuned": f1_score(y, pred, average="macro"),
           "testlike": f1_score(y[tl], pred[tl], average="macro"),
           "fold_testlike": [f1_score(y[tl & (folds == k)], pred[tl & (folds == k)], average="macro") for k in range(5)]}
    print(f"{name:10s} OOF raw {res['raw']:.4f} tuned {res['tuned']:.4f} | test-like {res['testlike']:.4f} "
          f"(est. public ~{res['testlike'] - LB_OFFSET:.3f}) | per fold " + " ".join(f"{v:.4f}" for v in res["fold_testlike"]))
    return res, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--base_w", type=float, nargs="+", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--cand_w", type=float, nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    y = pd.read_csv(ROOT / "data/train.csv").target.values
    pp.Y = y
    tl = testlike_mask(len(y))
    folds = fold_of(a.base[0], len(y))
    assert (folds >= 0).all()
    b, pb = evaluate("base", blend(a.base, a.base_w, len(y)), y, tl, folds)
    c, pc = evaluate("candidate", blend(a.cand, a.cand_w, len(y)), y, tl, folds)
    wins = sum(cv > bv for cv, bv in zip(c["fold_testlike"], b["fold_testlike"]))
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tl)
    better = 0
    for _ in range(a.boot):
        s = rng.choice(idx, len(idx))
        better += f1_score(y[s], pc[s], average="macro") > f1_score(y[s], pb[s], average="macro")
    changed = np.mean(pc != pb)
    print(f"delta: OOF tuned {c['tuned'] - b['tuned']:+.4f} | test-like {c['testlike'] - b['testlike']:+.4f} | "
          f"test-like folds won {wins}/5 | bootstrap P(candidate better on test-like) {better / a.boot:.2f} | "
          f"OOF predictions changed {changed:.1%}")
    ok = c["tuned"] > b["tuned"] and c["testlike"] > b["testlike"] and wins == 5
    print("PASS: candidate beats the base everywhere" if ok else
          "NOT PASSED: keep the base" + (" (borderline: better overall, not in every fold)"
                                          if c["tuned"] > b["tuned"] and c["testlike"] > b["testlike"] else ""))


if __name__ == "__main__":
    main()
