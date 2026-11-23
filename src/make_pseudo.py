"""
Pseudo-labels (semi-supervised self-training): our own models label the test queries they are very confident
about; MuRIL is then retrained on train + those rows. Model-generated labels only: no human labelling of test rows,
no external data. Use in a submission only if the organisers confirm it is allowed.

    # leak-free experiment for fold k: the teacher excludes every model that saw fold k's validation rows
    python src/make_pseudo.py --runs runs/muril_lr6 runs/tfidf_aug --exclude_fold 0 --thresh 0.92
    # final round: teacher = all folds
    python src/make_pseudo.py --runs runs/muril_lr6 runs/tfidf_aug --thresh 0.92

Writes data/pseudo_<tag>.csv with columns query_text, target, weight (= teacher confidence), the format
kaggle/train.py --pseudo reads.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from postprocess import softmax

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="*")
    ap.add_argument("--exclude_fold", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=0.92)
    a = ap.parse_args()
    te = pd.read_csv(ROOT / "data/test.csv")
    w = a.weights or [1.0] * len(a.runs)
    probs = 0
    for wi, r in zip(w, a.runs):
        folds = sorted(p for p in Path(r).glob("fold*") if (p / "metrics.json").exists()
                       and p.name != f"fold{a.exclude_fold}")
        assert folds, f"no usable folds in {r}"
        probs = probs + wi * np.mean([softmax(np.load(f / "test_logits.npy")) for f in folds], 0)
        print(f"{r}: teacher folds {[f.name for f in folds]}")
    probs = probs / sum(w)
    conf, lab = probs.max(1), probs.argmax(1)
    keep = conf >= a.thresh
    rom = ~te.query_text.str.contains("[ऀ-ॿ]")
    tag = ("ex%d_" % a.exclude_fold if a.exclude_fold is not None else "all_") + f"t{int(a.thresh * 100)}"
    out = ROOT / f"data/pseudo_{tag}.csv"
    pd.DataFrame({"query_text": te.query_text[keep], "target": lab[keep], "weight": conf[keep].round(3)}).to_csv(out, index=False)
    print(f"kept {keep.sum()} / {len(te)} test rows (conf >= {a.thresh}) | classes covered {len(set(lab[keep]))}/56 | "
          f"Roman share kept {rom[keep].mean():.2f} (test {rom.mean():.2f}) | class-55 share {np.mean(lab[keep] == 55):.3f}")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
