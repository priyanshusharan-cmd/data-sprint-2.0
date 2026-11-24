"""
Confident-learning style label-noise weights from out-of-fold predictions (no labels are changed, no test rows used).

A training row is "suspect" when the OOF probability of its given label is very low while another class is very
confident (e.g. an alarm-setting query labelled 55 that every fold calls class 49). Suspect rows get a reduced loss
weight in kaggle/train.py (--sample_weights), so they stop pulling the decision boundary.

    python src/noise_weights.py runs/muril_lr6 runs/tfidf_aug     # writes data/sample_weights.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import postprocess as pp

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--p_label", type=float, default=0.05, help="max OOF prob of the given label")
    ap.add_argument("--p_other", type=float, default=0.90, help="min OOF prob of the top other class")
    ap.add_argument("--weight", type=float, default=0.3, help="loss weight for suspect rows")
    ap.add_argument("--out", default=str(ROOT / "data/sample_weights.csv"))
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    pp.Y = y = tr.target.values
    loaded = [pp.load_run(r, len(tr)) for r in a.runs]
    mask = np.logical_and.reduce([c for _, _, c in loaded])
    assert mask.all(), "need full 5-fold OOF coverage"
    p = np.mean([o for o, _, _ in loaded], 0)
    p_lab = p[np.arange(len(y)), y]
    other = p.copy(); other[np.arange(len(y)), y] = 0
    suspect = (p_lab < a.p_label) & (other.max(1) > a.p_other)
    w = np.where(suspect, a.weight, 1.0)
    pd.DataFrame({"row": np.arange(len(y)), "weight": w}).to_csv(a.out, index=False)
    print(f"suspect rows: {suspect.sum()} / {len(y)} ({suspect.mean():.2%}) -> weight {a.weight}; wrote {a.out}")
    pairs = pd.Series([f"{l}->{o}" for l, o in zip(y[suspect], other.argmax(1)[suspect])]).value_counts().head(10)
    print("most common label -> predicted pairs among suspects:\n" + pairs.to_string())


if __name__ == "__main__":
    main()
