"""
Pilot gate: compare a new HingRoBERTa fold against v007's (runs/hing_lr5) on the same fold, alone and blended 50/50
with TF-IDF, on all fold rows and on the test-like "practice exam" rows. The new run must win on BOTH blended numbers.

    python tools/compare_fold.py runs/dapt_hing_ft/fold0            # default fold taken from the dir name
    python tools/compare_fold.py runs/dapt_hing_ft/fold0 --base runs/hing_lr5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from postprocess import softmax  # noqa: E402
from testlike import testlike_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new", help="fold dir of the new run, e.g. runs/dapt_hing_ft/fold0")
    ap.add_argument("--base", default=str(ROOT / "runs/hing_lr5"))
    ap.add_argument("--tfidf", default=str(ROOT / "runs/tfidf_aug"))
    a = ap.parse_args()
    new = Path(a.new); k = new.name
    y = pd.read_csv(ROOT / "data/train.csv").target.values
    tl = testlike_mask(len(y))
    idx = np.load(new / "val_idx.npy")
    assert (idx == np.load(Path(a.base) / k / "val_idx.npy")).all(), "fold rows differ"
    t = softmax(np.load(Path(a.tfidf) / k / "val_logits.npy"))
    rows = {"all fold rows": np.ones(len(idx), bool), "practice exam": tl[idx]}
    print(f"{k}: {len(idx)} rows, {tl[idx].sum()} test-like")
    res = {}
    for label, d in [("v007 HingRoBERTa", Path(a.base) / k), ("new", new)]:
        h = softmax(np.load(d / "val_logits.npy"))
        for mode, p in [("alone", h), ("50/50 + TF-IDF", 0.5 * h + 0.5 * t)]:
            for rname, m in rows.items():
                res[(label, mode, rname)] = f1_score(y[idx][m], p[m].argmax(1), average="macro")
    print(f"{'':22s}{'all fold rows':>16s}{'practice exam':>16s}")
    for label in ["v007 HingRoBERTa", "new"]:
        for mode in ["alone", "50/50 + TF-IDF"]:
            print(f"{label[:8] + ' ' + mode:22s}" + "".join(f"{res[(label, mode, r)]:16.4f}" for r in rows))
    d_all = res[("new", "50/50 + TF-IDF", "all fold rows")] - res[("v007 HingRoBERTa", "50/50 + TF-IDF", "all fold rows")]
    d_tl = res[("new", "50/50 + TF-IDF", "practice exam")] - res[("v007 HingRoBERTa", "50/50 + TF-IDF", "practice exam")]
    print(f"blended change: all {d_all:+.4f} | practice exam {d_tl:+.4f}")
    print("GATE PASSED: worth the full model" if d_all > 0.002 and d_tl > 0.002 else "GATE FAILED: keep v007")


if __name__ == "__main__":
    main()
