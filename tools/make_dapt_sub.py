"""
Build a submission from the DAPT full-data HingRoBERTa with v007's exact recipe: 50/50 with the full TF-IDF model and
v007's frozen class-55 rule (tau + per-class biases tuned on 5-fold OOF). Only the encoder's starting weights differ.

    python tools/make_dapt_sub.py runs/dapt_hing_full/full     # -> submissions/cand_dapt.csv (+ diff vs v007)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from postprocess import apply_rule, softmax  # noqa: E402


def main():
    full = Path(sys.argv[1])
    rule = json.loads((ROOT / "submissions/v007_final-hingfull+tfidf.json").read_text())
    te = pd.read_csv(ROOT / "data/test.csv")
    n_enc = json.loads((full / "metrics.json").read_text())["n_params"]
    n_svm = json.loads((ROOT / "runs/tfidf_aug/full/metrics.json").read_text())["n_params"]
    print(f"params {n_enc:,} + {n_svm:,} = {n_enc + n_svm:,}")
    assert n_enc + n_svm <= 300_000_000
    w = np.array(rule["weights"]) / sum(rule["weights"])
    p = w[0] * softmax(np.load(full / "test_logits.npy")) + w[1] * softmax(np.load(ROOT / "runs/tfidf_aug/full/test_logits.npy"))
    pred = apply_rule(p, np.array(rule["bias"]), rule["tau"])
    out = ROOT / "submissions/cand_dapt.csv"
    pd.DataFrame({"example_id": te.example_id, "target": pred}).to_csv(out, index=False)
    v7 = pd.read_csv(ROOT / "submissions/v007_final-hingfull+tfidf.csv").target.values
    print(f"wrote {out} | class-55 share {np.mean(pred == 55):.3f} | differs from v007 on {(pred != v7).sum()} of {len(pred)} rows")


if __name__ == "__main__":
    main()
