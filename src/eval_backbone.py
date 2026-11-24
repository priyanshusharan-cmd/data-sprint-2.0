"""
Fold-0 gate for a new backbone against v007, on the same rows, plus candidate files from its full-data model.

    python src/eval_backbone.py runs/indic_ft runs/indic_full [--write]

Compares on fold-0 validation rows (all rows and the practice-exam rows):
  v007 (HingRoBERTa + TF-IDF, arithmetic, v007 rule) and NEW-G (geometric, NEW-G rule)
  X + TF-IDF   arithmetic (v007 rule) and geometric (NEW-G rule)          -> a one-encoder swap (demo-safe size)
  H + X + TF-IDF geometric 0.25/0.25/0.5 (NEW-G rule)                     -> needs trimming, two encoders
With --write, writes submissions/cand_<name>_*.csv/.json for the X + TF-IDF options that beat v007 on both scores.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
import postprocess as pp  # noqa: E402
from testlike import testlike_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sm = pp.softmax


def geo(ps, ws):
    return sm(sum(w * np.log(p + 1e-9) for p, w in zip(ps, ws)) / sum(ws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold_run")
    ap.add_argument("full_run", nargs="?")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
    y = tr.target.values; tl = testlike_mask(len(y))
    r7 = json.loads((ROOT / "submissions/v007_final-hingfull+tfidf.json").read_text())
    rG = json.loads((ROOT / "submissions/cand_G_geo-hing+tfidf.json").read_text())
    b7, bG = np.array(r7["bias"]), np.array(rG["bias"])

    def f0(run):
        d = ROOT / run / "fold0"
        return np.load(d / "val_idx.npy"), sm(np.load(d / "val_logits.npy"))
    i0, H = f0("runs/hing_lr5"); iT, T = f0("runs/tfidf_aug"); iX, X = f0(a.fold_run)
    assert (i0 == iT).all() and (i0 == iX).all(), "fold-0 rows differ (different split?)"
    y0, t0 = y[i0], tl[i0]
    name = Path(a.fold_run).name
    rows = [("v007  H+T arith", 0.5 * H + 0.5 * T, b7), ("NEW-G H+T geo", geo([H, T], [.5, .5]), bG),
            (f"{name} alone", X, np.zeros(56)),
            (f"{name}+T arith", 0.5 * X + 0.5 * T, b7), (f"{name}+T geo", geo([X, T], [.5, .5]), bG),
            (f"H+{name}+T geo", geo([H, X, T], [.25, .25, .5]), bG)]
    res = {}
    print(f"fold-0 rows ({len(y0)}), practice-exam rows ({t0.sum()})")
    for nm, p, b in rows:
        pr = pp.apply_rule(p, b, 0.0)
        res[nm] = (f1_score(y0, pr, average="macro"), f1_score(y0[t0], pr[t0], average="macro"))
        print(f"  {nm:28s} all {res[nm][0]:.4f}  practice {res[nm][1]:.4f}")
    base = res["v007  H+T arith"]
    if a.full_run and a.write:
        v7 = pd.read_csv(ROOT / "submissions/v007_final-hingfull+tfidf.csv").target.values
        XT = sm(np.load(ROOT / a.full_run / "full/test_logits.npy"))
        TT = sm(np.load(ROOT / "runs/tfidf_aug/full/test_logits.npy"))
        nx = json.loads((ROOT / a.full_run / "full/metrics.json").read_text())["n_params"]
        for kind, p, rule in (("arith", 0.5 * XT + 0.5 * TT, r7), ("geo", geo([XT, TT], [.5, .5]), rG)):
            sc = res[f"{name}+T {kind}"]
            if not (sc[0] > base[0] and sc[1] > base[1]):
                print(f"  {name}+T {kind}: does not beat v007 on both scores -> no file"); continue
            pred = pp.apply_rule(p, np.array(rule["bias"]), 0.0)
            stem = ROOT / f"submissions/cand_{name}_{kind}+tfidf"
            pd.DataFrame({"example_id": te.example_id, "target": pred}).to_csv(stem.with_suffix(".csv"), index=False)
            total = nx + 9291576
            stem.with_suffix(".json").write_text(json.dumps({
                "option": f"{name} full model + TF-IDF, {kind} blend, rule of {'v007' if kind == 'arith' else 'NEW-G'}",
                "full_models": [f"{a.full_run}/full", "runs/tfidf_aug/full"], "blend": "geometric" if kind == "geo" else "arithmetic",
                "weights": [0.5, 0.5], "tau": 0.0, "bias": rule["bias"], "fold0_all": sc[0], "fold0_practice": sc[1],
                "fold0_v007": base, "n_params": {f"{a.full_run}/full": nx, "runs/tfidf_aug/full": 9291576},
                "total_params": total, "test_answers_changed_vs_v007": int(np.sum(pred != v7))}, indent=1))
            assert total <= 300_000_000
            print(f"  wrote {stem.name}.csv ({int(np.sum(pred != v7))} test answers differ from v007, {total:,} params)")


if __name__ == "__main__":
    main()
