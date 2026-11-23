"""
Final, rule-compliant submission: ONE model per component at inference (param cap <= 300M summed).

Blend weights, class-55 threshold tau and per-class biases are tuned on the 5-fold OOF predictions of each
component; they are then applied to the test logits of the single full-data model of that component.

    python src/assemble.py --oof runs/muril_lr6 runs/tfidf_aug --full runs/muril_full/full runs/tfidf_aug/full
    ... --knn      # + nearest-neighbour vote over training rows (src/neighbors.py), applied before the class rule

Each --full dir needs test_logits.npy and metrics.json with "n_params" (written by kaggle/train.py --full and
src/baseline.py --save_run). Writes submissions/final_<tag>.csv + .json (weights, tau, bias, params, estimate).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import neighbors as nb
import postprocess as pp
from testlike import testlike_scores

ROOT = Path(__file__).resolve().parents[1]
PARAM_CAP = 300_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True, help="5-fold run dirs, one per component")
    ap.add_argument("--full", nargs="+", required=True, help="full-data model dirs, same order as --oof")
    ap.add_argument("--weights", type=float, nargs="*", help="fixed blend weights (default: grid search on OOF)")
    ap.add_argument("--knn", action="store_true", help="add the nearest-neighbour vote (src/neighbors.py)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert len(a.oof) == len(a.full), "--oof and --full must list the same components in the same order"

    tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
    pp.Y = Y = tr.target.values

    # 1) parameter cap: sum over the models that actually run at inference
    n_params = [json.loads((Path(d) / "metrics.json").read_text())["n_params"] for d in a.full]
    total = sum(n_params)
    for d, n in zip(a.full, n_params):
        print(f"  {d}: {n:,} params")
    print(f"total inference params {total:,} (cap {PARAM_CAP:,})")
    assert total <= PARAM_CAP, "over the 300M parameter cap"

    # 2) tune on OOF
    loaded = [pp.load_run(r, len(tr)) for r in a.oof]
    mask = np.logical_and.reduce([c for _, _, c in loaded])
    assert mask.all(), "every OOF run must cover all train rows (5 finished folds)"
    if a.weights:
        w = a.weights
    else:
        grid = [g for g in itertools.product(range(11), repeat=len(a.oof)) if sum(g) == 10]
        w = list(max(grid, key=lambda g: f1_score(Y, sum(gi * l[0] for gi, l in zip(g, loaded)).argmax(1), average="macro")))
        w = [x / 10 for x in w]
    oof = sum(wi * l[0] for wi, l in zip(w, loaded)) / sum(w)
    raw = f1_score(Y, oof.argmax(1), average="macro")
    bias, tau = pp.tune(oof, Y)
    knn_info = None
    if a.knn:  # class rule tuned without it (as v005), vote mixed in before the rule; index = training rows only
        kp, best = nb.oof_knn(tr.query_text.values, Y)
        print(f"neighbour vote: OOF tuned without {f1_score(Y, pp.apply_rule(oof, bias, tau), average='macro'):.4f}", end=" ")
        oof = nb.mix(oof, kp, best)
        knn_info = {"k": nb.K, "gate": nb.GATE, "share": nb.SHARE, "temp": nb.TEMP}
    tuned_pred = pp.apply_rule(oof, bias, tau)
    tuned = f1_score(Y, tuned_pred, average="macro")
    tl = testlike_scores(Y, tuned_pred, mask)
    print(f"weights {w} | OOF raw {raw:.4f} -> tuned {tuned:.4f} (tau {tau}) | "
          f"test-like {tl.get('testlike', float('nan')):.4f} -> estimated public LB ~{tl.get('est_lb', float('nan')):.3f} (+-0.01)")

    # 3) apply to the single full-data models
    test = sum(wi * pp.softmax(np.load(Path(d) / "test_logits.npy")) for wi, d in zip(w, a.full)) / sum(w)
    assert test.shape == (len(te), 56)
    if a.knn:
        tk, tbest = nb.test_knn(tr.query_text.values, Y, te.query_text.values)
        test = nb.mix(test, tk, tbest)
        knn_info["test_rows_voted"] = float(np.mean(tbest >= nb.GATE))
        print(f"neighbour vote active on {np.mean(tbest >= nb.GATE):.2f} of test rows")
    pred = pp.apply_rule(test, bias, tau)
    fold_avg = sum(wi * l[1] for wi, l in zip(w, loaded)) / sum(w)
    agree = np.mean(pred == pp.apply_rule(fold_avg, bias, tau))
    print(f"test: class-55 share {np.mean(pred == pp.OOS):.3f} | agreement with fold-average predictions {agree:.3f}")

    tag = "+".join(Path(r).name for r in a.oof) + ("+knn" if a.knn else "")
    out = Path(a.out or ROOT / f"submissions/final_{tag}.csv")
    pd.DataFrame({"example_id": te.example_id, "target": pred}).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({
        "oof_runs": a.oof, "full_models": a.full, "weights": w, "tau": tau, "bias": bias.round(3).tolist(),
        "oof_raw": raw, "oof_tuned": tuned, "testlike": tl.get("testlike"), "est_lb": tl.get("est_lb"),
        "n_params": dict(zip(a.full, n_params)), "total_params": total, "fold_agreement": agree, "knn": knn_info}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
