"""
Blend fold outputs of one or more runs, tune the out-of-scope rule and per-class biases on OOF, write a submission.

    python src/postprocess.py runs/muril                       # one model
    python src/postprocess.py runs/muril runs/hingroberta      # blend (equal weights)
    python src/postprocess.py runs/muril runs/hingroberta --weights 0.6 0.4 --pseudo 0.92
    python src/postprocess.py runs/muril_lr6 runs/tfidf_aug --search   # grid-search blend weights on OOF

Each run dir holds fold{k}/{val_idx,val_logits,test_logits}.npy written by kaggle/train.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from testlike import testlike_scores

ROOT = Path(__file__).resolve().parents[1]
OOS = 55


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_run(d, n_train):
    d = Path(d)
    folds = sorted(p for p in d.glob("fold*") if (p / "metrics.json").exists())
    if not folds:
        raise SystemExit(f"no finished folds in {d}")
    oof = np.full((n_train, 56), np.nan)
    tests = []
    for f in folds:
        idx = np.load(f / "val_idx.npy")
        oof[idx] = softmax(np.load(f / "val_logits.npy"))
        tests.append(softmax(np.load(f / "test_logits.npy")))
    covered = ~np.isnan(oof[:, 0])
    print(f"{d.name}: {len(folds)} folds, OOF coverage {covered.mean():.2f}, "
          f"OOF macro-F1 {f1_score(Y[covered], oof[covered].argmax(1), average='macro'):.4f}")
    return oof, np.mean(tests, 0), covered


def apply_rule(p, bias, tau):
    pred = (np.log(p + 1e-9) + bias).argmax(1)
    pred[p[:, :OOS].max(1) < tau] = OOS  # low confidence on every in-scope intent -> fallback
    return pred


def tune(p, y):
    bias = np.zeros(56)
    taus = [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    best_tau = max(taus, key=lambda t: f1_score(y, apply_rule(p, bias, t), average="macro"))
    best = f1_score(y, apply_rule(p, bias, best_tau), average="macro")
    for _ in range(2):  # coordinate ascent on per-class log-prob offsets
        for c in range(56):
            for d in (-0.6, -0.3, 0.3, 0.6):
                b = bias.copy(); b[c] += d
                s = f1_score(y, apply_rule(p, b, best_tau), average="macro")
                if s > best + 1e-4:
                    best, bias = s, b
    return bias * 0.5, best_tau  # shrink: OOF is train-distribution, test is shifted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", type=float, nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--search", action="store_true", help="grid-search blend weights (step 0.1) on OOF macro-F1")
    ap.add_argument("--pseudo", type=float, default=0.0, help="write confident test rows (prob >= this) for round 2")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
    global Y
    Y = tr.target.values
    w = a.weights or [1.0] * len(a.runs)
    loaded = [load_run(r, len(tr)) for r in a.runs]
    mask = np.logical_and.reduce([c for _, _, c in loaded])
    if a.search and len(a.runs) > 1:
        import itertools
        grid = [g for g in itertools.product(range(11), repeat=len(a.runs)) if sum(g) == 10]
        def score(g):
            o = sum(gi * l[0][mask] for gi, l in zip(g, loaded))
            return f1_score(Y[mask], o.argmax(1), average="macro")
        res = sorted(((score(g), g) for g in grid), reverse=True)
        for sc, g in res[:5]:
            print(f"  weights {[x / 10 for x in g]} -> OOF raw macro-F1 {sc:.4f}")
        w = [x / 10 for x in res[0][1]]
        print(f"search picked weights {w} on {mask.sum()} OOF rows")
    oof = sum(wi * o for wi, (o, _, _) in zip(w, loaded)) / sum(w)
    test = sum(wi * t for wi, (_, t, _) in zip(w, loaded)) / sum(w)
    p, y = oof[mask], Y[mask]
    raw = f1_score(y, p.argmax(1), average="macro")
    bias, tau = tune(p, y)
    tuned = f1_score(y, apply_rule(p, bias, tau), average="macro")
    pred = apply_rule(test, bias, tau)
    print(f"blend OOF macro-F1 raw {raw:.4f} -> tuned {tuned:.4f} (tau {tau}, shrunk bias)")
    full = np.zeros(len(Y), int)
    full[mask] = apply_rule(p, bias, tau)
    tl = testlike_scores(Y, full, mask)
    if "testlike" in tl:
        print(f"test-like exam (tuned, {tl['n_testlike']} rows): {tl['testlike']:.4f} -> estimated public LB ~{tl['est_lb']:.3f} (+-0.01)")
    print(f"test: predicted class-55 share {np.mean(pred == OOS):.3f} (train share {np.mean(Y == OOS):.3f})")
    tag = "+".join(Path(r).name for r in a.runs)
    out = Path(a.out or ROOT / f"submissions/{tag}.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame({"example_id": te.example_id, "target": pred}).to_csv(out, index=False)
    np.save(out.with_suffix(".probs.npy"), test)
    (out.with_suffix(".json")).write_text(json.dumps({"runs": a.runs, "weights": w, "oof_raw": raw, "oof_tuned": tuned,
                                                      "tau": tau, "bias": bias.round(3).tolist(),
                                                      "testlike": tl.get("testlike"), "est_lb": tl.get("est_lb")}, indent=1))
    print("wrote", out)
    if a.pseudo:
        conf = test.max(1)
        keep = conf >= a.pseudo
        ps = pd.DataFrame({"query_text": te.query_text[keep], "target": test.argmax(1)[keep], "weight": conf[keep].round(3)})
        pth = ROOT / f"data/pseudo_{tag}.csv"
        ps.to_csv(pth, index=False)
        print(f"pseudo-labels: {keep.sum()} / {len(te)} test rows (prob >= {a.pseudo}) -> {pth}")


if __name__ == "__main__":
    main()
