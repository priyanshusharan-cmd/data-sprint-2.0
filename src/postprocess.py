"""
Blend fold outputs of one or more runs, tune the out-of-scope rule and per-class biases on OOF, write a submission.

    python src/postprocess.py runs/muril                       # one model
    python src/postprocess.py runs/muril runs/hingroberta      # blend (equal weights)
    python src/postprocess.py runs/muril runs/hingroberta --weights 0.6 0.4 --pseudo 0.92
    python src/postprocess.py runs/muril_lr6 runs/tfidf_aug --search   # grid-search blend weights on OOF
    python src/postprocess.py runs/muril_lr6 runs/tfidf_aug --search --temps --group_vote --prior_em --stack

Opt-in extras (each prints its effect; keep only what improves the test-like exam):
  --temps       fit one softmax temperature per run on OOF log-loss before blending
  --group_vote  average probabilities over queries that are the same after removing mined fillers
                (13% of test rows are filler variants of one query; batch-only, the live demo predicts one query)
  --prior_em    Saerens EM: re-estimate test class priors from the posteriors and re-weight (label shift)
  --stack       logistic-regression stacker on the runs' OOF log-probs instead of a weighted average

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


def fit_temperature(d, n_train):
    """Temperature minimising OOF log-loss of a run's raw logits."""
    idx, lo = [], []
    for f in sorted(p for p in Path(d).glob("fold*") if (p / "metrics.json").exists()):
        idx.append(np.load(f / "val_idx.npy")); lo.append(np.load(f / "val_logits.npy"))
    idx, lo = np.concatenate(idx), np.concatenate(lo)
    y = Y[idx]

    def nll(t):
        z = lo / t
        z = z - z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))
    return min(np.r_[np.arange(0.3, 3.0, 0.05)], key=nll)


def load_run(d, n_train, temp=1.0):
    d = Path(d)
    folds = sorted(p for p in d.glob("fold*") if (p / "metrics.json").exists())
    if not folds:
        raise SystemExit(f"no finished folds in {d}")
    oof = np.full((n_train, 56), np.nan)
    tests = []
    for f in folds:
        idx = np.load(f / "val_idx.npy")
        oof[idx] = softmax(np.load(f / "val_logits.npy") / temp)
        tests.append(softmax(np.load(f / "test_logits.npy") / temp))
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


def group_ids(texts):
    """Group id per query: identical after clean() and removing mined filler words (groups of size 1 get -1)."""
    from augment import mine_fillers
    from normalize import clean
    pre, suf = mine_fillers()
    fill = {w for f in pre + suf for w in f.split()}
    core = pd.Series([" ".join(w for w in clean(t).split() if w not in fill) for t in texts])
    ok = core.str.split().str.len() >= 3
    size = core.map(core.value_counts())
    gid = pd.factorize(core)[0]
    gid[~(ok & (size >= 2)).values] = -1
    return gid


def group_vote(p, gid):
    p = p.copy()
    for g in np.unique(gid[gid >= 0]):
        m = gid == g
        p[m] = p[m].mean(0)
    return p


def prior_em(p, prior_train, iters=20, damp=0.5):
    """Saerens et al. (2002) EM for label shift; damped towards the train prior for stability."""
    q = prior_train.copy()
    for _ in range(iters):
        post = p * (q / prior_train)
        post /= post.sum(1, keepdims=True)
        q = damp * post.mean(0) + (1 - damp) * prior_train
    return post, q


def stack_fit(oofs, y, C=1.0):
    """Multinomial LR on concatenated log-probs; returns (5-fold OOF probs, model fitted on all rows)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    X = np.hstack([np.log(o + 1e-9) for o in oofs])
    out = np.zeros((len(y), 56))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=11).split(X, y):
        out[b] = LogisticRegression(C=C, max_iter=3000).fit(X[a], y[a]).predict_proba(X[b])
    return out, LogisticRegression(C=C, max_iter=3000).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", type=float, nargs="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--search", action="store_true", help="grid-search blend weights (step 0.1) on OOF macro-F1")
    ap.add_argument("--pseudo", type=float, default=0.0, help="write confident test rows (prob >= this) for round 2")
    ap.add_argument("--temps", action="store_true")
    ap.add_argument("--group_vote", action="store_true")
    ap.add_argument("--prior_em", action="store_true")
    ap.add_argument("--stack", action="store_true")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
    global Y
    Y = tr.target.values
    w = a.weights or [1.0] * len(a.runs)
    temps = [fit_temperature(r, len(tr)) if a.temps else 1.0 for r in a.runs]
    if a.temps:
        print("temperatures", dict(zip(a.runs, np.round(temps, 2))))
    loaded = [load_run(r, len(tr), t) for r, t in zip(a.runs, temps)]
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
    if a.stack:
        st_oof, st = stack_fit([o[mask] for o, _, _ in loaded], Y[mask])
        print(f"stacker: OOF raw {f1_score(Y[mask], st_oof.argmax(1), average='macro'):.4f} "
              f"vs weighted average {f1_score(Y[mask], oof[mask].argmax(1), average='macro'):.4f}")
        oof = oof.copy(); oof[mask] = st_oof
        test = st.predict_proba(np.hstack([np.log(t + 1e-9) for _, t, _ in loaded]))
    if a.group_vote:
        g_tr, g_te = group_ids(tr.query_text), group_ids(te.query_text)
        before = f1_score(Y[mask], oof[mask].argmax(1), average="macro")
        oof = oof.copy(); oof[mask] = group_vote(oof[mask], g_tr[mask])
        print(f"group vote: train groups cover {np.mean(g_tr >= 0):.3f} of rows, OOF raw {before:.4f} -> "
              f"{f1_score(Y[mask], oof[mask].argmax(1), average='macro'):.4f}; test groups cover {np.mean(g_te >= 0):.3f}, "
              f"changed {np.mean(group_vote(test, g_te).argmax(1) != test.argmax(1)):.3f} of test argmax")
        test = group_vote(test, g_te)
    if a.prior_em:
        prior = np.bincount(Y, minlength=56) / len(Y)
        new, q = prior_em(test, prior)
        top = np.argsort(-(q / prior))[:5]
        print(f"prior EM: changed {np.mean(new.argmax(1) != test.argmax(1)):.3f} of test argmax; "
              f"most up-weighted classes {dict(zip(top.tolist(), np.round(q[top] / prior[top], 2).tolist()))}")
        test = new
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
    (out.with_suffix(".json")).write_text(json.dumps({"runs": a.runs, "weights": w, "temps": [float(t) for t in temps],
                                                      "extras": {k: getattr(a, k) for k in ("temps", "group_vote", "prior_em", "stack")},
                                                      "oof_raw": raw, "oof_tuned": tuned,
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
