"""
Experiment: a second small linear model to fill the parameter headroom (300M - 287.38M = 12.6M -> <= ~225K features x 56).

    python analysis/linear2.py cnb        # writes runs/lin2_cnb/fold{k} + full, same format as baseline.py --save_run
    python analysis/linear2.py svmword
    python analysis/linear2.py sgdlog

Then blend: python src/postprocess.py runs/hing_lr5 runs/tfidf_aug runs/lin2_cnb --weights ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline import SEED, prep, with_aug  # noqa: E402

MAX_FEAT = 220_000


def make(kind):
    if kind == "cnb":
        vecs = [("key", CountVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, binary=True)),
                ("key", CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, binary=True))]
        clf = lambda: ComplementNB(alpha=0.3)
        score = lambda c, X: c.predict_log_proba(X) if hasattr(c, "predict_log_proba") else c.decision_function(X)
    elif kind == "svmword":
        vecs = [("key", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, sublinear_tf=True)),
                ("raw", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))]
        clf = lambda: LinearSVC(C=0.3, class_weight="balanced")
        score = lambda c, X: c.decision_function(X)
    elif kind == "sgdlog":
        vecs = [("key", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3, sublinear_tf=True)),
                ("key", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True))]
        clf = lambda: SGDClassifier(loss="log_loss", alpha=2e-6, class_weight="balanced", max_iter=30, tol=None, random_state=SEED)
        score = lambda c, X: c.decision_function(X)
    else:
        raise SystemExit(kind)
    return vecs, clf, score


def featurize(vecs, fit_df, others):
    fit, oth = [], [[] for _ in others]
    for col, v in vecs:
        fit.append(v.fit_transform(fit_df[col]))
        for i, d in enumerate(others):
            oth[i].append(v.transform(d[col]))
    return hstack(fit).tocsr(), [hstack(m).tocsr() for m in oth]


def main():
    kind = sys.argv[1]
    out = ROOT / f"runs/lin2_{kind}"
    tr = prep(pd.read_csv(ROOT / "data/train.csv")); te = prep(pd.read_csv(ROOT / "data/test.csv"))
    y = tr.target.values
    margins = np.zeros((len(tr), 56)); folds = []
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y)):
        vecs, mk, score = make(kind)
        fit_df = with_aug(tr.iloc[a], SEED + k)
        Xa, (Xb, Xt) = featurize(vecs, fit_df, [tr.iloc[b], te])
        c = mk().fit(Xa, fit_df.target.values)
        margins[b] = score(c, Xb); folds.append((b, score(c, Xt)))
        print(f"fold {k}: {Xa.shape[1]:,} feats, macro-F1 {f1_score(y[b], margins[b].argmax(1), average='macro'):.4f}", flush=True)
    margins -= margins.max(1, keepdims=True)

    def nll(t):
        z = margins * t; z = z - z.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))
    temp = min(np.concatenate([np.arange(0.05, 1, 0.05), np.arange(1, 41, 0.5)]), key=nll)
    print(f"CV macro-F1 {f1_score(y, margins.argmax(1), average='macro'):.4f} | temperature {temp:.2f}")
    for k, (b, tm) in enumerate(folds):
        d = out / f"fold{k}"; d.mkdir(parents=True, exist_ok=True)
        np.save(d / "val_idx.npy", b)
        np.save(d / "val_logits.npy", (margins[b] * temp).astype(np.float32))
        np.save(d / "test_logits.npy", ((tm - tm.max(1, keepdims=True)) * temp).astype(np.float32))
        (d / "metrics.json").write_text(json.dumps({"best_score": f1_score(y[b], margins[b].argmax(1), average="macro")}))

    vecs, mk, score = make(kind)
    fit_df = with_aug(tr, SEED)
    X, (Xt,) = featurize(vecs, fit_df, [te])
    c = mk().fit(X, fit_df.target.values)
    tm = score(c, Xt)
    n_params = int(X.shape[1] * 56 + 56)
    d = out / "full"; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "test_logits.npy", ((tm - tm.max(1, keepdims=True)) * temp).astype(np.float32))
    (d / "metrics.json").write_text(json.dumps({"n_params": n_params, "temperature": float(temp)}))
    print(f"full: {X.shape[1]:,} features -> {n_params:,} params ({'OK' if n_params <= 12_621_712 else 'OVER BUDGET'})")


if __name__ == "__main__":
    main()
