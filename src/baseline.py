"""
CPU baseline: TF-IDF (char + word n-grams on raw text and on the phonetic key) + LinearSVC.

    python src/baseline.py            # 5-fold CV macro-F1 + writes submissions/baseline_tfidf.csv
    python src/baseline.py --aug      # + test-style augmentation (romanize / colloquial / digits / ASR noise)
    python src/baseline.py --aug --save_run runs/tfidf_aug   # also write fold outputs blendable by postprocess.py

--save_run writes <dir>/fold{k}/{val_idx,val_logits,test_logits}.npy + metrics.json in the same format and with
the same StratifiedKFold(5, shuffle, 42) splits as kaggle/train.py. SVM margins are scaled by one temperature
fitted on OOF log-loss so that softmax gives usable probabilities for blending.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from augment import augment  # noqa: E402
from normalize import clean, phonetic_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def vectorizers():
    return [
        ("raw_char", lambda d: d.raw, TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("key_char", lambda d: d.key, TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
        ("key_word", lambda d: d.key, TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ]


def featurize(fit_df, other_dfs):
    mats_fit, mats_other = [], [[] for _ in other_dfs]
    for _, col, vec in vectorizers():
        mats_fit.append(vec.fit_transform(col(fit_df)))
        for i, d in enumerate(other_dfs):
            mats_other[i].append(vec.transform(col(d)))
    return hstack(mats_fit).tocsr(), [hstack(m).tocsr() for m in mats_other]


def prep(df):
    df = df.copy()
    df["raw"] = df.query_text.map(clean)
    df["key"] = df.query_text.map(phonetic_key)
    df["is_dev"] = df.query_text.str.contains("[ऀ-ॿ]")
    return df


def with_aug(df, seed, copies=2):
    """Training rows plus `copies` augmented views of each (labels kept)."""
    rng = random.Random(seed)
    views = [df[["query_text", "target"]]]
    for _ in range(copies):
        v = df[["query_text", "target"]].copy()
        v["query_text"] = [augment(t, rng) for t in v.query_text]
        views.append(v)
    return prep(pd.concat(views, ignore_index=True))


def main():
    use_aug = "--aug" in sys.argv
    tr = prep(pd.read_csv(ROOT / "data/train.csv"))
    te = prep(pd.read_csv(ROOT / "data/test.csv"))
    y = tr.target.values

    save_run = Path(sys.argv[sys.argv.index("--save_run") + 1]) if "--save_run" in sys.argv else None
    oof = np.zeros(len(tr), dtype=int)
    margins = np.zeros((len(tr), 56))
    folds = []
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y)):
        fit_df = with_aug(tr.iloc[a], SEED + k) if use_aug else tr.iloc[a]
        Xa, (Xb, Xt) = featurize(fit_df, [tr.iloc[b], te] if save_run else [tr.iloc[b], te.iloc[:0]])
        clf = LinearSVC(C=0.5, class_weight="balanced").fit(Xa, fit_df.target.values)
        assert list(clf.classes_) == list(range(56))
        margins[b] = clf.decision_function(Xb)
        oof[b] = margins[b].argmax(1)
        folds.append((b, clf.decision_function(Xt) if save_run else None))
        print(f"fold {k}: macro-F1 {f1_score(y[b], oof[b], average='macro'):.4f}", flush=True)

    if save_run:
        def nll(t):
            z = margins * t
            z = z - z.max(1, keepdims=True)
            return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))
        temp = min(np.arange(1, 41, 0.5), key=nll)
        print(f"temperature {temp} (OOF NLL {nll(temp):.4f})")
        for k, (b, tm) in enumerate(folds):
            d = save_run / f"fold{k}"
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "val_idx.npy", b)
            np.save(d / "val_logits.npy", (margins[b] * temp).astype(np.float32))
            np.save(d / "test_logits.npy", (tm * temp).astype(np.float32))
            (d / "metrics.json").write_text(json.dumps({"best_score": f1_score(y[b], oof[b], average="macro"),
                                                       "temperature": temp, "aug": use_aug}))
        print("wrote fold outputs to", save_run)

    print(f"CV macro-F1 {f1_score(y, oof, average='macro'):.4f}")
    for flag, name in [(True, "Devanagari"), (False, "Roman")]:
        m = tr.is_dev.values == flag
        print(f"  {name:10s} rows {m.sum():5d}  macro-F1 {f1_score(y[m], oof[m], average='macro'):.4f}")

    fit_df = with_aug(tr, SEED) if use_aug else tr
    X, (Xt,) = featurize(fit_df, [te])
    clf = LinearSVC(C=0.5, class_weight="balanced").fit(X, fit_df.target.values)
    pred = clf.predict(Xt)
    if save_run:  # the single full-data SVM used at inference (param cap: coef_ + intercept_)
        d = save_run / "full"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "test_logits.npy", (clf.decision_function(Xt) * temp).astype(np.float32))
        n_params = int(clf.coef_.size + clf.intercept_.size)
        (d / "metrics.json").write_text(json.dumps({"n_params": n_params, "temperature": temp, "aug": use_aug}))
        print(f"full-data SVM: {n_params:,} params -> {d}")
    out = ROOT / ("submissions/baseline_tfidf_aug.csv" if use_aug else "submissions/baseline_tfidf.csv")
    pd.DataFrame({"example_id": te.example_id, "target": pred}).to_csv(out, index=False)
    print("wrote", out, "test Devanagari share", round(te.is_dev.mean(), 3))


if __name__ == "__main__":
    main()
