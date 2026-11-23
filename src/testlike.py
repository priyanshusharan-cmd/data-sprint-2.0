"""
Test-like local exam (adversarial validation). Uses test TEXT only: no test labels, no hand labelling, no new sentences.

A classifier learns to tell train queries from test queries on the script-neutral phonetic key (so it looks at
wording and content, not just Devanagari vs Roman). Train rows it mistakes for test rows are the most test-like
real examples we have; their out-of-fold predictions form a local exam that should track the public LB better
than plain CV.

    python src/testlike.py            # writes data/testlike.csv (row, p_test, testlike) and prints the AUC

Other scripts call testlike_scores(y, pred, covered) to report macro-F1 on all / test-like / test-like Roman rows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from normalize import phonetic_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/testlike.csv"
TOP_FRAC = 0.25  # share of train rows kept as the test-like exam
# Public LB ~= tuned test-like macro-F1 - LB_OFFSET. Real submissions so far (test-like -> public LB):
#   v002 TF-IDF raw 0.8724 -> 0.81230 (0.060)   v003 MuRIL fold0 tuned 0.8831 -> 0.81917 (0.064)
#   v005 full MuRIL + TF-IDF tuned 0.9086 -> 0.84464 (0.064)
# The two tuned pipelines (what postprocess/assemble print) agree on 0.064. Treat estimates as +-0.01.
LB_OFFSET = 0.064


def build():
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/test.csv")
    text = pd.concat([tr.query_text, te.query_text], ignore_index=True).map(phonetic_key)
    is_test = np.r_[np.zeros(len(tr), int), np.ones(len(te), int)]
    p = np.zeros(len(text))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(text, is_test):
        vecs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)]
        Xa = hstack([v.fit_transform(text.iloc[a]) for v in vecs]).tocsr()
        Xb = hstack([v.transform(text.iloc[b]) for v in vecs]).tocsr()
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xa, is_test[a])
        p[b] = clf.predict_proba(Xb)[:, 1]
    auc = roc_auc_score(is_test, p)
    pt = p[: len(tr)]
    flag = pt >= np.quantile(pt, 1 - TOP_FRAC)
    dev = tr.query_text.str.contains("[ऀ-ॿ]").values
    pd.DataFrame({"row": np.arange(len(tr)), "p_test": pt.round(4), "testlike": flag.astype(int)}).to_csv(OUT, index=False)
    print(f"train-vs-test AUC {auc:.3f} (0.5 = indistinguishable)")
    print(f"test-like rows: {flag.sum()} of {len(tr)} | Roman share {(~dev[flag]).mean():.2f} "
          f"(all train {(~dev).mean():.2f}, test {(~te.query_text.str.contains('[ऀ-ॿ]')).mean():.2f})")
    print("wrote", OUT.relative_to(ROOT))


def testlike_mask(n_train):
    if not OUT.exists():
        return None
    d = pd.read_csv(OUT)
    assert len(d) == n_train, "data/testlike.csv is stale; rerun src/testlike.py"
    return d.testlike.values.astype(bool)


def testlike_scores(y, pred, covered, texts=None):
    """Macro-F1 on covered rows: all and test-like (+ Roman-script test-like rows, diagnostic only:
    that subset ranked TF-IDF above MuRIL, the opposite of the LB, so don't decide on it)."""
    m = testlike_mask(len(y))
    out = {"all": f1_score(y[covered], pred[covered], average="macro")}
    if m is not None:
        tl = covered & m
        out["testlike"] = f1_score(y[tl], pred[tl], average="macro")
        if texts is not None:
            rom = ~pd.Series(texts).str.contains("[ऀ-ॿ]").values
            out["testlike_roman"] = f1_score(y[tl & rom], pred[tl & rom], average="macro")
        out["n_testlike"] = int(tl.sum())
        out["est_lb"] = out["testlike"] - LB_OFFSET
    return out


if __name__ == "__main__":
    build()
