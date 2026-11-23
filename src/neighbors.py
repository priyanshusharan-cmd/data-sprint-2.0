"""
Nearest-neighbour vote over the official labelled training rows (no test labels, no external data).

Each query is mapped to its script-neutral phonetic key; the 6 most similar training rows (cosine similarity on
char n-gram TF-IDF of the key) vote for their labels, weighted by similarity. When the best match is close
(similarity >= GATE) the vote (softmax(25 * similarity) over the 6 neighbours) gets SHARE of the final probability, otherwise the model blend decides alone.

For honest evaluation the index for fold k contains only the rows outside fold k (same StratifiedKFold(5, 42)
splits as kaggle/train.py), so a validation row can never find itself or be voted on by its own fold.

    python src/neighbors.py      # 5-fold check: v005 blend with vs without the neighbour vote
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from normalize import phonetic_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K, GATE, SHARE, TEMP, N_CLASSES = 6, 0.75, 0.25, 25.0, 56


def _vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, dtype=np.float32)


def knn_probs(index_keys, index_y, query_keys, k=K):
    """Return (probs [n, 56] from similarity-weighted top-k labels, best similarity [n])."""
    vec = _vec().fit(index_keys)
    A, Q = vec.transform(index_keys), vec.transform(query_keys)
    probs = np.zeros((len(query_keys), N_CLASSES))
    best = np.zeros(len(query_keys))
    for s in range(0, Q.shape[0], 512):
        sim = (Q[s:s + 512] @ A.T).toarray()
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        ts = np.take_along_axis(sim, top, 1)
        for i in range(len(top)):
            w = np.exp(TEMP * (ts[i] - ts[i].max()))  # softmax(25 * sim): the closest match dominates
            probs[s + i] = np.bincount(index_y[top[i]], weights=w / w.sum(), minlength=N_CLASSES)
            best[s + i] = ts[i].max()
    return probs, best


def oof_knn(train_text, y):
    keys = pd.Series(train_text).map(phonetic_key).values
    probs = np.zeros((len(y), N_CLASSES)); best = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(keys, y):
        probs[b], best[b] = knn_probs(keys[a], y[a], keys[b])
    return probs, best


def test_knn(train_text, y, test_text):
    tk = pd.Series(train_text).map(phonetic_key).values
    qk = pd.Series(test_text).map(phonetic_key).values
    return knn_probs(tk, y, qk)


def mix(model_probs, knn_p, best, gate=GATE, share=SHARE):
    use = best >= gate
    out = model_probs.copy()
    out[use] = (1 - share) * model_probs[use] + share * knn_p[use]
    return out


def main():
    import json
    import postprocess as pp
    from sklearn.metrics import f1_score
    from testlike import testlike_mask
    tr = pd.read_csv(ROOT / "data/train.csv"); y = tr.target.values; pp.Y = y
    m, _, _ = pp.load_run(ROOT / "runs/muril_lr6", len(y)); t, _, _ = pp.load_run(ROOT / "runs/tfidf_aug", len(y))
    blend = 0.5 * m + 0.5 * t
    kp, best = oof_knn(tr.query_text.values, y)
    tl = testlike_mask(len(y))
    cfg = json.loads((ROOT / "submissions/v005_final-murilfull+tfidf.json").read_text())
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    base = pp.apply_rule(blend, np.array(cfg["bias"]), cfg["tau"])
    print(f"rows with best similarity >= {GATE}: {np.mean(best >= GATE):.2f}")
    print(f"{'variant':34s} {'all':>7s} {'testlike':>9s}  per-fold all")
    def report(name, pred):
        pf = [f1_score(y[b], pred[b], average='macro') for _, b in folds]
        print(f"{name:34s} {f1_score(y, pred, average='macro'):.4f} {f1_score(y[tl], pred[tl], average='macro'):9.4f}  "
              + " ".join(f"{x:.4f}" for x in pf))
        return pf
    pf0 = report("v005 (blend + v005 bias/tau)", base)
    for gate, share in [(0.75, 0.25), (0.7, 0.25), (0.8, 0.25), (0.75, 0.15), (0.75, 0.35)]:
        mixed = mix(blend, kp, best, gate, share)
        p_fixed = pp.apply_rule(mixed, np.array(cfg["bias"]), cfg["tau"])
        pf = report(f"+knn gate {gate} share {share} (v005 bias)", p_fixed)
        if (gate, share) == (GATE, SHARE):
            fixed = np.sum((base != y) & (p_fixed == y)); broken = np.sum((base == y) & (p_fixed != y))
            print(f"   fixed {fixed}, broken {broken}, folds improved {sum(a > b for a, b in zip(pf, pf0))}/5")
            bias, tau = pp.tune(mixed, y)
            report(f"+knn {gate}/{share} (bias retuned)", pp.apply_rule(mixed, bias, tau))


if __name__ == "__main__":
    main()
