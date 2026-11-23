"""Does augmentation close the script/style gap?  Honest test: fit on Devanagari rows, score genuine Roman rows."""
import sys, random
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from normalize import phonetic_key
from augment import romanize, augment, DEV

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
is_dev = tr.query_text.str.contains("[ऀ-ॿ]").values
rom = ~is_dev
romanvocab = set(" ".join(pd.concat([tr.query_text[rom], te.query_text]).str.lower()).split())
rng = random.Random(0)
r_new = [romanize(s, rng) for s in tr.query_text[is_dev]]
def hit(lst): t = " ".join(lst).split(); return np.mean([w in romanvocab for w in t])
from normalize import clean
print(f"token hit-rate vs real Roman vocab: dataset-style romanizer {hit(r_new):.3f}  |  Amazon transliterate_indic {hit([clean(s) for s in tr.query_text[is_dev]]):.3f}")

def raw_vec(): return TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2, sublinear_tf=True, lowercase=True)
def fit_eval(trX, trY, vaX, vaY, prep=lambda s: s):
    v = raw_vec(); A = v.fit_transform([prep(s) for s in trX]); B = v.transform([prep(s) for s in vaX])
    return f1_score(vaY, LinearSVC(C=0.5, class_weight="balanced").fit(A, trY).predict(B), average="macro")

Xd, yd = tr.query_text[is_dev].tolist(), tr.target[is_dev].values
Xr, yr = tr.query_text[rom].tolist(), tr.target[rom].values
print("\nE1  fit Devanagari-only -> score genuine Roman rows (macro-F1)")
print(f"  raw text, no aug                 {fit_eval(Xd, yd, Xr, yr):.4f}")
print(f"  raw text + romanized copies      {fit_eval(Xd + r_new, np.r_[yd, yd], Xr, yr):.4f}")
print(f"  phonetic_key (current baseline)  {fit_eval(Xd, yd, Xr, yr, phonetic_key):.4f}")
print(f"  phonetic_key + romanized copies  {fit_eval(Xd + r_new, np.r_[yd, yd], Xr, yr, phonetic_key):.4f}")

print("\nE2  5-fold CV, all rows; val scored clean and with a test-like shifted view (different seed)")
y = tr.target.values; X = tr.query_text.tolist()
res = {k: [] for k in ["base_clean", "base_shift", "aug_clean", "aug_shift"]}
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    if k > 1: break
    ra, rb = random.Random(100 + k), random.Random(900 + k)
    Xa = [X[i] for i in a]; Xb = [X[i] for i in b]; Xb_s = [augment(s, rb) for s in Xb]
    Xa_aug = Xa + [augment(s, ra) for s in Xa] + [augment(s, ra) for s in Xa]
    ya_aug = np.r_[y[a], y[a], y[a]]
    for tag, (TX, TY) in {"base": (Xa, y[a]), "aug": (Xa_aug, ya_aug)}.items():
        res[f"{tag}_clean"].append(fit_eval(TX, TY, Xb, y[b], phonetic_key))
        res[f"{tag}_shift"].append(fit_eval(TX, TY, Xb_s, y[b], phonetic_key))
for k, v in res.items(): print(f"  {k:11s} {np.mean(v):.4f}")
