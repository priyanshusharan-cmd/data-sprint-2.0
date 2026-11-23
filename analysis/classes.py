import sys, re
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from normalize import phonetic_key
from sklearn.feature_extraction.text import CountVectorizer
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
tr["key"] = tr.query_text.map(phonetic_key); te["key"] = te.query_text.map(phonetic_key)
cv = CountVectorizer(min_df=3, binary=True); X = cv.fit_transform(tr.key); voc = np.array(cv.get_feature_names_out())
y = tr.target.values; df = np.asarray(X.sum(0)).ravel() / len(tr)
for c in range(56):
    p = np.asarray(X[y == c].mean(0)).ravel()
    lift = p * np.log((p + 1e-9) / (df + 1e-9))
    print(f"{c:2d} n={np.sum(y==c):4d}: " + " ".join(voc[np.argsort(-lift)[:8]]))
words = ["yaar","yar","bhai","plz","please","zara","na","jaldi","hey","ok","karo","kar","do","the","to","of","ka","mein","me"]
print("\nfiller/English rate per row  train vs test:")
for w in words:
    a = tr.key.str.contains(rf"\b{w}\b").mean(); b = te.key.str.contains(rf"\b{w}\b").mean()
    print(f"  {w:7s} {a:.3f}  {b:.3f}")
print("rows with digits: train", tr.query_text.str.contains(r"\d").mean().round(3), "test", te.query_text.str.contains(r"\d").mean().round(3))
tv = set(" ".join(tr.key).split()); tt = " ".join(te.key).split()
print("test token OOV rate vs train vocab:", round(np.mean([t not in tv for t in tt]), 3))
