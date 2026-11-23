import sys, re, collections
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from normalize import clean, phonetic_key
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
DEV = re.compile("[ऀ-ॿ]"); LAT = re.compile("[A-Za-z]")
def script(s):
    d, l = bool(DEV.search(s)), bool(LAT.search(s))
    return "mixed" if d and l else "dev" if d else "roman" if l else "other"
for n, d in [("train", tr), ("test", te)]:
    d["script"] = d.query_text.map(script)
    d["ntok"] = d.query_text.str.split().str.len()
    d["nchar"] = d.query_text.str.len()
    print(f"== {n}: rows {len(d)}  script {d.script.value_counts(normalize=True).round(3).to_dict()}")
    print(f"   tokens mean {d.ntok.mean():.1f} p50 {d.ntok.median():.0f} p95 {d.ntok.quantile(.95):.0f} max {d.ntok.max()}  chars p95 {d.nchar.quantile(.95):.0f}")
    print(f"   exact dup rows within: {d.query_text.duplicated().sum()}")
# per-class script mix in train
ps = pd.crosstab(tr.target, tr.script, normalize="index").round(2)
print("train classes with <15% roman:", (ps.get("roman",0) < .15).sum(), "  >60% roman:", (ps.get("roman",0) > .6).sum())
print("class 55 script:", tr[tr.target==55].script.value_counts().to_dict())
# label conflicts
g = tr.groupby(tr.query_text.map(phonetic_key)).target.nunique()
print("phonetic-key groups with conflicting labels:", (g > 1).sum())
# exact / key overlap test<->train
tk = set(tr.query_text.map(phonetic_key)); te["key"] = te.query_text.map(phonetic_key)
print("test rows whose phonetic key appears in train:", te.key.isin(tk).mean().round(3), " exact text:", te.query_text.isin(set(tr.query_text)).mean().round(3))
# shift: nearest-train similarity, val-vs-train baseline vs test-vs-train
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), min_df=2, sublinear_tf=True)
rng = np.random.RandomState(0); idx = rng.permutation(len(tr)); va, fit = idx[:3000], idx[3000:]
Xf = vec.fit_transform(tr.query_text.iloc[fit].map(phonetic_key))
nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Xf)
for n, q in [("val(train)", tr.query_text.iloc[va]), ("test", te.query_text)]:
    s = 1 - nn.kneighbors(vec.transform(q.map(phonetic_key)))[0][:, 0]
    print(f"nearest-train cosine {n:10s}: mean {s.mean():.3f} p10 {np.quantile(s,.1):.3f} p25 {np.quantile(s,.25):.3f} share<0.5 {np.mean(s<.5):.3f}")
    if n == "test": te["nnsim"] = s
for sc in ["dev", "roman", "mixed"]:
    m = te.script == sc
    if m.any(): print(f"   test {sc:6s} n={m.sum():5d} nnsim mean {te.nnsim[m].mean():.3f}")
print("\n-- lowest-similarity test queries (likely OOS / heavy noise):")
for s in te.sort_values("nnsim").query_text.head(15): print("  ", s)
print("\n-- random test roman:"); [print("  ", s) for s in te[te.script=="roman"].query_text.sample(12, random_state=1)]
print("\n-- random train roman:"); [print("  ", s) for s in tr[tr.script=="roman"].query_text.sample(8, random_state=1)]
print("\n-- class 55 train samples:"); [print("  ", s) for s in tr[tr.target==55].query_text.sample(15, random_state=2)]
# char inventory oddities
ch = collections.Counter("".join(te.query_text)) ; cht = collections.Counter("".join(tr.query_text))
print("\nchars in test not in train:", [c for c in ch if c not in cht][:40])
