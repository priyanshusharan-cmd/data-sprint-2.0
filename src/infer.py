"""
Inference pipeline used by the demo and by the final-submission check: one encoder + one TF-IDF SVM
(+ optional nearest-neighbour vote), the class-55 rule and biases tuned on 5-fold OOF.

    python src/infer.py --check          # rebuild TF-IDF, reproduce the v005 submission, time single queries
    python src/infer.py "kal subah 7 baje ka alarm laga do"

Components (all loaded from local files; no network at query time):
  encoder  runs/<run>/full/model.pt + tokenizer/   (kaggle/train.py --full)
  tfidf    runs/tfidf_aug/full/tfidf.joblib         (built here from train.csv exactly like baseline.py --aug)
  rule     submissions/v005_final-murilfull+tfidf.json  (weights, tau, per-class bias)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import baseline  # noqa: E402
from normalize import clean, phonetic_key  # noqa: E402
from verify_model import load as load_encoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OOS = 55
PARAM_CAP = 300_000_000
# v007 (public LB 0.86786): HingRoBERTa + TF-IDF. v005 (0.84464): encoder runs/muril_full/full,
# rule submissions/v005_final-murilfull+tfidf.json
DEFAULT = {"encoder": "runs/hing_full/full", "tfidf": "runs/tfidf_aug/full/tfidf.joblib",
           "rule": "submissions/v007_final-hingfull+tfidf.json", "knn": False}


def build_tfidf(out_path):
    """Refit the full-data TF-IDF + LinearSVC exactly as `baseline.py --aug` does (same seed, same augmentation)."""
    from sklearn.svm import LinearSVC
    tr = baseline.prep(pd.read_csv(ROOT / "data/train.csv"))
    fit_df = baseline.with_aug(tr, baseline.SEED)
    vecs = [(name, col, vec) for name, col, vec in baseline.vectorizers()]
    from scipy.sparse import hstack
    X = hstack([vec.fit_transform(col(fit_df)) for _, col, vec in vecs]).tocsr()
    clf = LinearSVC(C=0.5, class_weight="balanced").fit(X, fit_df.target.values)
    temp = json.loads((ROOT / "runs/tfidf_aug/full/metrics.json").read_text())["temperature"]
    bundle = {"vecs": [(name, vec) for name, _, vec in vecs], "clf": clf, "temp": temp}
    joblib.dump(bundle, out_path)
    return bundle


def pin_fast_cores(n_logical=8):
    """Keep inference on the first logical CPUs (the 4 hyper-threaded P-cores on hybrid Intel laptops) at high
    priority, so a query never waits on a slow E-core thread. Measured on the i5-13500H: max latency 42 ms and
    0 of 2,000 queries over 50 ms, vs spikes to 60-85 ms unpinned. No-op if psutil is missing."""
    try:
        import psutil
        proc = psutil.Process()
        cpus = list(range(min(n_logical, psutil.cpu_count() or n_logical)))
        proc.cpu_affinity(cpus)
        proc.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, "HIGH_PRIORITY_CLASS") else -5)
    except Exception:
        pass


def export_onnx(encoder, tok, out_path):
    """Export the fine-tuned encoder + head to ONNX (same weights; dynamic batch and length)."""
    enc = tok(["warm up query"], return_tensors="pt")
    torch.onnx.export(encoder, (enc["input_ids"], enc["attention_mask"]), str(out_path),
                      input_names=["ids", "mask"], output_names=["logits"],
                      dynamic_axes={"ids": {0: "b", 1: "t"}, "mask": {0: "b", 1: "t"}, "logits": {0: "b"}},
                      opset_version=17, dynamo=False)


class IntentPipeline:
    def __init__(self, encoder=DEFAULT["encoder"], tfidf=DEFAULT["tfidf"], rule=DEFAULT["rule"], knn=DEFAULT["knn"],
                 threads=None, engine="onnx", ort_threads=6):
        """engine: "onnx" (default) = ONNX Runtime, exporting model.onnx once if missing; "auto" = ONNX only if it exists;
        "pytorch" = always PyTorch."""
        pin_fast_cores()
        if threads:
            torch.set_num_threads(threads)
        self.encoder, self.tok, cfg = load_encoder(ROOT / encoder)
        self.max_len = cfg["max_len"]
        self.encoder_name = {"google/muril-base-cased": "MuRIL",
                             "l3cube-pune/hing-roberta-mixed": "HingRoBERTa"}.get(cfg["hf_name"], cfg["hf_name"])
        tpath = ROOT / tfidf
        self.tf = joblib.load(tpath) if tpath.exists() else build_tfidf(tpath)
        r = json.loads((ROOT / rule).read_text())
        self.w = np.array(r["weights"], dtype=float) / sum(r["weights"])
        self.tau, self.bias = r["tau"], np.array(r["bias"])
        self.knn = None
        if knn:
            import neighbors as nb
            tr = pd.read_csv(ROOT / "data/train.csv")
            self.knn = nb.KnnIndex(tr.query_text.values, tr.target.values)
        n_enc = sum(p.numel() for p in self.encoder.parameters())
        n_svm = int(self.tf["clf"].coef_.size + self.tf["clf"].intercept_.size)
        self.params = {"encoder": n_enc, "tfidf_svm": n_svm, "total": n_enc + n_svm}
        assert self.params["total"] <= PARAM_CAP, self.params
        # Same weights, faster engine: ONNX Runtime with 6 threads gave identical answers on all 4,912 test queries
        # and p50 18 ms / max 35 ms over 1,000 queries on the i5-13500H (PyTorch: p50 27 ms, spikes to 60+ ms).
        self.engine = "pytorch"
        onnx_path = ROOT / encoder / "model.onnx"
        if engine != "pytorch":
            if engine == "onnx" and not onnx_path.exists():
                try:  # one-time export (~1 min, needs the onnx package); otherwise stay on PyTorch
                    export_onnx(self.encoder, self.tok, onnx_path)
                except Exception as err:
                    print(f"ONNX export skipped ({err}); using PyTorch", flush=True)
            if onnx_path.exists():
                try:
                    import onnxruntime as ort
                    so = ort.SessionOptions()
                    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    so.intra_op_num_threads = ort_threads
                    self._sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                    self.engine = f"onnxruntime ({ort_threads} threads)"
                except ImportError:
                    pass
        import gc
        gc.collect(); gc.freeze()  # no garbage-collector pauses during queries
        for _ in range(5):
            self.predict("warm up query")

    @staticmethod
    def _softmax(z):
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    def _tfidf_logits(self, texts):
        from scipy.sparse import hstack
        d = pd.DataFrame({"raw": [clean(t) for t in texts], "key": [phonetic_key(t) for t in texts]})
        cols = {"raw_char": d.raw, "key_char": d.key, "key_word": d.key}
        X = hstack([vec.transform(cols[name]) for name, vec in self.tf["vecs"]]).tocsr()
        return self.tf["clf"].decision_function(X) * self.tf["temp"]

    def _encoder_logits(self, texts):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        if self.engine != "pytorch":
            return self._sess.run(None, {"ids": enc["input_ids"].numpy(), "mask": enc["attention_mask"].numpy()})[0]
        return self.encoder(enc["input_ids"], enc["attention_mask"]).float().numpy()

    def probs(self, texts):
        p = self.w[0] * self._softmax(self._encoder_logits(texts)) + self.w[1] * self._softmax(self._tfidf_logits(texts))
        if self.knn is not None:
            import neighbors as nb
            kp, best = self.knn.query(texts)
            p = nb.mix(p, kp, best)
        return p

    def decide(self, p):
        """Same rule as src/postprocess.py apply_rule: biased argmax, then fallback to 55 if no intent is confident."""
        pred = (np.log(p + 1e-9) + self.bias).argmax(-1)
        pred[p[:, :OOS].max(-1) < self.tau] = OOS
        return pred

    def predict(self, text):
        t0 = time.perf_counter()
        p = self.probs([text])
        label = int(self.decide(p)[0])
        ms = (time.perf_counter() - t0) * 1000
        top = np.argsort(-p[0])[:3]
        return {"label": label, "top3": [(int(c), float(p[0, c])) for c in top], "ms": ms,
                "out_of_scope": label == OOS}

    def predict_batch(self, texts, bs=64):
        return np.concatenate([self.decide(self.probs(texts[i:i + bs])) for i in range(0, len(texts), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--knn", action="store_true")
    ap.add_argument("--encoder", default=DEFAULT["encoder"])
    ap.add_argument("--rule", default=DEFAULT["rule"])
    a = ap.parse_args()
    t0 = time.perf_counter()
    pipe = IntentPipeline(encoder=a.encoder, rule=a.rule, knn=a.knn)
    print(f"loaded in {time.perf_counter() - t0:.1f}s | params {pipe.params}")
    if a.text:
        print(pipe.predict(a.text))
    if a.check:
        te = pd.read_csv(ROOT / "data/test.csv")
        ref = ROOT / ("submissions/v006_final-murilfull+tfidf+knn.csv" if a.knn else Path(a.rule).with_suffix(".csv"))
        sub = pd.read_csv(ref)
        mine = pipe.predict_batch(te.query_text.tolist())
        print(f"reproduces {ref.name}: {np.mean(mine == sub.target.values):.4%} identical "
              f"({int(np.sum(mine != sub.target.values))} differ of {len(te)})")
        ms = np.array([pipe.predict(t)["ms"] for t in te.query_text.tolist()[:200]])
        print(f"end-to-end single query (encoder + TF-IDF + rule): p50 {np.percentile(ms, 50):.1f} ms | "
              f"p95 {np.percentile(ms, 95):.1f} ms | max {ms.max():.1f} ms | under 50 ms: {np.mean(ms < 50):.1%}")


if __name__ == "__main__":
    main()
