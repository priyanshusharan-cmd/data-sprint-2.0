"""
Build kaggle/reproduce_candG.ipynb for candidate NEW-G: v007's exact pipeline (same TF-IDF, same HingRoBERTa training,
same seed) with ONE change: the two probability vectors are combined by a geometric mean (average of
log-probabilities) instead of an arithmetic mean, with the class-55 rule re-tuned on 5-fold OOF for that blend.

It starts from Arsh's verified v007 notebook builder (tools/build_repro_notebook.py, which rebuilt v007 100% on a
Kaggle T4) and only swaps the frozen rule, the blend line, the titles and the reference file.

    python tools/build_repro_geo.py --rule submissions/cand_G_geo-hing+tfidf.json --csv submissions/cand_G_geo-hing+tfidf.csv
"""
import argparse
import base64
import gzip
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(ROOT / "kaggle/reproduce_candG.ipynb"))
    a = ap.parse_args()
    rule = json.loads(Path(a.rule).read_text())
    assert rule.get("blend") == "geometric"

    spec = importlib.util.spec_from_file_location("v007nb", ROOT / "tools/build_repro_notebook.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    tmp = Path(a.out).with_suffix(".v007tmp.ipynb")
    base.OUT = tmp
    base.main()
    nb = json.loads(tmp.read_text(encoding="utf-8"))
    tmp.unlink()

    frozen = {"weights": rule["weights"], "tau": rule["tau"], "bias": rule["bias"], "blend": "geometric",
              "oof_tuned_macro_f1": round(rule["oof_tuned"], 5), "testlike_macro_f1": round(rule["testlike"], 5)}
    ref = base64.b64encode(gzip.compress(Path(a.csv).read_bytes())).decode()
    done = {"title": 0, "rule": 0, "rule_md": 0, "blend": 0, "ref": 0, "ref_md": 0}
    for c in nb["cells"]:
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if src.startswith("# Data Sprint 2.0 NLP: reproduce our final submission (v007)"):
            src = src.replace("# Data Sprint 2.0 NLP: reproduce our final submission (v007)",
                              "# Data Sprint 2.0 NLP: reproduce candidate NEW-G (v007 models, geometric blend)")
            src = src.replace("probabilities averaged 50/50,",
                              "combined by a **geometric mean** (average of log-probabilities, 50/50) instead of v007's "
                              "arithmetic mean,")
            src = re.sub(r"Public leaderboard: \*\*0\.86786\*\*\.", "Everything else is identical to v007.", src)
            done["title"] += 1
        elif src.startswith("RULE = "):
            src = "RULE = " + json.dumps(frozen) + "\n" + src.split("\n", 1)[1].replace("rule_v007.json", "rule_candG.json")
            done["rule"] += 1
        elif src.startswith("## 4) Frozen decision rule"):
            src = ("## 4) Frozen decision rule (tuned on 5-fold out-of-fold predictions of train.csv, geometric blend)\n"
                   f"Weights {frozen['weights']} on log-probabilities, class-55 threshold tau = {frozen['tau']}, 56 per-class "
                   f"log-probability biases. On the 16,936 OOF rows this rule gives macro-F1 {frozen['oof_tuned_macro_f1']} "
                   f"({frozen['testlike_macro_f1']} on the 4,234 most test-like rows); v007: 0.91689 / 0.91915.")
            done["rule_md"] += 1
        elif "p = w[0] * softmax(h) + w[1] * softmax(t)" in src:
            src = src.replace("p = w[0] * softmax(h) + w[1] * softmax(t)",
                              "p = softmax(w[0] * np.log(softmax(h) + 1e-9) + w[1] * np.log(softmax(t) + 1e-9))   # geometric mean")
            done["blend"] += 1
        elif src.startswith("import gzip, base64, io") and "V007 = '" in src:
            src = re.sub(r"V007 = '[^']*'", f"V007 = '{ref}'", src)
            done["ref"] += 1
        elif src.startswith("## 10) Agreement"):
            src = ("## 10) Agreement with the candidate file we produced (NEW-G)\n"
                   "GPU training is not bit-exact across machines, so expect a high but not perfect match.")
            done["ref_md"] += 1
        if "RECOMPUTE_RULE" in src and "postprocess.py runs/hing_lr5 runs/tfidf_aug" in src:
            src = src.replace("RECOMPUTE_RULE = False", "RECOMPUTE_RULE = False   # (re-derives the arithmetic-blend rule; "
                              "the geometric rule was tuned with the same procedure in our repo)")
        c["source"] = src
    assert all(v == 1 for v in done.values()), done
    Path(a.out).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", a.out, f"({len(nb['cells'])} cells) | replaced", done)


if __name__ == "__main__":
    main()
