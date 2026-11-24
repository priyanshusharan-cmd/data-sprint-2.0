"""
Phase 1: squeeze more out of the models we ALREADY have (no new training). One command, CPU only, a few minutes.
Run it on the machine that holds the run folders (fold outputs + full-data models):

    python src/phase1.py                     # estimates + candidate files
    python src/phase1.py --trim              # + lossless vocabulary trimming of both encoders (needs model.pt)

Inputs (defaults = our run names; override with flags):
  5-fold OOF   runs/hing_lr5  runs/muril_lr6  runs/tfidf_aug         (fold*/val_idx, val_logits, test_logits)
  full-data    runs/hing_full/full  runs/muril_full/full  runs/tfidf_aug/full   (test_logits.npy, metrics.json)

What it measures, every option against v007 on the SAME out-of-fold rows, with the assemble.py tuning procedure:
  A  v007 itself (sanity: must give OOF 0.9169 / test-like 0.9191 and reproduce the v007 file)
  B  v007 + nearest-neighbour vote (src/neighbors.py)
  C  HingRoBERTa + MuRIL + TF-IDF, three pre-declared weightings (TF-IDF keeps its half; MuRIL gets at most half of
     the transformer half because it transferred worse to the real test: public gap 0.064 vs 0.051)
  D  best C + neighbour vote
For each: OOF macro-F1, test-like ("practice exam") macro-F1, per-fold test-like, paired bootstrap P(better than
v007), estimated public score, and how many test answers change. Pseudo-labels are not re-estimated here: leak-free
they gave 0.000 (TF-IDF) and -0.006 (MuRIL, test-like) in docs/JOURNEY.md.

Three-model options need BOTH encoders vocabulary-trimmed to fit 300M. Trimming (src/trim_vocab.py) keeps every token of
train/test/augmented text plus the --top most frequent pieces and asserts identical predictions on all test queries,
so the candidate files written here (from the untrimmed test logits) are exactly what the trimmed models produce.
Writes submissions/cand_p1_<name>.csv + .json for every option that beats v007. Never submits anything.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
import neighbors as nb  # noqa: E402
import postprocess as pp  # noqa: E402
from testlike import LB_OFFSET, testlike_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARAM_CAP = 300_000_000
V007 = ROOT / "submissions/v007_final-hingfull+tfidf.csv"


def fold_of(run, n):
    f = np.full(n, -1)
    for k, d in enumerate(sorted(p for p in Path(run).glob("fold*") if (p / "metrics.json").exists())):
        f[np.load(d / "val_idx.npy")] = k
    return f


def trimmed_size(full_dir, top):
    """Parameters of the encoder after trimming, from the tokenizer alone (no weights needed)."""
    from transformers import AutoConfig, AutoTokenizer
    import trim_vocab as tv
    full_dir = Path(full_dir)
    cfg = json.loads((full_dir.parent / "config.json").read_text())
    tok_dir = full_dir / "tokenizer"
    tok = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else cfg["hf_name"])
    texts, _ = tv.corpus_texts()
    keep, n_used = tv.keep_ids(tok, texts, top)
    enc = AutoConfig.from_pretrained(full_dir / "encoder_config" if (full_dir / "encoder_config").exists() else cfg["hf_name"])
    n_full = json.loads((full_dir / "metrics.json").read_text())["n_params"]
    return int(n_full - (enc.vocab_size - len(keep)) * enc.hidden_size), len(keep), n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hing", default="runs/hing_lr5")
    ap.add_argument("--muril", default="runs/muril_lr6")
    ap.add_argument("--tfidf", default="runs/tfidf_aug")
    ap.add_argument("--hing_full", default="runs/hing_full/full")
    ap.add_argument("--muril_full", default="runs/muril_full/full")
    ap.add_argument("--tfidf_full", default="runs/tfidf_aug/full")
    ap.add_argument("--top", type=int, default=60000, help="most frequent vocabulary pieces kept on top of the used ones")
    ap.add_argument("--trim", action="store_true", help="also write the trimmed encoders (needs model.pt)")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    R = lambda p: ROOT / p if not Path(p).is_absolute() else Path(p)  # noqa: E731

    tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
    y = tr.target.values; pp.Y = y; n = len(y)
    tl = testlike_mask(n)
    have_muril = (R(a.muril) / "fold0").exists() and (R(a.muril_full) / "test_logits.npy").exists()
    comp = {"H": (a.hing, a.hing_full), "T": (a.tfidf, a.tfidf_full)}
    if have_muril:
        comp["M"] = (a.muril, a.muril_full)
    oof, test, nparams = {}, {}, {}
    for k, (run, full) in comp.items():
        o, _, cov = pp.load_run(R(run), n)
        assert cov.all(), f"{run}: need 5 finished folds"
        oof[k] = o
        test[k] = pp.softmax(np.load(R(full) / "test_logits.npy"))
        nparams[k] = json.loads((R(full) / "metrics.json").read_text())["n_params"]
    folds = fold_of(R(a.hing), n)
    print("\nloading neighbour vote (in-fold index for OOF, training rows for test)...", flush=True)
    kp_oof, best_oof = nb.oof_knn(tr.query_text.values, y)
    kp_te, best_te = nb.test_knn(tr.query_text.values, y, te.query_text.values)
    v007 = pd.read_csv(V007).target.values

    def run(name, w, knn=False):
        p_oof = sum(wi * oof[k] for k, wi in w.items()) / sum(w.values())
        p_te = sum(wi * test[k] for k, wi in w.items()) / sum(w.values())
        bias, tau = pp.tune(p_oof, y)  # rule tuned without the vote, vote mixed in before the rule (as assemble.py)
        if knn:
            p_oof, p_te = nb.mix(p_oof, kp_oof, best_oof), nb.mix(p_te, kp_te, best_te)
        pred = pp.apply_rule(p_oof, bias, tau)
        tpred = pp.apply_rule(p_te, bias, tau)
        r = {"name": name, "weights": w, "knn": knn, "bias": bias, "tau": tau, "pred": pred, "tpred": tpred,
             "tuned": f1_score(y, pred, average="macro"), "testlike": f1_score(y[tl], pred[tl], average="macro"),
             "folds": [f1_score(y[tl & (folds == k)], pred[tl & (folds == k)], average="macro") for k in range(5)],
             "test_changed_vs_v007": int(np.sum(tpred != v007))}
        return r

    res = [run("A_v007", {"H": 0.5, "T": 0.5})]
    base = res[0]
    print(f"\nsanity: v007 recomputed OOF {base['tuned']:.4f} test-like {base['testlike']:.4f} (registry: 0.9169 / 0.9191); "
          f"test answers differing from the submitted v007 file: {base['test_changed_vs_v007']}")
    res.append(run("B_v007+knn", {"H": 0.5, "T": 0.5}, knn=True))
    if have_muril:
        for nm, w in [("C1_3way_25-25-50", {"H": 0.25, "M": 0.25, "T": 0.5}),
                      ("C2_3way_30-20-50", {"H": 0.30, "M": 0.20, "T": 0.5}),
                      ("C3_3way_35-15-50", {"H": 0.35, "M": 0.15, "T": 0.5})]:
            res.append(run(nm, w))
        best_c = max((r for r in res if r["name"].startswith("C")), key=lambda r: (r["testlike"], -r["weights"]["M"]))
        res.append(run("D_" + best_c["name"][3:] + "+knn", best_c["weights"], knn=True))
    else:
        print("MuRIL runs not found: skipping the three-model options (C, D)")

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(tl)
    samples = [rng.choice(idx, len(idx)) for _ in range(a.boot)]
    f_base = [f1_score(y[s], base["pred"][s], average="macro") for s in samples]
    print(f"\n{'option':24s} {'OOF':>7s} {'test-like':>9s} {'est.public':>10s} {'folds won':>9s} {'P(better)':>9s} {'test changed':>12s}")
    for r in res:
        wins = sum(c > b for c, b in zip(r["folds"], base["folds"]))
        r["wins"] = wins
        r["p_better"] = float(np.mean([f1_score(y[s], r["pred"][s], average="macro") > fb for s, fb in zip(samples, f_base)])) \
            if r is not base else 0.5
        print(f"{r['name']:24s} {r['tuned']:7.4f} {r['testlike']:9.4f} {r['testlike'] - LB_OFFSET:10.3f} "
              f"{wins if r is not base else '-':>9} {r['p_better']:9.2f} {r['test_changed_vs_v007']:12d}")

    # parameter budget of each option (three-model options need both encoders trimmed)
    print("\nparameter budget:")
    sizes = {"T": nparams["T"], "H": nparams["H"]}
    if have_muril:
        for k, full in (("H", a.hing_full), ("M", a.muril_full)):
            s, kept, used = trimmed_size(R(full), a.top)
            sizes[k + "_trim"] = s
            print(f"  {k}: {nparams[k]:,} -> trimmed {s:,} (keeps {kept:,} tokens; competition text uses {used:,})")
        three = sizes["H_trim"] + sizes["M_trim"] + sizes["T"]
        print(f"  three-model total {three:,} ({'OK' if three <= PARAM_CAP else 'OVER'} the 300M cap)")
        assert three <= PARAM_CAP

    out_dir = ROOT / "submissions"
    for r in res[1:]:
        if not (r["tuned"] > base["tuned"] and r["testlike"] > base["testlike"]):
            continue
        three_model = "M" in r["weights"]
        order = ["H", "M", "T"] if three_model else ["H", "T"]
        trim_dir = lambda f: str(Path(f).parent / "trim")  # noqa: E731
        full = {"H": trim_dir(a.hing_full) if three_model else a.hing_full, "M": trim_dir(a.muril_full), "T": a.tfidf_full}
        npar = {k: sizes[k + "_trim"] if three_model and k != "T" else nparams[k] for k in order}
        stem = out_dir / f"cand_p1_{r['name']}"
        pd.DataFrame({"example_id": te.example_id, "target": r["tpred"]}).to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".json").write_text(json.dumps({
            "phase": 1, "option": r["name"], "oof_runs": [comp[k][0] for k in order], "full_models": [full[k] for k in order],
            "weights": [r["weights"][k] for k in order], "tau": r["tau"], "bias": np.round(r["bias"], 3).tolist(),
            "knn": {"k": nb.K, "gate": nb.GATE, "share": nb.SHARE, "temp": nb.TEMP} if r["knn"] else None,
            "oof_tuned": r["tuned"], "testlike": r["testlike"], "est_lb": r["testlike"] - LB_OFFSET,
            "testlike_folds_won_vs_v007": r["wins"], "bootstrap_p_better_than_v007": r["p_better"],
            "test_answers_changed_vs_v007": r["test_changed_vs_v007"],
            "n_params": {full[k]: int(npar[k]) for k in order}, "total_params": int(sum(npar.values()))}, indent=1))
        verdict = "PASSES every fold" if r["wins"] == 5 else f"better overall, wins {r['wins']}/5 folds"
        print(f"wrote {stem.name}.csv/.json ({verdict})")

    if a.trim and have_muril:
        import trim_vocab as tv
        for full in (a.hing_full, a.muril_full):
            if (R(full) / "model.pt").exists():
                tv.trim(R(full), R(full).parent / "trim", a.top)
            else:
                print(f"no model.pt in {full}: trimming skipped")


if __name__ == "__main__":
    main()
