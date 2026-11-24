"""
Build kaggle/reproduce_v008.ipynb: a self-contained Kaggle notebook that rebuilds a two-encoder candidate from the
competition data alone: v007's HingRoBERTa (seed 42) + a second encoder, both vocabulary-trimmed, + TF-IDF/LinearSVC,
optionally with the nearest-neighbour vote (if the rule json has "knn"). Same structure as
tools/build_repro_notebook.py (v007), which it does not change.

    --second dapt    NEW-A: Arsh's DAPT HingRoBERTa (masked-LM 6 epochs on train+test text, then v007 fine-tuning, seed 42)
    --second hing2   Phase 2: HingRoBERTa adapted to the competition text by masked-LM training, seed 7
    --second muril   Phase 1: v005's MuRIL (lr 6e-5, llrd 0.95, 10 epochs, seed 42)

    python tools/build_repro_v008.py --second muril --rule submissions/cand_p1_<option>.json --csv submissions/cand_p1_<option>.csv
"""
import argparse
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED = ["src/normalize.py", "src/augment.py", "src/baseline.py", "src/postprocess.py", "src/testlike.py",
         "src/verify_model.py", "src/trim_vocab.py", "kaggle/train.py", "kaggle/mlm_pretrain.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--second", choices=["dapt", "hing2", "muril"], required=True)
    ap.add_argument("--rule", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(ROOT / "kaggle/reproduce_v008.ipynb"))
    a = ap.parse_args()
    rule = json.loads(Path(a.rule).read_text())
    frozen = {"weights": rule["weights"], "tau": rule["tau"], "bias": rule["bias"],
              "oof_tuned_macro_f1": round(rule["oof_tuned"], 5), "testlike_macro_f1": round(rule["testlike"], 5)}
    ref = base64.b64encode(gzip.compress(Path(a.csv).read_bytes())).decode()
    n = rule.get("n_params", {})
    total = rule.get("total_params")
    knn = rule.get("knn")
    two = a.second
    d2 = {"hing2": "hing2_full", "muril": "muril_full", "dapt": "dapt_hing_full"}[two]
    embed = EMBED + (["src/neighbors.py"] if knn else [])
    enc2_doc = {"hing2": "HingRoBERTa first adapted to the competition's own text with masked-language-model training "
                         "(train + test queries, **no labels**), then fine-tuned with seed 7",
                "dapt": "HingRoBERTa first adapted to the competition's own text with masked-language-model training "
                        "(train + test queries + augmented train, **no labels**, 6 epochs), then fine-tuned exactly like "
                        "v007 (seed 42): the encoder of our v009",
                "muril": "MuRIL (`google/muril-base-cased`, Apache-2.0) fine-tuned on all rows, seed 42 (the encoder of our v005)"}[two]
    title = "HingRoBERTa + MuRIL" if two == "muril" else "two-HingRoBERTa"

    cells = [md(
        f"# Data Sprint 2.0 NLP: reproduce the {title} + TF-IDF submission\n"
        "**Model:** two fine-tuned encoders and a character TF-IDF / LinearSVC, probabilities averaged "
        f"with weights {frozen['weights']} (the two encoders share the transformer half), then the class-55 fallback rule "
        "with per-class biases tuned on 5-fold out-of-fold predictions.\n"
        "* encoder 1 = exactly v007's HingRoBERTa (seed 42)\n"
        f"* encoder 2 = {enc2_doc}\n"
        "* both encoders are **vocabulary-trimmed** (`trim_vocab.py`): embedding rows for tokens that neither the "
        "competition text nor the 60,000 most frequent vocabulary pieces use are removed; this changes no prediction "
        "(checked on every test query) and brings each encoder to about 135-145M parameters; section 11 prints the exact "
        f"total and asserts it is at most 300,000,000 (our submitted files were made with {total:,}).\n"
        + ("* then a nearest-neighbour vote over the training rows (`neighbors.py`) is mixed in before the rule\n" if knn else "")
        + "\n"
        "**Run:** Accelerator = GPU (T4 or P100), Internet = On (pretrained checkpoint from Hugging Face), competition "
        "data attached. *Run All* takes about 50 minutes on one T4.\n\n"
        "**What it uses:** only `train.csv` / `test.csv`, the open-source checkpoint `l3cube-pune/hing-roberta-mixed` "
        "(CC-BY-4.0) and rule-based augmentation. No external data, no commercial LLM APIs, no hand labelling."),
        code("# 1) config\n"
             "SMOKE = False   # True = tiny CPU run (300 rows, 1 epoch) just to test the plumbing\n"
             "REPO = '/kaggle/working/repo' if __import__('os').path.isdir('/kaggle/working') else 'repo'\n"),
        code("# 2) data\n"
             "import os, glob, shutil, json\n"
             "import pandas as pd\n"
             "for d in ['data', 'src', 'kaggle', 'submissions']: os.makedirs(f'{REPO}/{d}', exist_ok=True)\n"
             "def find(name):\n"
             "    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True)) or sorted(glob.glob(f'**/data/{name}', recursive=True))\n"
             "    assert hits, f'{name} not found: attach the competition data'\n"
             "    return hits[0]\n"
             "for n in ['train.csv', 'test.csv']:\n"
             "    if not os.path.exists(f'{REPO}/data/{n}'): shutil.copy(find(n), f'{REPO}/data/{n}')\n"
             "tr, te = pd.read_csv(f'{REPO}/data/train.csv'), pd.read_csv(f'{REPO}/data/test.csv')\n"
             "print('train', tr.shape, '| test', te.shape)\n"
             "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"),
        md("## 3) Source files (exact copies from our repository)"),
    ]
    for f in embed:
        cells.append(code(f"%%writefile {{REPO}}/{f}\n" + (ROOT / f).read_text(encoding="utf-8")))
    cells += [
        md("## 4) Frozen decision rule (tuned on 5-fold out-of-fold predictions of train.csv)\n"
           + (rule["note_scores"] if "note_scores" in rule else
              f"OOF macro-F1 {frozen['oof_tuned_macro_f1']} ({frozen['testlike_macro_f1']} on the 4,234 most test-like rows).")),
        code("RULE = " + json.dumps(frozen) + "\nprint('rule loaded')"),
        md("## 5) TF-IDF + LinearSVC (same as v007)"),
        code("!cd {REPO} && python src/baseline.py --aug --save_run runs/tfidf_aug"),
        md("## 6) Encoder 1: HingRoBERTa fine-tuned on all rows, seed 42 (identical to v007)"),
        code("extra = '--limit 300 --epochs 1' if SMOKE else '--epochs 10'\n"
             "!cd {REPO} && python kaggle/train.py --model hingroberta --full --lr 5e-5 --llrd 0.95 {extra} --seed 42 --run hing_full --out runs"),
        *({"hing2": [md("## 7) Domain-adaptive masked-LM training on the competition text (train + test queries, no labels)"),
                     code("mlm = '--limit 300 --epochs 1' if SMOKE else '--epochs 3'\n"
                          "!cd {REPO} && python kaggle/mlm_pretrain.py --model hingroberta {mlm} --out runs/mlm_hing"),
                     md("## 8) Encoder 2: the adapted HingRoBERTa fine-tuned on all rows, seed 7"),
                     code("!cd {REPO} && python kaggle/train.py --model hingroberta --full --lr 5e-5 --llrd 0.95 {extra} --seed 7 "
                          "--init_from runs/mlm_hing --run hing2_full --out runs")],
           "dapt": [md("## 7) Domain-adaptive masked-LM training on the competition text (train + test queries + augmented train, no labels)"),
                    code("mlm = '--limit 300 --epochs 1' if SMOKE else '--epochs 6'\n"
                         "!cd {REPO} && python kaggle/mlm_pretrain.py --model hingroberta {mlm} --out runs/dapt_hing"),
                    md("## 8) Encoder 2: the adapted HingRoBERTa fine-tuned on all rows exactly like v007 (seed 42)"),
                    code("!cd {REPO} && python kaggle/train.py --model hingroberta --full --lr 5e-5 --llrd 0.95 {extra} --seed 42 "
                         "--init runs/dapt_hing --run dapt_hing_full --out runs")],
           "muril": [md("## 7-8) Encoder 2: MuRIL fine-tuned on all rows, seed 42 (v005's settings)"),
                     code("!cd {REPO} && python kaggle/train.py --model muril --full --lr 6e-5 --llrd 0.95 {extra} --seed 42 "
                          "--run muril_full --out runs")]}[two]),
        md("## 9) Lossless vocabulary trimming of both encoders\n"
           "Each call asserts that the trimmed encoder predicts every test query exactly like the untrimmed one."),
        code("!cd {REPO} && python src/trim_vocab.py runs/hing_full/full --top 60000\n"
             f"!cd {{REPO}} && python src/trim_vocab.py runs/{d2}/full --top 60000"),
        md("## 10) Blend and write `submission.csv`"),
        code("import numpy as np\n"
             "def softmax(z):\n"
             "    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)\n"
             f"parts = [np.load(f'{{REPO}}/runs/{{d}}/test_logits.npy') for d in ['hing_full/trim', '{d2}/trim', 'tfidf_aug/full']]\n"
             "w = np.array(RULE['weights']) / sum(RULE['weights'])\n"
             "p = sum(wi * softmax(z) for wi, z in zip(w, parts))\n"
             + ("import sys; sys.path.insert(0, f'{REPO}/src'); import neighbors as nb\n"
                "kp, best = nb.test_knn(tr.query_text.values, tr.target.values, te.query_text.values)\n"
                "p = nb.mix(p, kp, best)   # nearest-neighbour vote, mixed in before the rule (as in assemble.py --knn)\n"
                if knn else "")
             + "pred = (np.log(p + 1e-9) + np.array(RULE['bias'])).argmax(1)\n"
             "pred[p[:, :55].max(1) < RULE['tau']] = 55   # no supported intent confident enough -> out of scope\n"
             "sub = pd.DataFrame({'example_id': te.example_id, 'target': pred})\n"
             "sub.to_csv('/kaggle/working/submission.csv' if os.path.isdir('/kaggle/working') else 'submission.csv', index=False)\n"
             "print(sub.shape, '| class-55 share', round((pred == 55).mean(), 3)); sub.head()"),
        md("## 11) Parameter check (organisers' assertion) on the models that run at inference"),
        code("import sys; sys.path.insert(0, f'{REPO}/src')\n"
             "from verify_model import load\n"
             f"models = [load(f'{{REPO}}/runs/{{d}}/trim')[0] for d in ['hing_full', '{d2}']]\n"
             "svm_params = json.load(open(f'{REPO}/runs/tfidf_aug/full/metrics.json'))['n_params']\n"
             "torch_params = sum(p.numel() for m in models for p in m.parameters())\n"
             "print(f'2 trimmed encoders {torch_params:,} + TF-IDF/LinearSVC {svm_params:,} = {torch_params + svm_params:,}')\n"
             "assert torch_params + svm_params <= 300_000_000"),
        md("## 12) Agreement with the file we produced\nGPU training is not bit-exact across machines: expect a high, not always perfect, match."),
        code("import gzip, base64, io\n"
             f"REF = '{ref}'\n"
             "ref = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(REF))))\n"
             "assert (ref.example_id.values == sub.example_id.values).all()\n"
             "print(f'identical predictions: {np.mean(ref.target.values == sub.target.values):.2%} of {len(sub)} test rows')"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = Path(a.out)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, f"({len(cells)} cells, {out.stat().st_size // 1024} KB)", "| params", n)


if __name__ == "__main__":
    main()
