"""
Build kaggle/reproduce_v007.ipynb: a single self-contained Kaggle notebook that rebuilds our final submission (v007 =
HingRoBERTa + TF-IDF/LinearSVC 50/50) from the competition data, without access to our private GitHub repo.

The notebook embeds the exact source files it needs (via %%writefile), the frozen decision rule tuned on 5-fold
out-of-fold predictions, and a compressed copy of our submitted v007 file to measure agreement.

    python tools/build_repro_notebook.py      # regenerate after any change to the embedded files
"""
import base64
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED = ["src/normalize.py", "src/augment.py", "src/baseline.py", "src/postprocess.py", "src/testlike.py", "kaggle/train.py"]
OUT = ROOT / "kaggle/reproduce_v007.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def main():
    rule = json.loads((ROOT / "submissions/v007_final-hingfull+tfidf.json").read_text())
    frozen = {"weights": rule["weights"], "tau": rule["tau"], "bias": rule["bias"],
              "oof_tuned_macro_f1": round(rule["oof_tuned"], 5), "testlike_macro_f1": round(rule["testlike"], 5)}
    v007 = base64.b64encode(gzip.compress((ROOT / "submissions/v007_final-hingfull+tfidf.csv").read_bytes())).decode()

    cells = [md(
        "# Data Sprint 2.0 NLP: reproduce our final submission (v007)\n"
        "**Model:** HingRoBERTa-mixed (fine-tuned, 278,086,712 params) + character TF-IDF / LinearSVC "
        "(9,291,576 params), probabilities averaged 50/50, then a class-55 fallback rule with per-class biases. "
        "**Total 287,378,288 parameters** (limit 300,000,000). Public leaderboard: **0.86786**.\n\n"
        "**Run:** Accelerator = GPU (T4 or P100), Internet = On (to download the pretrained HingRoBERTa from Hugging Face), "
        "competition data attached. *Run All* takes about 25 minutes: TF-IDF 5-fold + full fit on CPU (~5 min), "
        "HingRoBERTa full-data fine-tuning on GPU (~16 min), blending (seconds).\n\n"
        "**What it uses:** only the competition's `train.csv` / `test.csv`, the open-source pretrained checkpoint "
        "`l3cube-pune/hing-roberta-mixed` (CC-BY-4.0), and rule-based augmentation (`augment.py`). No external datasets, "
        "no commercial LLM APIs, no hand labelling. The decision rule below was tuned on 5-fold out-of-fold predictions of "
        "`train.csv` (section 7 re-derives it; optional, ~65 extra GPU minutes).\n\n"
        "Sections: 1 config · 2 data · 3 source files · 4 frozen rule · 5 TF-IDF · 6 HingRoBERTa · 7 optional rule re-tuning · "
        "8 blend + submission · 9 parameter check · 10 agreement with our submitted file."),
        code("# 1) config\n"
             "SMOKE = False            # True = tiny CPU run (300 rows, 1 epoch) just to test the plumbing\n"
             "RECOMPUTE_RULE = False   # True = re-derive weights/tau/bias from 5-fold OOF (adds ~65 GPU minutes)\n"
             "SEED = 42\n"
             "REPO = '/kaggle/working/repo' if __import__('os').path.isdir('/kaggle/working') else 'repo'\n"),
        code("# 2) data: find the competition CSVs and place them where the scripts expect them\n"
             "import os, glob, shutil, subprocess, json\n"
             "import pandas as pd\n"
             "os.makedirs(f'{REPO}/data', exist_ok=True); os.makedirs(f'{REPO}/src', exist_ok=True)\n"
             "os.makedirs(f'{REPO}/kaggle', exist_ok=True); os.makedirs(f'{REPO}/submissions', exist_ok=True)\n"
             "def find(name):\n"
             "    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True)) or sorted(glob.glob(f'**/data/{name}', recursive=True))\n"
             "    assert hits, f'{name} not found: attach the competition data'\n"
             "    return hits[0]\n"
             "for n in ['train.csv', 'test.csv']:\n"
             "    if not os.path.exists(f'{REPO}/data/{n}'):\n"
             "        shutil.copy(find(n), f'{REPO}/data/{n}')\n"
             "tr, te = pd.read_csv(f'{REPO}/data/train.csv'), pd.read_csv(f'{REPO}/data/test.csv')\n"
             "print('train', tr.shape, '| test', te.shape, '| classes', tr.target.nunique())\n"
             "import torch; print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"),
        md("## 3) Source files\nExact copies of the files in our repository (`src/`, `kaggle/train.py`). "
           "`normalize.py`: Devanagari→Latin transliteration and Hinglish phonetic key. `augment.py`: rule-based "
           "test-style augmentation (romanisation, fillers, digits, ASR noise). `baseline.py`: TF-IDF + LinearSVC. "
           "`train.py`: transformer fine-tuning. `postprocess.py` / `testlike.py`: OOF blending, rule tuning and the "
           "test-like validation subset (used only in optional section 7)."),
    ]
    for f in EMBED:
        cells.append(code(f"%%writefile {{REPO}}/{f}\n" + (ROOT / f).read_text(encoding="utf-8")))
    cells += [
        md("## 4) Frozen decision rule (tuned on 5-fold out-of-fold predictions of train.csv)\n"
           f"Weights {frozen['weights']}, class-55 threshold tau = {frozen['tau']}, 56 per-class log-probability biases. "
           f"On the 16,936 OOF rows this rule gives macro-F1 {frozen['oof_tuned_macro_f1']} "
           f"({frozen['testlike_macro_f1']} on the 4,234 most test-like rows)."),
        code("RULE = " + json.dumps(frozen) + "\n"
             "json.dump(RULE, open(f'{REPO}/rule_v007.json', 'w'))\nprint('rule saved')"),
        md("## 5) TF-IDF + LinearSVC\n5-fold CV (prints macro-F1 ≈ 0.886), fits the softmax temperature on OOF "
           "log-loss, then fits one model on all augmented training rows and saves its test scores."),
        code("!cd {REPO} && python src/baseline.py --aug --save_run runs/tfidf_aug"),
        md("## 6) HingRoBERTa-mixed, fine-tuned on all training rows\n"
           "Mean-pooled encoder + linear head, multi-sample dropout, sqrt-inverse class weights, label smoothing 0.05, "
           "layer-wise LR decay 0.95, cosine schedule, fp16, 10 epochs, lr 5e-5, 60% of rows replaced by a rule-based "
           "augmented view each epoch. Saves `model.pt` and its test scores."),
        code("extra = '--limit 300 --epochs 1' if SMOKE else '--epochs 10'\n"
             "!cd {REPO} && python kaggle/train.py --model hingroberta --full --lr 5e-5 --llrd 0.95 {extra} "
             "--seed {SEED} --run hing_full --out runs"),
        md("## 7) Optional: re-derive the decision rule from 5-fold OOF (`RECOMPUTE_RULE = True`)\n"
           "Trains HingRoBERTa on each of 5 folds (~13 min each), builds the test-like subset, and tunes tau + "
           "per-class biases on the OOF blend with `postprocess.py`. The result should match section 4."),
        code("if RECOMPUTE_RULE:\n"
             "    !cd {REPO} && python src/testlike.py\n"
             "    !cd {REPO} && python kaggle/train.py --model hingroberta --lr 5e-5 --llrd 0.95 --epochs 10 --run hing_lr5 --out runs\n"
             "    !cd {REPO} && python src/postprocess.py runs/hing_lr5 runs/tfidf_aug --weights 0.5 0.5 --out submissions/oof_rule.csv\n"
             "    RULE = {**RULE, **{k: v for k, v in json.load(open(f'{REPO}/submissions/oof_rule.json')).items() if k in ('tau', 'bias', 'weights')}}\n"
             "    print('re-derived tau', RULE['tau'])\n"
             "else:\n"
             "    print('using the frozen rule from section 4')"),
        md("## 8) Blend and write `submission.csv`"),
        code("import numpy as np\n"
             "def softmax(z):\n"
             "    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)\n"
             "h = np.load(f'{REPO}/runs/hing_full/full/test_logits.npy')\n"
             "t = np.load(f'{REPO}/runs/tfidf_aug/full/test_logits.npy')\n"
             "w = np.array(RULE['weights']) / sum(RULE['weights'])\n"
             "p = w[0] * softmax(h) + w[1] * softmax(t)\n"
             "pred = (np.log(p + 1e-9) + np.array(RULE['bias'])).argmax(1)\n"
             "pred[p[:, :55].max(1) < RULE['tau']] = 55   # no supported intent confident enough -> out of scope\n"
             "sub = pd.DataFrame({'example_id': te.example_id, 'target': pred})\n"
             "sub.to_csv('/kaggle/working/submission.csv' if os.path.isdir('/kaggle/working') else 'submission.csv', index=False)\n"
             "print(sub.shape, '| class-55 share', round((pred == 55).mean(), 3)); sub.head()"),
        md("## 9) Parameter check (organisers' assertion)"),
        code("import sys; sys.path.insert(0, f'{REPO}/kaggle'); sys.path.insert(0, f'{REPO}/src')\n"
             "import importlib, train as T\n"
             "cfg = json.load(open(f'{REPO}/runs/hing_full/config.json'))\n"
             "net = T.Net(cfg['hf_name']); net.load_state_dict(torch.load(f'{REPO}/runs/hing_full/full/model.pt', map_location='cpu'))\n"
             "models = [net]\n"
             "svm_params = json.load(open(f'{REPO}/runs/tfidf_aug/full/metrics.json'))['n_params']\n"
             "torch_params = sum(p.numel() for m in models for p in m.parameters())\n"
             "total = torch_params + svm_params\n"
             "print(f'HingRoBERTa {torch_params:,} + TF-IDF/LinearSVC {svm_params:,} = {total:,}')\n"
             "assert total <= 300_000_000"),
        md("## 10) Agreement with the file we submitted (v007, public 0.86786)\n"
           "GPU training is not bit-exact across machines, so expect a high but not perfect match."),
        code("import gzip, base64, io\n"
             f"V007 = '{v007}'\n"
             "ref = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode(V007))))\n"
             "assert (ref.example_id.values == sub.example_id.values).all()\n"
             "print(f'identical predictions: {np.mean(ref.target.values == sub.target.values):.2%} of {len(sub)} test rows')"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), f"({len(cells)} cells, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
