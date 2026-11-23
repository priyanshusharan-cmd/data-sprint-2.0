"""
Load the full-data brain (model.pt) on this machine's CPU and check it against Kaggle.

    python src/verify_model.py runs/muril_full/full --n 200

1. Rebuilds the exact network from kaggle/train.py (MuRIL encoder + mean-pool + linear head) without
   downloading the pretrained weights again (only the 411-byte config), loads model.pt, prints the param count.
2. Re-predicts the first N test queries locally and compares with Kaggle's saved test_logits.npy
   (Kaggle ran fp16 on GPU, here fp32 on CPU, so logits differ slightly; the predicted classes should match).
3. Times single-query inference (tokenize + forward + argmax), the live-demo setting.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


class Brain(nn.Module):
    """Same parameter layout as kaggle/train.py Net (dropouts have no params), built from config only."""

    def __init__(self, hf_name, n_classes=56):
        super().__init__()
        self.enc = AutoModel.from_config(AutoConfig.from_pretrained(hf_name))
        self.head = nn.Linear(self.enc.config.hidden_size, n_classes)

    @torch.inference_mode()
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1.0))


def load(full_dir):
    full_dir = Path(full_dir)
    cfg = json.loads((full_dir.parent / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(full_dir / "tokenizer")
    model = Brain(cfg["hf_name"])
    state = torch.load(full_dir / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    state = {k: v for k, v in state.items() if not k.startswith("drops.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tok, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_dir")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    t0 = time.perf_counter()
    model, tok, cfg = load(a.full_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded in {time.perf_counter() - t0:.1f}s | params {n_params:,} | torch threads {torch.get_num_threads()}")

    te = pd.read_csv(ROOT / "data/test.csv").head(a.n)
    kaggle = np.load(Path(a.full_dir) / "test_logits.npy")[: a.n]
    local = []
    for i in range(0, len(te), 64):
        enc = tok(te.query_text.iloc[i:i + 64].tolist(), padding=True, truncation=True,
                  max_length=cfg["max_len"], return_tensors="pt")
        local.append(model(enc["input_ids"], enc["attention_mask"]).float().numpy())
    local = np.concatenate(local)
    same = (local.argmax(1) == kaggle.argmax(1)).mean()
    print(f"first {a.n} test queries: predicted class identical to Kaggle for {same:.1%} | "
          f"max |logit diff| {np.abs(local - kaggle).max():.4f} (fp32 CPU vs fp16 GPU)")

    q = te.query_text.tolist()
    for t in q[:5]:  # warm-up
        enc = tok([t], truncation=True, max_length=cfg["max_len"], return_tensors="pt")
        model(enc["input_ids"], enc["attention_mask"])
    ms = []
    for t in q[:100]:
        s = time.perf_counter()
        enc = tok([t], truncation=True, max_length=cfg["max_len"], return_tensors="pt")
        model(enc["input_ids"], enc["attention_mask"]).argmax(1)
        ms.append((time.perf_counter() - s) * 1000)
    ms = np.array(ms)
    print(f"single-query latency (tokenize+forward, fp32 CPU): p50 {np.percentile(ms, 50):.1f} ms | "
          f"p95 {np.percentile(ms, 95):.1f} ms | max {ms.max():.1f} ms")


if __name__ == "__main__":
    sys.exit(main())
