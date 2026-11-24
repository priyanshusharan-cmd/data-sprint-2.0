"""
Domain-adaptive pretraining (DAPT): continue HingRoBERTa's masked-language-model training on the competition's own
text (train + test queries, no labels) plus rule-based augmented views of train (src/augment.py). The fine-tuning
in kaggle/train.py can then start from this checkpoint with --init <out>. Same architecture and parameter count.

    python kaggle/mlm_pretrain.py --out /kaggle/working/dapt_hing                 # Kaggle T4: ~15-20 min
    python kaggle/mlm_pretrain.py --limit 200 --epochs 1 --out runs/dapt_smoke    # CPU smoke test
"""
import argparse
import os
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))
from augment import augment  # noqa: E402
from train import MODELS, find_data, seed_all  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hingroberta")
    p.add_argument("--data", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--aug_views", type=int, default=2, help="augmented copies of each train row per epoch")
    p.add_argument("--mlm_p", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    seed_all(a.seed)
    name = MODELS.get(a.model, a.model)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = find_data(a.data)
    tr = pd.read_csv(data / "train.csv").query_text.tolist()
    te = pd.read_csv(data / "test.csv").query_text.tolist()
    if a.limit:
        tr, te = tr[: a.limit], te[: a.limit]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name).to(dev)
    coll = DataCollatorForLanguageModeling(tok, mlm_probability=a.mlm_p)
    base, head = getattr(model, model.base_model_prefix), (model.lm_head if hasattr(model, "lm_head") else model.cls)
    use_fp16 = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    n_per_epoch = len(tr) * (1 + a.aug_views) + len(te)
    steps = a.epochs * math.ceil(n_per_epoch / a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    print(f"{name} | corpus/epoch {n_per_epoch:,} sentences (train {len(tr):,} x{1 + a.aug_views}, test {len(te):,}) | {steps:,} steps", flush=True)
    for ep in range(a.epochs):
        rng = random.Random(a.seed + ep)
        texts = tr + te + [augment(t, rng) for t in tr for _ in range(a.aug_views)]
        rng.shuffle(texts)
        model.train(); t0 = time.time(); tot = n = 0
        for i in range(0, len(texts), a.bs):
            enc = tok(texts[i:i + a.bs], truncation=True, max_length=a.max_len)
            batch = coll([{"input_ids": x} for x in enc["input_ids"]])
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
                # score the vocabulary only at masked positions (~15%): full 250K-word logits for every token run out of GPU memory
                h = base(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                m = batch["labels"] != -100
                loss = F.cross_entropy(head(h[m]).float(), batch["labels"][m])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep}: MLM loss {tot / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
