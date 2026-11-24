"""
Domain-adaptive masked-LM pretraining (Gururangan et al. 2020, "Don't Stop Pretraining") on the competition text:
train + unlabelled test queries + augmented views. No labels, no external data. Output is a drop-in encoder for
kaggle/train.py --init_from <out>.

    python kaggle/mlm_pretrain.py --model muril --epochs 2 --out /kaggle/working/runs/mlm_muril
    python kaggle/train.py --model muril --init_from /kaggle/working/runs/mlm_muril --full ...
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))
from augment import augment  # noqa: E402
from train import MODELS, find_data  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="muril")
    p.add_argument("--data", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--views", type=int, default=2, help="augmented copies of each train/test query")
    p.add_argument("--mlm_prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    name = MODELS.get(a.model, a.model)
    default_root = Path("/kaggle/working/runs") if Path("/kaggle/working").exists() else ROOT / "runs"
    out = Path(a.out) if a.out else default_root / f"mlm_{a.model}"
    if (out / "done.json").exists():
        print("MLM already done:", out); return
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)
    data = find_data(a.data)
    texts = pd.concat([pd.read_csv(data / "train.csv").query_text, pd.read_csv(data / "test.csv").query_text]).tolist()
    if a.limit:
        texts = texts[: a.limit]
    rng = random.Random(a.seed)
    corpus = texts + [augment(t, rng) for t in texts for _ in range(a.views)]
    print(f"MLM corpus: {len(corpus)} lines ({len(texts)} real + {len(corpus) - len(texts)} augmented)", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForMaskedLM.from_pretrained(name).to(dev)
    coll = DataCollatorForLanguageModeling(tok, mlm_probability=a.mlm_prob)
    enc = [tok(t, truncation=True, max_length=a.max_len)["input_ids"] for t in corpus]
    dl = torch.utils.data.DataLoader(enc, batch_size=a.bs, shuffle=True, collate_fn=lambda b: coll([{"input_ids": x} for x in b]))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    fp16 = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=fp16):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
        rec = {"epoch": ep, "mlm_loss": round(tot / len(dl), 4), "sec": round(time.time() - t0)}
        hist.append(rec); print(json.dumps(rec), flush=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    (out / "done.json").write_text(json.dumps({"base": name, "history": hist, "lines": len(corpus)}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
