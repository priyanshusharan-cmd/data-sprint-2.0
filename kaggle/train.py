"""
Fine-tune a multilingual encoder for 56-way Hindi/Hinglish intent classification (Kaggle GPU).

Resumable: every finished fold writes its outputs to <out>/<run>/fold{k}/ and is skipped on re-run.

    python kaggle/train.py --model muril --folds 0 1 2 3 4
    python kaggle/train.py --model hingroberta --folds 0 1 2 3 4
    python kaggle/train.py --model muril --full --epochs 5          # all data, saves weights for demo

Opt-in upgrades (all off by default, so earlier runs reproduce exactly):
    --importance 0.5            up-weight test-like train rows (data/testlike.csv p_test rank), mean weight 1
    --sample_weights data/sample_weights.csv   per-row loss weights (src/noise_weights.py: suspected label noise)
    --logit_adj 1.0 --class_weight 0           logit-adjusted loss (Menon et al. 2021) instead of class weights
    --ema 0.999                 evaluate/save an exponential moving average of the weights
    --fgm 1.0                   FGM adversarial perturbation of the word embeddings (noise robustness)
    --teacher t.npz --kd_alpha 0.5 --kd_T 2 [--kd_test 1]  distil soft labels (npz: train, test prob arrays);
                                with --kd_test 1 the test queries join training as soft-label-only rows
    --init_from runs/mlm_muril  start from a domain-adapted encoder (kaggle/mlm_pretrain.py)

Outputs per fold:  val_idx.npy  val_logits.npy  test_logits.npy  test_emb.npy  metrics.json
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from augment import augment  # noqa: E402

MODELS = {
    "muril": "google/muril-base-cased",
    "hingroberta": "l3cube-pune/hing-roberta-mixed",
    "indicbert": "ai4bharat/IndicBERTv2-MLM-Sam-TLM",
    "xlmr": "xlm-roberta-base",
    "e5small": "intfloat/multilingual-e5-small",
}
N_CLASSES = 56
PARAM_CAP = 300_000_000


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="muril", help=f"alias {list(MODELS)} or any HF id")
    p.add_argument("--data", default=None, help="dir with train.csv/test.csv (auto-detected on Kaggle)")
    p.add_argument("--out", default=None, help="output root (default /kaggle/working/runs or ./runs)")
    p.add_argument("--run", default=None, help="run name (default = model alias)")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--full", action="store_true", help="train on all rows, save weights")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--llrd", type=float, default=0.9, help="layer-wise lr decay")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--aug_p", type=float, default=0.6, help="prob a training row is replaced by an augmented view")
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--pseudo", default=None, help="csv(query_text,target,weight) of pseudo-labelled test rows")
    p.add_argument("--fp16", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="smoke test: use first N train rows")
    p.add_argument("--importance", type=float, default=0.0)
    p.add_argument("--sample_weights", default=None)
    p.add_argument("--logit_adj", type=float, default=0.0)
    p.add_argument("--class_weight", type=int, default=1)
    p.add_argument("--ema", type=float, default=0.0)
    p.add_argument("--fgm", type=float, default=0.0)
    p.add_argument("--teacher", default=None)
    p.add_argument("--kd_alpha", type=float, default=0.5)
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--kd_test", type=int, default=0)
    p.add_argument("--init_from", default=None)
    return p.parse_args()


def find_data(arg):
    for d in [arg, "/kaggle/input/anveshan-data-sprint-2-nlp", ROOT / "data"]:
        if d and (Path(d) / "train.csv").exists():
            return Path(d)
    hits = list(Path("/kaggle/input").glob("**/train.csv")) if Path("/kaggle/input").exists() else []
    if hits:
        return hits[0].parent
    raise FileNotFoundError("train.csv not found; pass --data")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Net(nn.Module):
    def __init__(self, name, n_msd=5):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)  # an --init_from dir works too (MLM head weights are dropped)
        self.drops = nn.ModuleList(nn.Dropout(0.1 * (i + 1) / n_msd + 0.05) for i in range(n_msd))
        self.head = nn.Linear(self.enc.config.hidden_size, N_CLASSES)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        emb = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.training:  # multi-sample dropout
            logits = torch.stack([self.head(d(emb)) for d in self.drops]).mean(0)
        else:
            logits = self.head(emb)
        return logits, emb


class DS(torch.utils.data.Dataset):
    """label -100 = soft-label-only row (distillation on test queries)."""
    def __init__(self, texts, labels, weights, aug_p=0.0, seed=0, soft=None):
        self.t, self.y, self.w, self.aug_p, self.seed, self.epoch = list(texts), list(labels), list(weights), aug_p, seed, 0
        self.soft = soft  # (n, 56) float32 teacher probabilities or None

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        t = self.t[i]
        if self.aug_p:
            r = random.Random(hash((self.seed, self.epoch, i)))
            if r.random() < self.aug_p:
                t = augment(t, r)
        s = self.soft[i] if self.soft is not None else np.zeros(N_CLASSES, np.float32)
        return t, self.y[i], self.w[i], s


def loader(ds, tok, bs, max_len, shuffle):
    def collate(b):
        t, y, w, s = zip(*b)
        enc = tok(list(t), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return (enc["input_ids"], enc["attention_mask"], torch.tensor(y), torch.tensor(w, dtype=torch.float),
                torch.tensor(np.stack(s), dtype=torch.float))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                                       num_workers=2 if torch.cuda.is_available() else 0)  # macOS spawn can't pickle collate


def param_groups(model, a):
    enc = model.enc
    layers = getattr(getattr(enc, "encoder", None), "layer", None)
    if layers is None:
        return [{"params": enc.parameters(), "lr": a.lr}, {"params": model.head.parameters(), "lr": a.head_lr}]
    n = len(layers)
    groups = [{"params": model.head.parameters(), "lr": a.head_lr}]
    for i, layer in enumerate(layers):
        groups.append({"params": layer.parameters(), "lr": a.lr * a.llrd ** (n - 1 - i)})
    layer_ids = {id(p) for layer in layers for p in layer.parameters()}
    rest = [p for p in enc.parameters() if id(p) not in layer_ids]  # embeddings, pooler
    groups.append({"params": rest, "lr": a.lr * a.llrd ** n})
    for g in groups:
        g["params"] = list(g["params"])
    return groups


@torch.no_grad()
def predict(model, dl, dev, fp16):
    model.eval()
    L, E = [], []
    for ids, mask, *_ in dl:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=fp16):
            lo, em = model(ids.to(dev), mask.to(dev))
        L.append(lo.float().cpu()); E.append(em.float().cpu())
    return torch.cat(L).numpy(), torch.cat(E).numpy().astype(np.float16)


def swap_ema(model, ema):
    """Exchange live weights and EMA weights in place (call twice to restore)."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in ema:
                tmp = p.data.clone(); p.data.copy_(ema[n]); ema[n].copy_(tmp)


def train_one(a, name, tok, tr_t, tr_y, tr_w, val_sets, test_t, out_dir, dev, soft=None):
    """Train; pick best epoch on mean macro-F1 over val_sets (clean + shifted). Returns metrics."""
    seed_all(a.seed)
    model = Net(a.init_from or name).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= PARAM_CAP, f"{n_params} params > cap"
    hard = np.array([v for v in tr_y if v >= 0])
    counts = np.bincount(hard, minlength=N_CLASSES).astype(float)
    cw = torch.tensor((counts.clip(1) ** -0.5) / np.mean(counts.clip(1) ** -0.5), dtype=torch.float, device=dev)
    log_prior = torch.tensor(np.log(counts.clip(1) / counts.sum()), dtype=torch.float, device=dev)
    ds = DS(tr_t, tr_y, tr_w, a.aug_p, a.seed, soft)
    dl = loader(ds, tok, a.bs, a.max_len, True)
    opt = torch.optim.AdamW(param_groups(model, a), weight_decay=0.01)
    steps = a.epochs * len(dl)
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_fp16 = bool(a.fp16) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    ema = {n: p.detach().clone() for n, p in model.named_parameters()} if a.ema else None
    emb_w = model.enc.get_input_embeddings().weight
    step = 0

    def loss_fn(logits, y, w, s):
        z = logits.float()
        has_hard = y >= 0
        ce = F.cross_entropy(z + a.logit_adj * log_prior if a.logit_adj else z, y,
                             weight=cw if a.class_weight else None, label_smoothing=a.label_smoothing,
                             ignore_index=-100, reduction="none")
        if soft is None:
            return (ce * w).mean()
        st = s.clamp_min(1e-8) ** (1.0 / a.kd_T)
        st = st / st.sum(1, keepdim=True)
        kd = (st * (st.log() - F.log_softmax(z / a.kd_T, 1))).sum(1) * a.kd_T ** 2
        row = torch.where(has_hard, (1 - a.kd_alpha) * ce + a.kd_alpha * kd, kd)
        return (row * w).mean()

    best, best_state, hist = -1.0, None, []
    for ep in range(a.epochs):
        ds.epoch = ep
        model.train(); t0 = time.time(); tot = 0.0
        for ids, mask, y, w, s in dl:
            ids, mask, y, w, s = ids.to(dev), mask.to(dev), y.to(dev), w.to(dev), s.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
                logits, _ = model(ids, mask)
            loss = loss_fn(logits, y, w, s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if a.fgm and emb_w.grad is not None:  # FGM: one adversarial step on the embedding matrix
                g = emb_w.grad.float()
                norm = g.norm()
                if torch.isfinite(norm) and norm > 0:
                    r = (a.fgm * g / norm).to(emb_w.dtype)
                    emb_w.data.add_(r)
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
                        adv_logits, _ = model(ids, mask)
                    scaler.scale(loss_fn(adv_logits, y, w, s)).backward()
                    emb_w.data.sub_(r)
                    del r
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            step += 1
            if ema is not None:
                d = min(a.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        ema[n].mul_(d).add_(p.detach(), alpha=1 - d)
            tot += loss.item()
        rec = {"epoch": ep, "loss": round(tot / len(dl), 4), "sec": round(time.time() - t0)}
        if ema is not None:
            swap_ema(model, ema)  # evaluate (and snapshot) the EMA weights
        if val_sets:
            for vname, (vt, vy) in val_sets.items():
                lo, _ = predict(model, loader(DS(vt, vy, [1.0] * len(vy)), tok, 128, a.max_len, False), dev, use_fp16)
                rec[vname] = round(f1_score(vy, lo.argmax(1), average="macro"), 4)
            score = np.mean([rec[v] for v in val_sets])
        else:
            score = ep  # full mode: keep last epoch
        hist.append(rec); print(json.dumps(rec), flush=True)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ema is not None:
            swap_ema(model, ema)  # back to live weights for the next epoch
    model.load_state_dict(best_state)
    res = {"best_score": float(best), "best_epoch": int(max(hist, key=lambda r: np.mean([r[v] for v in val_sets]) if val_sets else r["epoch"])["epoch"]),
           "history": hist, "n_params": int(n_params)}
    tdl = loader(DS(test_t, [0] * len(test_t), [1.0] * len(test_t)), tok, 128, a.max_len, False)
    tl, te = predict(model, tdl, dev, use_fp16)
    np.save(out_dir / "test_logits.npy", tl); np.save(out_dir / "test_emb.npy", te)
    if val_sets:
        vt, vy = val_sets["val_clean"]
        vl, ve = predict(model, loader(DS(vt, vy, [1.0] * len(vy)), tok, 128, a.max_len, False), dev, use_fp16)
        np.save(out_dir / "val_logits.npy", vl); np.save(out_dir / "val_emb.npy", ve)
    return model, res


def main():
    a = parse()
    name = MODELS.get(a.model, a.model)
    run = a.run or a.model
    out_root = Path(a.out or ("/kaggle/working/runs" if Path("/kaggle/working").exists() else ROOT / "runs")) / run
    out_root.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = find_data(a.data)
    tr = pd.read_csv(data / "train.csv"); te = pd.read_csv(data / "test.csv")
    if a.limit:
        tr = tr.sample(a.limit, random_state=a.seed).reset_index(drop=True); te = te.head(a.limit)
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = os.environ.get("GIT_COMMIT", "unknown")  # code shipped as an archive (Kaggle CLI runner)
    cfg = {**vars(a), "hf_name": name, "commit": commit, "device": str(dev), "data": str(data)}
    (out_root / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg), flush=True)
    tok = AutoTokenizer.from_pretrained(name)

    w_tr = np.ones(len(tr))
    if a.sample_weights:
        sw = pd.read_csv(a.sample_weights)
        assert len(sw) == len(tr) or a.limit, "sample_weights rows != train rows"
        if not a.limit:
            w_tr *= sw.weight.values
        print(f"sample weights: {np.sum(w_tr < 1)} down-weighted rows", flush=True)
    if a.importance:
        tl = pd.read_csv(ROOT / "data/testlike.csv")
        if not a.limit:
            rank = tl.p_test.rank(pct=True).values
            w_tr *= (1 - a.importance) + a.importance * 2 * rank
        print(f"importance weighting alpha={a.importance}: weight range {w_tr.min():.2f}-{w_tr.max():.2f}", flush=True)
    soft_tr = soft_te = None
    if a.teacher:
        z = np.load(a.teacher)
        soft_tr, soft_te = z["train"].astype(np.float32), z["test"].astype(np.float32)
        if a.limit:
            soft_tr, soft_te = soft_tr[: len(tr)], soft_te[: len(te)]
        assert soft_tr.shape == (len(tr), N_CLASSES) and soft_te.shape == (len(te), N_CLASSES)
        print(f"teacher: {a.teacher} kd_alpha={a.kd_alpha} T={a.kd_T} test rows as soft-only: {bool(a.kd_test)}", flush=True)

    extra_t, extra_y, extra_w, extra_s = [], [], [], []
    if a.pseudo:
        ps = pd.read_csv(a.pseudo)
        extra_t, extra_y, extra_w = ps.query_text.tolist(), ps.target.astype(int).tolist(), ps.weight.astype(float).tolist()
        extra_s = [np.eye(N_CLASSES, dtype=np.float32)[v] for v in extra_y]
        print(f"pseudo rows: {len(ps)}", flush=True)
    if a.teacher and a.kd_test:
        extra_t += te.query_text.tolist(); extra_y += [-100] * len(te); extra_w += [1.0] * len(te)
        extra_s += list(soft_te)

    def soft_for(idx):
        return None if soft_tr is None else np.concatenate([soft_tr[idx], np.array(extra_s, np.float32).reshape(-1, N_CLASSES)])

    y = tr.target.values
    if a.full:
        fdir = out_root / "full"; fdir.mkdir(exist_ok=True)
        if (fdir / "metrics.json").exists():
            print("full run already done; delete", fdir, "to redo"); return
        model, res = train_one(a, name, tok, tr.query_text.tolist() + extra_t, list(y) + extra_y,
                               list(w_tr) + extra_w, None, te.query_text.tolist(), fdir, dev, soft_for(np.arange(len(tr))))
        torch.save(model.state_dict(), fdir / "model.pt")
        tok.save_pretrained(fdir / "tokenizer")
        (fdir / "metrics.json").write_text(json.dumps(res, indent=2))
        return

    skf = list(StratifiedKFold(a.n_folds, shuffle=True, random_state=42).split(tr, y))
    for k in a.folds:
        fdir = out_root / f"fold{k}"
        if (fdir / "metrics.json").exists():
            print(f"fold {k} done, skipping", flush=True); continue
        fdir.mkdir(exist_ok=True)
        tri, vai = skf[k]
        vt = tr.query_text.iloc[vai].tolist(); vy = y[vai].tolist()
        rs = random.Random(1000 + k)
        val_sets = {"val_clean": (vt, vy), "val_shift": ([augment(t, rs) for t in vt], vy)}
        print(f"=== fold {k}: train {len(tri)} val {len(vai)}", flush=True)
        _, res = train_one(a, name, tok, tr.query_text.iloc[tri].tolist() + extra_t, list(y[tri]) + extra_y,
                           list(w_tr[tri]) + extra_w, val_sets, te.query_text.tolist(), fdir, dev, soft_for(tri))
        np.save(fdir / "val_idx.npy", vai)
        (fdir / "metrics.json").write_text(json.dumps(res, indent=2))  # written last = fold complete
        print(f"fold {k} best {res['best_score']:.4f} @ epoch {res['best_epoch']}", flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
