"""
Lossless vocabulary trimming of a fine-tuned encoder, so that two transformers fit under the 300M parameter cap.

Most of HingRoBERTa's 278M parameters are its 250,002 x 768 word-embedding table, but the competition text uses only
~8,200 of those tokens. Trimming keeps:
  * every token id that appears in train.csv, test.csv and augmented training views (the exact training inputs),
  * every special token,
  * the `--top` most frequent pieces of the original vocabulary (SentencePiece ids are ordered by score), so demo
    queries with new words still tokenize normally,
and drops the rest. The kept rows are copied unchanged, so for any query whose tokens are all kept the trimmed model
computes exactly the same numbers. A dropped token maps to <unk>. Nothing is retrained.

    python src/trim_vocab.py runs/hing_full/full --top 30000          # writes runs/hing_full/trim/

The output dir holds model.pt (trimmed), keep.npy (kept original ids, sorted), encoder_config/, tokenizer/,
test_logits.npy and metrics.json ("n_params" = parameters at inference, read by src/assemble.py), and records how many
test queries the trimmed model predicts identically to the untrimmed one (must be 100%).
At inference `src/verify_model.load()` wraps the tokenizer with RemapTokenizer (original id -> trimmed id).
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]
EMB_KEY = "enc.embeddings.word_embeddings.weight"


class RemapTokenizer:
    """Wraps a Hugging Face tokenizer: tokenizes with the original vocabulary, then maps ids to the trimmed table."""

    def __init__(self, tok, keep, vocab_size):
        self.tok = tok
        keep = np.asarray(keep, dtype=np.int64)
        unk_new = int(np.searchsorted(keep, tok.unk_token_id))
        assert keep[unk_new] == tok.unk_token_id, "unk token must be kept"
        remap = np.full(vocab_size, unk_new, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        self.remap = torch.from_numpy(remap)
        self.remap_np = remap

    def __call__(self, *args, **kw):
        enc = self.tok(*args, **kw)
        ids = enc["input_ids"]
        if isinstance(ids, torch.Tensor):
            enc["input_ids"] = self.remap[ids]
        elif isinstance(ids, np.ndarray):
            enc["input_ids"] = self.remap_np[ids]
        elif ids and isinstance(ids[0], list):
            enc["input_ids"] = [self.remap_np[np.asarray(x, dtype=np.int64)].tolist() for x in ids]
        else:
            enc["input_ids"] = self.remap_np[np.asarray(ids, dtype=np.int64)].tolist()
        return enc

    def __getattr__(self, name):
        return getattr(self.tok, name)


def corpus_texts(seeds=(0, 1, 2)):
    """train + test + augmented training views in both augmentation modes (the inputs the model was trained on)."""
    import os
    import augment
    tr = pd.read_csv(ROOT / "data/train.csv").query_text.tolist()
    te = pd.read_csv(ROOT / "data/test.csv").query_text.tolist()
    out = tr + te
    old = os.environ.get("AUG_MODE")
    try:
        for mode in ("v007", "mined"):
            os.environ["AUG_MODE"] = mode
            for s in seeds:
                rng = random.Random(1000 + s)
                out += [augment.augment(t, rng) for t in tr]
    finally:
        if old is None:
            os.environ.pop("AUG_MODE", None)
        else:
            os.environ["AUG_MODE"] = old
    return out, te


def keep_ids(tok, texts, top):
    used = set()
    for i in range(0, len(texts), 2048):
        for ids in tok(texts[i:i + 2048], add_special_tokens=True)["input_ids"]:
            used.update(ids)
    vocab = len(tok)
    keep = used | set(tok.all_special_ids) | set(range(min(top, vocab)))
    return np.array(sorted(keep), dtype=np.int64), len(used)


@torch.inference_mode()
def logits(model, tok, texts, max_len, dev, bs=128):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        out.append(model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def trim(full_dir, out_dir=None, top=30000):
    from transformers import AutoConfig
    from verify_model import Brain, load
    full_dir = Path(full_dir)
    out_dir = Path(out_dir) if out_dir else full_dir.parent / "trim"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok, cfg = load(full_dir)  # untrimmed brain (also caches encoder_config/ next to model.pt)
    vocab = model.enc.config.vocab_size
    texts, te = corpus_texts()
    keep, n_used = keep_ids(tok, texts, top)
    pad = tok.pad_token_id
    assert int(np.searchsorted(keep, pad)) == pad, "pad id must keep its index (RoBERTa position ids depend on it)"

    state = {k: v for k, v in model.state_dict().items()}
    state[EMB_KEY] = state[EMB_KEY][torch.from_numpy(keep)].clone()
    enc_cfg = AutoConfig.from_pretrained(full_dir / "encoder_config")
    enc_cfg.vocab_size = len(keep)
    enc_cfg.save_pretrained(out_dir / "encoder_config")
    small = Brain(cfg["hf_name"], local_config=out_dir / "encoder_config")
    small.load_state_dict(state, strict=True)
    small.eval()
    torch.save(small.state_dict(), out_dir / "model.pt")
    np.save(out_dir / "keep.npy", keep)
    if (out_dir / "tokenizer").exists():
        shutil.rmtree(out_dir / "tokenizer")
    shutil.copytree(full_dir / "tokenizer", out_dir / "tokenizer")

    # equality check on every test query: untrimmed vs trimmed, same precision (fp32), same device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rtok = RemapTokenizer(tok, keep, vocab)
    big = logits(model.to(dev), tok, te, cfg["max_len"], dev)
    model.to("cpu")
    lo = logits(small.to(dev), rtok, te, cfg["max_len"], dev)
    same = float(np.mean(big.argmax(1) == lo.argmax(1)))
    diff = float(np.abs(big - lo).max())
    np.save(out_dir / "test_logits.npy", lo.astype(np.float32))
    n_full = sum(p.numel() for p in model.parameters())
    n_small = sum(p.numel() for p in small.parameters())
    res = {"n_params": int(n_small), "n_params_untrimmed": int(n_full), "trimmed_from": str(full_dir),
           "vocab_original": int(vocab), "kept_tokens": int(len(keep)), "tokens_used_by_competition_text": int(n_used),
           "top": int(top), "test_identical_to_untrimmed": same, "test_max_abs_logit_diff": diff}
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    assert same == 1.0 and diff < 1e-3, "trimmed model must predict exactly like the untrimmed one"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("full_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=30000)
    a = ap.parse_args()
    trim(a.full_dir, a.out, a.top)


if __name__ == "__main__":
    main()
