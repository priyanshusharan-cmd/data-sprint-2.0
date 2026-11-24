"""
Vocabulary trimming: shrink the word-embedding matrix of fine-tuned encoders so that two encoders + TF-IDF fit the
300M parameter cap. Weights of every kept token are untouched, so outputs are identical for any text whose tokens
are all kept (all train/test tokens are). Dropped tokens map to <unk> at inference.

Kept tokens: special tokens + every token produced on train, test and 5 augmented views of train + tokens made only
of Latin / Devanagari / digits / punctuation; if still over budget, the highest-id unused tokens are dropped.

    python tools/trim_vocab.py --total 299500000   # writes runs/hing_full_trim/full and runs/muril_full_trim/full
"""
import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from augment import augment  # noqa: E402
from verify_model import Brain, load  # noqa: E402

SCRIPT_OK = re.compile(r"^[▁ऀ-ॿ᳐-᳿꣠-ꣿa-zA-Z0-9\s.,!?'\"\-:;()/&%@#+*=_]*$")
MODELS = {"hing": "runs/hing_full/full", "muril": "runs/muril_full/full"}
SVM_PARAMS = 9_291_576


def used_ids(tok, texts):
    return {i for t in texts for i in tok(t, truncation=True, max_length=64)["input_ids"]}


def body_params(model):
    return sum(p.numel() for p in model.parameters()) - model.enc.get_input_embeddings().weight.numel()


@torch.inference_mode()
def logits(model, tok, texts, id_map=None, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=64, return_tensors="pt")
        ids = b["input_ids"] if id_map is None else torch.from_numpy(id_map[b["input_ids"].numpy()])
        out.append(model(ids, b["attention_mask"]).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=299_500_000, help="target total params incl. the TF-IDF SVM")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv"); te = pd.read_csv(ROOT / "data/test.csv")
    rng = random.Random(0)
    texts = list(tr.query_text) + list(te.query_text) + [augment(t, rng) for t in tr.query_text for _ in range(5)]

    loaded, keep = {}, {}
    for name, d in MODELS.items():
        model, tok, cfg = load(ROOT / d)
        V = model.enc.get_input_embeddings().weight.shape[0]
        used = used_ids(tok, texts) | set(tok.all_special_ids)
        script = {i for t, i in tok.get_vocab().items() if i < V and SCRIPT_OK.match(t.replace("##", ""))}
        loaded[name] = (model, tok, cfg, V, used, script)
        print(f"{name}: vocab {V:,} | used {len(used):,} | script-ok {len(script):,}")

    emb_budget = (a.total - SVM_PARAMS - sum(body_params(m) for m, *_ in loaded.values())) // 768
    n_used = {n: len(v[4]) for n, v in loaded.items()}
    spare = emb_budget - sum(n_used.values())
    n_script_extra = {n: len(v[5] - v[4]) for n, v in loaded.items()}
    scale = min(1.0, spare / sum(n_script_extra.values()))
    for name, (model, tok, cfg, V, used, script) in loaded.items():
        extra = sorted(script - used)[: int(n_script_extra[name] * scale)]  # lowest ids first = most frequent
        keep[name] = np.array(sorted(used | set(extra)))
        print(f"{name}: keep {len(keep[name]):,} tokens ({len(extra):,} unused script tokens kept, scale {scale:.3f})")

    report = {}
    for name, (model, tok, cfg, V, used, script) in loaded.items():
        k = keep[name]
        unk = tok.unk_token_id
        assert unk in set(k) and all(i == j for i, j in zip(k[:4], range(4))) or name == "muril", "special ids must keep their positions"
        id_map = np.full(V, int(np.searchsorted(k, unk)), dtype=np.int64)
        id_map[k] = np.arange(len(k))
        src = ROOT / MODELS[name]
        dst = ROOT / f"runs/{name}_full_trim/full"
        dst.mkdir(parents=True, exist_ok=True)
        enc_cfg = model.enc.config.__class__.from_dict(model.enc.config.to_dict())
        enc_cfg.vocab_size = len(k)
        enc_cfg.save_pretrained(dst / "encoder_config")
        small = Brain(cfg["hf_name"], local_config=dst / "encoder_config")
        state = {kk: v for kk, v in model.state_dict().items()}
        emb_key = [kk for kk in state if kk.endswith("word_embeddings.weight")][0]
        state[emb_key] = state[emb_key][torch.from_numpy(k)].clone()
        small.load_state_dict(state, strict=True); small.eval()
        n_params = sum(p.numel() for p in small.parameters())
        torch.save(small.state_dict(), dst / "model.pt")
        np.save(dst / "id_map.npy", id_map)
        shutil.copytree(src / "tokenizer", dst / "tokenizer", dirs_exist_ok=True)
        shutil.copy(src.parent / "config.json", dst.parent / "config.json")

        full_l = logits(model, tok, te.query_text.values)
        trim_l = logits(small, tok, te.query_text.values, id_map)
        diff = float(np.abs(full_l - trim_l).max())
        same = float(np.mean(full_l.argmax(1) == trim_l.argmax(1)))
        np.save(dst / "test_logits.npy", trim_l.astype(np.float32))
        (dst / "metrics.json").write_text(json.dumps({"n_params": n_params, "vocab_kept": int(len(k)), "vocab_orig": int(V),
                                                      "max_abs_logit_diff_vs_untrimmed": diff, "argmax_agreement": same}))
        report[name] = n_params
        print(f"{name}: {n_params:,} params | test logits max |diff| vs untrimmed {diff:.2e} | argmax agreement {same:.4f}")

    total = sum(report.values()) + SVM_PARAMS
    print(f"TOTAL = {report['hing']:,} + {report['muril']:,} + {SVM_PARAMS:,} = {total:,}")
    assert total <= 300_000_000


if __name__ == "__main__":
    main()
