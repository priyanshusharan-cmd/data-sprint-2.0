"""Param count (from config, no weights) + tokenizer fit on our test text for each candidate backbone."""
import warnings; warnings.filterwarnings("ignore")
import sys, torch, pandas as pd
from transformers import AutoConfig, AutoModel, AutoTokenizer
sys.path.insert(0, "src")
te = pd.read_csv("data/test.csv").query_text
dev = te[te.str.contains("[ऀ-ॿ]")].tolist()[:800]; rom = te[~te.str.contains("[ऀ-ॿ]")].tolist()[:800]
C = ["google/muril-base-cased", "xlm-roberta-base", "l3cube-pune/hing-roberta-mixed", "l3cube-pune/hing-mbert-mixed",
     "ai4bharat/IndicBERTv2-MLM-Sam-TLM", "microsoft/mdeberta-v3-base", "intfloat/multilingual-e5-base",
     "intfloat/multilingual-e5-small", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
print(f"{'model':58s} {'params':>8s} {'vocab':>7s} {'tok/word dev':>12s} {'rom':>5s} {'unk%':>5s}")
for m in C:
    try:
        cfg = AutoConfig.from_pretrained(m)
        with torch.device("meta"): n = sum(p.numel() for p in AutoModel.from_config(cfg).parameters())
        tok = AutoTokenizer.from_pretrained(m)
        f = lambda xs: sum(len(tok.tokenize(x)) for x in xs) / sum(len(x.split()) for x in xs)
        ids = [i for x in dev + rom for i in tok(x, add_special_tokens=False)["input_ids"]]
        unk = 100 * sum(i == tok.unk_token_id for i in ids) / len(ids)
        print(f"{m:58s} {n/1e6:7.0f}M {cfg.vocab_size:7d} {f(dev):12.2f} {f(rom):5.2f} {unk:5.2f}")
    except Exception as e:
        print(f"{m:58s} ERROR {type(e).__name__}: {str(e)[:80]}")
