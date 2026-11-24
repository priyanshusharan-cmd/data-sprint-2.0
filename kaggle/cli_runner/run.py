"""Kaggle script kernel pushed by the CLI (kaggle kernels push -p kaggle/cli_runner --accelerator NvidiaTeslaT4).
Code + data come from the PRIVATE dataset priyanshusharan/datasprint-code (a git archive of the branch), so no
GitHub token is needed. Trains only; never submits to the leaderboard.

Two queues run at the same time, one per T4 GPU (accelerator "GPU T4 x2"):
  GPU 0  v007's HingRoBERTa exactly (seed 42): 5 folds -> OOF for the blend check, then the full-data model.
  GPU 1  a second HingRoBERTa: domain-adaptive MLM on the competition text, then seed 7 from that encoder:
         5 folds, then the full-data model. If the MLM step fails it trains from the public checkpoint instead.
  Then both GPUs share v005's MuRIL (folds 0-2 on GPU 0 + full model, folds 3-4 on GPU 1) so the three-model
  Phase 1 option (HingRoBERTa + MuRIL + TF-IDF) can be measured from this run alone.
Each full model is then vocabulary-trimmed (src/trim_vocab.py, lossless, asserts identical test predictions) so the
two transformers + TF-IDF fit under 300M. The untrimmed weights are deleted only after a successful trim.
"""
import glob, os, shutil, subprocess, sys, tarfile, threading, time

W = "/kaggle/working/runs"
HING = "--model hingroberta --lr 5e-5 --llrd 0.95 --epochs 10"
MURIL = "--model muril --lr 6e-5 --llrd 0.95 --epochs 10"   # v005's MuRIL settings (registry v003/v005)
TRIM = "--top 60000"   # keep every competition token + the 60k most frequent pieces (demo coverage); lossless
PLAN = "backbones"   # "backbones": try new encoders vs HingRoBERTa (fold 0 + full model in parallel per encoder)
                     # "phase2": second HingRoBERTa + MuRIL (the earlier plan, kept for reference)
INDIC = "--model indicbert --lr 5e-5 --llrd 0.95 --epochs 10 --seed 42"            # ai4bharat/IndicBERTv2-MLM-Sam-TLM (MIT)
HMBERT = "--model l3cube-pune/hing-mbert-mixed --lr 5e-5 --llrd 0.95 --epochs 10 --seed 42"   # Hinglish mBERT (CC-BY-4.0)
QUEUES_BACKBONES = {
    "0": [f"python kaggle/train.py {INDIC} --folds 0 --run indic_ft",
          f"python kaggle/train.py {HMBERT} --folds 0 --run hmbert_ft"],
    "1": [f"python kaggle/train.py {INDIC} --full --run indic_full",
          f"python kaggle/train.py {HMBERT} --full --run hmbert_full"],
}
QUEUES_PHASE2 = {
    "0": [
        f"python kaggle/train.py {HING} --seed 42 --run hing_lr5",
        f"python kaggle/train.py {HING} --seed 42 --full --run hing_full",
        f"python src/trim_vocab.py {W}/hing_full/full {TRIM} && rm -f {W}/hing_full/full/model.pt",
        f"python kaggle/train.py {MURIL} --seed 42 --folds 0 1 2 --run muril_lr6",
        f"python kaggle/train.py {MURIL} --seed 42 --full --run muril_full",
        f"python src/trim_vocab.py {W}/muril_full/full {TRIM} && rm -f {W}/muril_full/full/model.pt",
    ],
    "1": [
        f"python kaggle/mlm_pretrain.py --model hingroberta --epochs 3 --out {W}/mlm_hing",
        f"python kaggle/train.py {HING} --seed 7 --run hing2 INIT",
        f"python kaggle/train.py {HING} --seed 7 --full --run hing2_full INIT",
        f"python src/trim_vocab.py {W}/hing2_full/full {TRIM} && rm -f {W}/hing2_full/full/model.pt",
        f"rm -rf {W}/mlm_hing",
        f"python kaggle/train.py {MURIL} --seed 42 --folds 3 4 --run muril_lr6",
    ],
}

QUEUES = QUEUES_BACKBONES if PLAN == "backbones" else QUEUES_PHASE2
REPO = "/tmp/repo"
shutil.rmtree(REPO, ignore_errors=True); os.makedirs(REPO)
arc = glob.glob("/kaggle/input/**/code.tar.gz", recursive=True)
if arc:
    tarfile.open(arc[0]).extractall(REPO)
else:  # Kaggle may have unpacked the archive
    root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/kaggle/train.py", recursive=True)[0]))
    shutil.copytree(root, REPO, dirs_exist_ok=True)
env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
for c in glob.glob("/kaggle/input/**/COMMIT.txt", recursive=True):
    env["GIT_COMMIT"] = open(c).read().strip()
print("code commit:", env.get("GIT_COMMIT", "unknown"), "| python", sys.version.split()[0], flush=True)
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv", shell=True)
subprocess.run([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                "torch.cuda.device_count(), 'GPUs | transformers', transformers.__version__)"])
os.makedirs(W, exist_ok=True)
n_gpu = int(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
                           capture_output=True, text=True).stdout.strip() or 0)
lock = threading.Lock()
status = []


def package():
    with lock:
        subprocess.run("cd /kaggle/working && rm -f runs_all.zip && zip -qr runs_all.zip runs -x 'runs/*/full/model.pt' 'runs/*/fold*/*emb.npy' "
                       "'runs/*/trim/model.pt' 'runs/mlm_*/*' && ls -la runs_all.zip", shell=True)
        subprocess.run(f"grep -H '\"best_score\"' {W}/*/fold*/metrics.json", shell=True)


def run_queue(gpu, jobs):
    qenv = dict(env, CUDA_VISIBLE_DEVICES=gpu if n_gpu > 1 else "0")
    for cmd in jobs:
        if "INIT" in cmd:  # start from the domain-adapted encoder only if the MLM step finished
            ok = os.path.exists(f"{W}/mlm_hing/config.json")  # written by save_pretrained at the end
            cmd = cmd.replace("INIT", f"--init_from {W}/mlm_hing" if ok else "")
            if not ok:
                print(f"[gpu{gpu}] MLM output missing: training from the public checkpoint", flush=True)
        if "train.py" in cmd and "--out" not in cmd:
            cmd += f" --out {W}"
        print(f"\n[gpu{gpu}] >>> {cmd}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, shell=True, cwd=REPO, env=qenv)  # a failing job does not stop the queue
        with lock:
            status.append((gpu, cmd.split()[1], r.returncode, round((time.time() - t0) / 60, 1)))
        print(f"[gpu{gpu}] <<< exit code {r.returncode} after {status[-1][3]} min", flush=True)
        package()


if n_gpu > 1:
    threads = [threading.Thread(target=run_queue, args=(g, j)) for g, j in QUEUES.items()]
    [t.start() for t in threads]
    [t.join() for t in threads]
else:  # one GPU: run the queues one after the other
    for g, j in QUEUES.items():
        run_queue(g, j)
print("\nJOB SUMMARY (gpu, script, exit code, minutes):", status, flush=True)
subprocess.run(f"du -sh {W}/* 2>/dev/null", shell=True)
