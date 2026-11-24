# Data Sprint 2.0 NLP: one-cell Kaggle runner for the backbone scan. Paste this whole file into ONE empty notebook cell
# (or import kaggle/backbone_scan.ipynb).
# Notebook settings: Accelerator = GPU T4 x2, Internet = On, Add-ons -> Secrets -> GITHUB_TOKEN (ticked).
# No dataset input needed: the code AND the competition CSVs (data/) come from the private GitHub repo.
# Run with Save Version -> Save & Run All (Commit), or just run the cell interactively and watch the log.
# Takes ~50 min on T4 x2 (two jobs at once, one per GPU). It only trains: it never submits to the leaderboard.
# When done: Output -> download runs_all.zip (small: fold-0 + full-model predictions, no weights) and give it to Claude.

BRANCH = "improvements"
HMBERT = "--model l3cube-pune/hing-mbert-mixed --lr 5e-5 --llrd 0.95 --epochs 10 --seed 42"   # Hinglish mBERT, 178M
INDIC = "--model indicbert --lr 5e-5 --llrd 0.95 --epochs 10 --seed 42"                      # IndicBERTv2, 278M
QUEUES = {  # GPU 0: fold 0 of each encoder (the gate vs v007) | GPU 1: full-data models (for the submission file)
    "0": [f"python kaggle/train.py {HMBERT} --folds 0 --run hmbert_ft",
          f"python kaggle/train.py {INDIC} --folds 0 --run indic_ft"],
    "1": [f"python kaggle/train.py {HMBERT} --full --run hmbert_full",
          f"python kaggle/train.py {INDIC} --full --run indic_full"],
}

import os, shutil, subprocess, threading, time
from kaggle_secrets import UserSecretsClient

_tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
shutil.rmtree("/tmp/repo", ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH,
                f"https://x-access-token:{_tok}@github.com/priyanshusharan-cmd/data-sprint-2.0.git", "/tmp/repo"], check=True)
del _tok
subprocess.run(["git", "-C", "/tmp/repo", "remote", "set-url", "origin",
                "https://github.com/priyanshusharan-cmd/data-sprint-2.0.git"], check=True)
print(subprocess.check_output(["git", "-C", "/tmp/repo", "log", "-1", "--format=%h %s"], text=True), flush=True)
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv", shell=True)
OUT = "/kaggle/working/runs"
os.makedirs(OUT, exist_ok=True)
n_gpu = int(subprocess.run(["python", "-c", "import torch; print(torch.cuda.device_count())"],
                           capture_output=True, text=True).stdout.strip() or 0)
lock, status = threading.Lock(), []


def package():
    with lock:
        subprocess.run("cd /kaggle/working && rm -f runs_all.zip && zip -qr runs_all.zip runs -x 'runs/*/full/model.pt' "
                       "'runs/*/fold*/*emb.npy' && ls -la runs_all.zip", shell=True)
        subprocess.run(f"grep -H '\"best_score\"' {OUT}/*/fold*/metrics.json", shell=True)


def run_queue(gpu, jobs):
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               CUDA_VISIBLE_DEVICES=gpu if n_gpu > 1 else "0")
    for cmd in jobs:
        cmd += f" --out {OUT}"
        print(f"\n[gpu{gpu}] >>> {cmd}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, shell=True, cwd="/tmp/repo", env=env)  # a failing job does not stop the queue
        with lock:
            status.append((gpu, cmd.split()[3], r.returncode, round((time.time() - t0) / 60, 1)))
        print(f"[gpu{gpu}] <<< exit {r.returncode} after {status[-1][3]} min", flush=True)
        package()


if n_gpu > 1:
    ts = [threading.Thread(target=run_queue, args=(g, j)) for g, j in QUEUES.items()]
    [t.start() for t in ts]; [t.join() for t in ts]
else:
    for g, j in QUEUES.items():
        run_queue(g, j)
print("\nJOB SUMMARY (gpu, model, exit code, minutes):", status, flush=True)
