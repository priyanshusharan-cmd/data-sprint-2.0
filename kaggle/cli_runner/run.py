"""Kaggle script kernel pushed by the CLI (kaggle kernels push -p kaggle/cli_runner --accelerator NvidiaTeslaT4).
Code + data come from the PRIVATE dataset priyanshusharan/datasprint-code (a git archive of the branch), so no
GitHub token is needed. Trains only; never submits to the leaderboard."""
import glob, os, shutil, subprocess, tarfile

JOBS = [
    # 1) HingRoBERTa fold 0: second-backbone check (compare val_clean with MuRIL fold 0 = 0.885)
    "python kaggle/train.py --model hingroberta --folds 0 --lr 5e-5 --epochs 10 --run hing_lr5",
    # 2) domain-adaptive MLM on train+test text, then 3) MuRIL fold 0 with all upgrades
    "python kaggle/mlm_pretrain.py --model muril --epochs 2 --out /kaggle/working/runs/mlm_muril",
    "python kaggle/train.py --model muril --folds 0 --lr 6e-5 --llrd 0.95 --epochs 10 --run muril_up"
    " --init_from /kaggle/working/runs/mlm_muril --ema 0.999 --fgm 1.0 --importance 0.5",
]

REPO = "/tmp/repo"
shutil.rmtree(REPO, ignore_errors=True); os.makedirs(REPO)
arc = glob.glob("/kaggle/input/**/code.tar.gz", recursive=True)
if arc:
    tarfile.open(arc[0]).extractall(REPO)
else:  # Kaggle may have unpacked the archive
    root = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/kaggle/train.py", recursive=True)[0]))
    shutil.copytree(root, REPO, dirs_exist_ok=True)
for c in glob.glob("/kaggle/input/**/COMMIT.txt", recursive=True):
    print("code commit:", open(c).read().strip(), flush=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
os.makedirs("/kaggle/working/runs", exist_ok=True)
for cmd in JOBS:
    full = cmd if "--out" in cmd else cmd + " --out /kaggle/working/runs"
    print("\n>>>", full, flush=True)
    r = subprocess.run(full, shell=True, cwd=REPO)
    print(f"<<< exit code {r.returncode}", flush=True)
subprocess.run("cd /kaggle/working && zip -qr runs_all.zip runs -x 'runs/*/full/model.pt' 'runs/mlm_*/*' && ls -la runs_all.zip", shell=True)
subprocess.run("grep -H '\"best_score\"' /kaggle/working/runs/*/fold*/metrics.json", shell=True)
