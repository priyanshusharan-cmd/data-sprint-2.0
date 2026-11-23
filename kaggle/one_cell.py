# Data Sprint 2.0 NLP: one-cell Kaggle runner. Paste this whole file into ONE empty notebook cell.
# Notebook settings: Accelerator = GPU T4 x2 (or P100), Internet = On, Add-ons -> Secrets -> GITHUB_TOKEN (ticked).
# Run with Save Version -> Save & Run All (Commit). Finished runs/folds are skipped when re-run.
# This notebook only trains. It never submits to the leaderboard.

BRANCH = "exp/priyanshu"
JOBS = [
    # 1) HingRoBERTa fold 0: second-backbone check (compare val_clean with MuRIL fold 0 = 0.885)
    "python kaggle/train.py --model hingroberta --folds 0 --lr 5e-5 --epochs 10 --run hing_lr5",
    # 2) domain-adaptive MLM on train+test text (~8 min), then 3) MuRIL fold 0 with all upgrades
    "python kaggle/mlm_pretrain.py --model muril --epochs 2 --out /kaggle/working/runs/mlm_muril",
    "python kaggle/train.py --model muril --folds 0 --lr 6e-5 --llrd 0.95 --epochs 10 --run muril_up"
    " --init_from /kaggle/working/runs/mlm_muril --ema 0.999 --fgm 1.0 --importance 0.5",
]

import glob, os, shutil, subprocess
from kaggle_secrets import UserSecretsClient

_tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
shutil.rmtree("/tmp/repo", ignore_errors=True)
subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH,
                f"https://x-access-token:{_tok}@github.com/priyanshusharan-cmd/data-sprint-2.0.git", "/tmp/repo"], check=True)
del _tok
subprocess.run(["git", "-C", "/tmp/repo", "remote", "set-url", "origin",
                "https://github.com/priyanshusharan-cmd/data-sprint-2.0.git"], check=True)
print(subprocess.check_output(["git", "-C", "/tmp/repo", "log", "-1", "--format=%h %s"], text=True))

os.makedirs("/kaggle/working/runs", exist_ok=True)
for src in glob.glob("/kaggle/input/*/runs/*/*"):  # resume: attach a previous version's output as Input
    dst = "/kaggle/working/runs/" + "/".join(src.split("/")[-2:])
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst); print("resumed", dst)
get_ipython().system("nvidia-smi --query-gpu=name,memory.total --format=csv")

for cmd in JOBS:
    out = "" if "--out" in cmd else " --out /kaggle/working/runs"
    print("\n>>>", cmd + out, flush=True)
    get_ipython().system(f"cd /tmp/repo && {cmd}{out}")

# small results zip (fold outputs + metrics); weights/MLM checkpoints stay as separate output files
get_ipython().system("cd /kaggle/working && zip -qr runs_all.zip runs -x 'runs/*/full/model.pt' 'runs/mlm_*/*' && ls -la runs_all.zip")
get_ipython().system("grep -H '\"best_score\"' /kaggle/working/runs/*/fold*/metrics.json")
