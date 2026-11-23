"""Kaggle script kernel pushed by the CLI (kaggle kernels push -p kaggle/cli_runner --accelerator NvidiaTeslaT4).
Code + data come from the PRIVATE dataset priyanshusharan/datasprint-code (a git archive of the branch), so no
GitHub token is needed. Trains only; never submits to the leaderboard."""
import glob, os, shutil, subprocess, sys, tarfile, time

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
env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
for c in glob.glob("/kaggle/input/**/COMMIT.txt", recursive=True):
    env["GIT_COMMIT"] = open(c).read().strip()
print("code commit:", env.get("GIT_COMMIT", "unknown"), "| python", sys.version.split()[0], flush=True)
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
subprocess.run([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                "'| transformers', transformers.__version__)"])
os.makedirs("/kaggle/working/runs", exist_ok=True)


def package():
    """Zip fold outputs + metrics after every job, so a cancel/timeout never loses finished work."""
    subprocess.run("cd /kaggle/working && rm -f runs_all.zip && zip -qr runs_all.zip runs -x 'runs/*/full/model.pt' "
                   "'runs/mlm_*/*' && ls -la runs_all.zip", shell=True)
    subprocess.run("grep -H '\"best_score\"' /kaggle/working/runs/*/fold*/metrics.json", shell=True)


status = []
for cmd in JOBS:
    full = cmd if "--out" in cmd else cmd + " --out /kaggle/working/runs"
    print("\n>>>", full, flush=True)
    t0 = time.time()
    r = subprocess.run(full, shell=True, cwd=REPO, env=env)  # a failing job does not stop the queue
    status.append((cmd.split()[1], r.returncode, round((time.time() - t0) / 60, 1)))
    print(f"<<< exit code {r.returncode} after {status[-1][2]} min", flush=True)
    package()
print("\nJOB SUMMARY:", status, flush=True)
