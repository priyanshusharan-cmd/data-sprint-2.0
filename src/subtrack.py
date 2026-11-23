"""
Submission registry: freeze every candidate submission, tie it to the git commit that produced it,
enforce the Kaggle daily quota, and record the public LB score. Standard library only.

    python src/subtrack.py add submissions/muril.csv --stage muril-5fold --note "first GPU run"
    python src/subtrack.py list
    python src/subtrack.py submitted v003            # after uploading submissions/v003_*.csv on Kaggle
    python src/subtrack.py lb v003 0.8412            # public LB score shown by Kaggle
    python src/subtrack.py show v003

`add` validates the file against data/test.csv, copies it to submissions/vNNN_<stage>.csv (frozen:
postprocess.py overwrites its own outputs on every run), copies the sidecar .json if present, and records
the commit hash plus whether the working tree had uncommitted changes. `submitted` creates the git tag
sub-vNNN on that commit. Push tags with:  git push origin --tags
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"
REG = SUB / "registry.csv"
DAILY_LIMIT = 5  # Kaggle resets the count at 00:00 UTC (05:30 IST)
FIELDS = ["version", "created_utc", "stage", "file", "commit", "dirty", "sha256", "oof_f1", "class55_share",
          "submitted_utc", "public_lb", "note"]


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def load():
    if not REG.exists():
        return []
    with open(REG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows):
    with open(REG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def find(rows, v):
    v = v if v.startswith("v") else f"v{int(v):03d}"
    for r in rows:
        if r["version"] == v:
            return r
    sys.exit(f"no version {v} in {REG.relative_to(ROOT)}")


def validate(path):
    """Return (n_rows, class55_share, sha256) or exit with the problem."""
    with open(ROOT / "data/test.csv", newline="", encoding="utf-8") as f:
        test_ids = [r["example_id"] for r in csv.DictReader(f)]
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != ["example_id", "target"]:
            sys.exit(f"bad header {rd.fieldnames}; expected ['example_id', 'target']")
        rows = list(rd)
    ids = [r["example_id"] for r in rows]
    errs = []
    if len(ids) != len(test_ids):
        errs.append(f"{len(ids)} rows, test has {len(test_ids)}")
    if len(set(ids)) != len(ids):
        errs.append(f"{len(ids) - len(set(ids))} duplicate example_id")
    if set(ids) != set(test_ids):
        errs.append(f"{len(set(test_ids) - set(ids))} test ids missing, {len(set(ids) - set(test_ids))} unknown ids")
    bad = [r["target"] for r in rows if not r["target"].isdigit() or not 0 <= int(r["target"]) <= 55]
    if bad:
        errs.append(f"{len(bad)} targets not integers in 0..55, e.g. {bad[:3]}")
    if errs:
        sys.exit("INVALID submission " + str(path) + ":\n  " + "\n  ".join(errs))
    counts = Counter(int(r["target"]) for r in rows)
    missing = [c for c in range(56) if c not in counts]
    if missing:
        print(f"warning: classes never predicted: {missing}")
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    return len(rows), counts[55] / len(rows), sha


def submitted_today(rows):
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return sum(r["submitted_utc"].startswith(today) for r in rows)


def cmd_add(a):
    rows = load()
    src = Path(a.csv).resolve()
    n, share55, sha = validate(src)
    dup = [r["version"] for r in rows if r["sha256"] == sha]
    if dup and not a.force:
        sys.exit(f"identical predictions already registered as {dup[0]} (use --force to add anyway)")
    v = f"v{len(rows) + 1:03d}"
    stage = "".join(ch if ch.isalnum() or ch in "-_+." else "-" for ch in a.stage)
    dst = SUB / f"{v}_{stage}.csv"
    shutil.copyfile(src, dst)
    oof = a.oof
    side = src.with_suffix(".json")
    if side.exists():
        shutil.copyfile(side, dst.with_suffix(".json"))
        if oof is None:
            oof = json.loads(side.read_text()).get("oof_tuned")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", "src", "kaggle", "analysis"))
    rows.append({"version": v, "created_utc": now(), "stage": stage, "file": dst.name, "commit": git("rev-parse", "--short", "HEAD"),
                 "dirty": int(dirty), "sha256": sha, "oof_f1": "" if oof is None else f"{float(oof):.4f}",
                 "class55_share": f"{share55:.3f}", "submitted_utc": "", "public_lb": "", "note": a.note})
    save(rows)
    print(f"registered {v}: {dst.relative_to(ROOT)}  ({n} rows, class-55 share {share55:.3f}, sha {sha})")
    if dirty:
        print("warning: src/ kaggle/ analysis/ have uncommitted changes; commit them so this version is reproducible")
    print(f"upload {dst.relative_to(ROOT)} on Kaggle, then: python src/subtrack.py submitted {v}")


def cmd_submitted(a):
    rows = load()
    r = find(rows, a.version)
    if r["submitted_utc"]:
        sys.exit(f"{r['version']} already marked submitted at {r['submitted_utc']} UTC")
    used = submitted_today(rows)
    if used >= DAILY_LIMIT and not a.force:
        sys.exit(f"{used}/{DAILY_LIMIT} submissions already used today (UTC). Quota resets 00:00 UTC = 05:30 IST.")
    r["submitted_utc"] = now()
    save(rows)
    tag = f"sub-{r['version']}"
    if r["commit"] and not git("tag", "-l", tag):
        git("tag", "-a", tag, r["commit"], "-m", f"{r['version']} {r['stage']} {r['file']}")
    print(f"{r['version']} marked submitted ({used + 1}/{DAILY_LIMIT} today, UTC). Tag {tag} -> {r['commit']}.")
    print(f"when Kaggle shows the score: python src/subtrack.py lb {r['version']} <score>")


def cmd_lb(a):
    rows = load()
    r = find(rows, a.version)
    r["public_lb"] = f"{a.score:.5f}"
    if not r["submitted_utc"]:
        r["submitted_utc"] = now()
    save(rows)
    best = max((float(x["public_lb"]), x["version"]) for x in rows if x["public_lb"])
    print(f"{r['version']} public LB {a.score:.5f}. Best so far: {best[1]} = {best[0]:.5f}")


def cmd_list(a):
    rows = load()
    if not rows:
        print("registry empty"); return
    cols = ["version", "stage", "commit", "dirty", "oof_f1", "class55_share", "public_lb", "submitted_utc", "note"]
    width = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    width["note"] = min(width["note"], 40)
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(r[c][:width[c]].ljust(width[c]) for c in cols))
    print(f"\nsubmitted today (UTC): {submitted_today(rows)}/{DAILY_LIMIT}")


def cmd_show(a):
    print(json.dumps(find(load(), a.version), indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("add"); x.add_argument("csv"); x.add_argument("--stage", required=True)
    x.add_argument("--note", default=""); x.add_argument("--oof", type=float); x.add_argument("--force", action="store_true")
    x.set_defaults(f=cmd_add)
    x = s.add_parser("submitted"); x.add_argument("version"); x.add_argument("--force", action="store_true"); x.set_defaults(f=cmd_submitted)
    x = s.add_parser("lb"); x.add_argument("version"); x.add_argument("score", type=float); x.set_defaults(f=cmd_lb)
    x = s.add_parser("list"); x.set_defaults(f=cmd_list)
    x = s.add_parser("show"); x.add_argument("version"); x.set_defaults(f=cmd_show)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
