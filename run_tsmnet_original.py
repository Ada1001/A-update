"""Run the unmodified author experiment in its own Python environment.

Use --prepare first, create the environment from the downloaded environment.yaml,
then run this launcher with that environment's Python and Hydra overrides after --.
"""
import argparse
from pathlib import Path
import subprocess
import sys

COMMIT = "90293b9d2982fa06a3030340d24d287155fd5a89"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--repo-dir", default="external/TSMNet-paper")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    repo = Path(args.repo_dir).resolve()
    git = ["git", "-c", "safe.directory="+repo.as_posix(), "-C", str(repo)]
    if not repo.exists():
        subprocess.run(["git", "clone", "https://github.com/rkobler/TSMNet.git", str(repo)],check=True)
        subprocess.run(git+["checkout", "--detach", COMMIT],check=True)
    head = subprocess.check_output(git+["rev-parse","HEAD"],text=True).strip()
    if head != COMMIT:
        raise RuntimeError("Existing checkout is not the pinned author commit; use another --repo-dir")
    changed = subprocess.check_output(git+["diff","HEAD","--",
        "spdnets","experiments","library","datasetio","environment.yaml"],text=True)
    if changed.strip():
        raise RuntimeError("Author experiment sources were modified; use a clean checkout")
    if args.prepare:
        print("Prepared author commit",head,"at",repo)
        print("Create a separate conda environment from",repo/"environment.yaml")
        return
    overrides = args.overrides[1:] if args.overrides[:1] == ["--"] else args.overrides
    subprocess.run([sys.executable,"main.py",*overrides],cwd=str(repo/"experiments"),check=True)


if __name__ == "__main__":
    main()
