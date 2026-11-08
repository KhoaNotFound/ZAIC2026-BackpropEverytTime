"""Download the wheels this environment is missing, for offline installs later.

Asks pip which packages `pip install -r requirements.txt` would add or upgrade
here (a dry run), then downloads exactly those, skipping torch and the CUDA
libraries the Kaggle image already provides. Install them later, offline, with

    pip install --no-index --no-deps <dir>/*

Run it on the same image (e.g. a Kaggle GPU notebook) that will use the wheels.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Provided by the Kaggle image with a CUDA-matched torch build; never replace them.
SKIP_PREFIXES = ("torch", "nvidia-", "triton", "cuda-")


def missing_packages(requirements):
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
                "-r", str(requirements), "--report", str(report),
            ],
            check=True,
        )
        items = json.loads(report.read_text())["install"]
    packages = [(i["metadata"]["name"], i["metadata"]["version"]) for i in items]
    return [
        (name, version)
        for name, version in packages
        if not name.lower().replace("_", "-").startswith(SKIP_PREFIXES)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requirements", type=Path, default=Path("requirements.txt"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    packages = missing_packages(args.requirements)
    print("Missing here:", packages or "nothing", flush=True)
    for name, version in packages:
        subprocess.run(
            [
                sys.executable, "-m", "pip", "download", "--quiet", "--no-deps",
                "-d", str(args.output), f"{name}=={version}",
            ],
            check=True,
        )
    (args.output / "packages.json").write_text(json.dumps(packages, indent=2) + "\n")
    print("Wheels:", sorted(p.name for p in args.output.iterdir()), flush=True)


if __name__ == "__main__":
    main()
