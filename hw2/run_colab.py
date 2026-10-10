"""The exact three GPU experiment commands used for the submitted results."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
COMMANDS = [
    ["--model", "mlp", "--methods", "zero_paper", "zero_author", "kaiming", "xavier",
     "--seeds", "0", "1", "2", "--epochs", "14", "--width", "2048",
     "--batch-size", "64", "--lr", "0.1", "--lr-decay-epoch", "12",
     "--rank-epochs", "0", "1", "7", "14", "--output", "hw2/results/paper"],
    ["--model", "mlp", "--methods", "identity", "zero_all", "--seeds", "0",
     "--epochs", "14", "--width", "2048", "--batch-size", "64", "--lr", "0.1",
     "--lr-decay-epoch", "12", "--rank-epochs", "0", "1", "7", "14",
     "--output", "hw2/results/controls"],
    ["--model", "transformer", "--methods", "standard", "identity", "zero_inspired", "zero_inspired_author",
     "--seeds", "0", "1", "2", "--epochs", "10", "--width", "64", "--depth", "2",
     "--heads", "4", "--ff-width", "256", "--patch-size", "7", "--batch-size", "128",
     "--optimizer", "adamw", "--lr", "0.001", "--weight-decay", "0",
     "--output", "hw2/results/transformer"],
]


def main():
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required; Colab CLI installs it on the runtime")
    results = ROOT / "results"
    for group in ("paper", "controls", "transformer"):
        if (results / group).exists() and any((results / group).iterdir()):
            raise FileExistsError(f"Existing results in {results / group}; use a fresh workspace")
    results.mkdir(exist_ok=True)
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    for arguments in COMMANDS:
        subprocess.run([uv, "run", "--no-project", "--python", sys.executable, "python", "-u",
                        "hw2/train.py", *arguments, "--device", "cuda"],
                       cwd=PROJECT, env=environment, check=True)


if __name__ == "__main__":
    main()
