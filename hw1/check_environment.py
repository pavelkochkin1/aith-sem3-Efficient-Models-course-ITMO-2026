"""Проверить CUDA в Colab и сохранить сведения об окружении в results/."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch


def main(output_dir="results"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Создай Colab-сессию с GPU T4.")

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Для этого эксперимента должна быть видна ровно одна GPU.")
    gpu = torch.cuda.get_device_properties(0)
    driver, uuid = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=driver_version,uuid", "--format=csv,noheader"],
        text=True,
    ).strip().split(", ")
    environment = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "numpy": version("numpy"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu.name,
        "gpu_memory_bytes": gpu.total_memory,
        "driver": driver,
        "gpu_uuid": uuid,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    results = Path(output_dir)
    results.mkdir(parents=True, exist_ok=True)
    output = json.dumps(environment, indent=2, ensure_ascii=False)
    (results / "environment.json").write_text(output + "\n", encoding="utf-8")
    print(output)
    print(f"Saved: {results / 'environment.json'}")
    return environment


if __name__ == "__main__":
    main()
