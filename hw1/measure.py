"""Замеры в Colab: main('smoke') для одной точки, main('grid') для всей сетки."""

import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from check_environment import main as check_environment
from models import build_model

BASE_IMAGE_SIZES = (32, 64, 128, 224, 256, 384, 512)
BASE_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEED = 42
WARMUP = 10
REPEATS = 30
ENERGY_SECONDS = 2.0


def measurement_grid():
    """63 базовые точки для calibration; 69 остальных — validation."""
    rng = random.Random(SEED)
    sizes = sorted((*BASE_IMAGE_SIZES, *rng.sample(
        [s for s in range(32, 513, 16) if s not in BASE_IMAGE_SIZES], 4,
    )))
    batches = sorted((*BASE_BATCH_SIZES, *rng.sample(
        [b for b in range(1, 257) if b not in BASE_BATCH_SIZES], 3,
    )))
    return [(s, b, s not in BASE_IMAGE_SIZES or b not in BASE_BATCH_SIZES)
            for s in sizes for b in batches]


@torch.inference_mode()
def measure_latency_memory(model, x):
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()

    # Baseline уже содержит параметры и исходный x. Ничего не вычитаем.
    torch.cuda.reset_peak_memory_stats()
    output = model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    if output.shape != (x.shape[0], 100) or not torch.isfinite(output).all().item():
        raise RuntimeError("Сеть вернула неверную форму или нечисловые значения.")
    del output

    samples = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = perf_counter()
        output = model(x)
        torch.cuda.synchronize()
        samples.append(perf_counter() - start)
        # Предыдущий выход не должен жить во время следующего forward.
        del output
    return median(samples), peak, samples


@torch.inference_mode()
def measure_energy(model, x, nvml, handle):
    """Энергия всей GPU за >=2 секунды / число завершённых forward."""
    torch.cuda.synchronize()
    start_mj = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    start = perf_counter()
    count = 0
    while count == 0 or perf_counter() - start < ENERGY_SECONDS:
        output = model(x)
        torch.cuda.synchronize()
        del output
        count += 1
    end_mj = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    elapsed = perf_counter() - start
    if end_mj <= start_mj:
        raise RuntimeError("Счётчик энергии NVML не увеличился.")
    return (end_mj - start_mj) / 1000.0 / count, count, elapsed


def layer_kernels(events):
    """GPU-ядра связываются со слоем по дереву CPU-событий профайлера."""
    names = set()
    for event in events:
        if not event.kernels:
            continue
        parent = event
        while parent is not None and not parent.name.startswith("layer_"):
            parent = parent.cpu_parent
        if parent is not None:
            names.update((parent.name, kernel.name) for kernel in event.kernels)
    return sorted(names)


@torch.inference_mode()
def profile_kernels(model, x):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as profile:
        # Kineto/CUPTI может терять первые GPU-события после простоя.
        # Прогреваем сборщик внутри capture; эти вызовы не имеют layer-меток.
        # https://github.com/pytorch/pytorch/issues/192021
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        value = x
        for index, layer in enumerate(model, start=1):
            with torch.profiler.record_function(f"layer_{index:02d}:{type(layer).__name__}"):
                value = layer(value)
        torch.cuda.synchronize()
        del value
    names = layer_kernels(profile.events())
    # Flatten — view, поэтому отдельного GPU-ядра у него нет.
    expected = {f"layer_{i:02d}:{type(layer).__name__}"
                for i, layer in enumerate(model, start=1)
                if not isinstance(layer, torch.nn.Flatten)}
    missing = expected - {layer for layer, _ in names}
    if missing:
        raise RuntimeError(f"Не найдены GPU-ядра слоёв: {sorted(missing)}")
    return names


def measure_point(model, size, batch, nvml, handle):
    """Нехватка памяти в forward не прерывает измерения остальных точек."""
    row = {"S": size, "B": batch, "status": "ok"}
    samples, kernels = [], []
    x = None
    try:
        torch.manual_seed(SEED + size * 257 + batch)
        x = torch.randn(batch, 3, size, size, device="cuda", dtype=torch.float32)
        latency, memory, samples = measure_latency_memory(model, x)
        row.update(latency_s=latency, memory_bytes=memory)
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    else:
        # Эти два запуска идут ПОСЛЕ основных замеров и не меняют их результат.
        try:
            energy, count, elapsed = measure_energy(model, x, nvml, handle)
            row.update(energy_j=energy, energy_forwards=count, energy_duration_s=elapsed)
        except (nvml.NVMLError, RuntimeError) as error:
            row["energy_error"] = str(error)
        try:
            kernels = profile_kernels(model, x)
        except RuntimeError as error:
            row["kernels_error"] = str(error)
    finally:
        del x
    return row, samples, kernels


def main(mode="smoke"):
    if mode not in {"smoke", "grid"}:
        raise ValueError("Режим должен быть smoke или grid.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(SEED)

    results = Path("results/smoke" if mode == "smoke" else "results")
    if (results / "measurements.csv").exists():
        raise FileExistsError(f"В {results} уже есть замеры. Сначала сохрани их в другой папке.")
    environment = check_environment(results)
    import pynvml as nvml

    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(environment["gpu_uuid"])
        model = build_model().cuda().eval()
        grid = [(32, 1, False)] if mode == "smoke" else measurement_grid()
        sources = ("models.py", "equations.py", "measure.py", "check_environment.py", "requirements.txt")
        source_dir = Path(__file__).resolve().parent
        protocol = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode, "seed": SEED, "input_seed": "42 + S*257 + B",
            "dtype": "float32", "eval": True, "inference_mode": True,
            "warmup": WARMUP, "latency_repeats": REPEATS,
            "latency_method": "median perf_counter, CUDA synchronize before/after each forward",
            "memory_method": "max_memory_allocated after reset, parameters and input included",
            "energy_method": "NVML total energy counter delta / completed forwards; no idle subtraction",
            "energy_min_duration_s": ENERGY_SECONDS, "nvidia_ml_py": version("nvidia-ml-py"),
            "profiler_warmup": WARMUP,
            "is_validation_rule": "S not in base sizes OR B not in base batches",
            "grid": [{"S": s, "B": b, "is_validation": v} for s, b, v in grid],
            "source_sha256": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                              for name in sources},
        }
        protocol_path = results / "protocol.json"
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        fields = ["S", "B", "latency_s", "memory_bytes", "energy_j", "status", "is_validation",
                  "energy_forwards", "energy_duration_s", "energy_error", "kernels_error"]
        with (results / "measurements.csv").open("w", newline="") as csv_file, \
             (results / "kernels.csv").open("w", newline="") as kernel_file, \
             (results / "latency_samples.csv").open("w", newline="") as sample_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            kernel_writer = csv.writer(kernel_file)
            sample_writer = csv.writer(sample_file)
            writer.writeheader()
            kernel_writer.writerow(["S", "B", "layer", "kernel_name"])
            sample_writer.writerow(["S", "B", "repeat", "latency_s"])
            for index, (size, batch, validation) in enumerate(grid, start=1):
                torch.cuda.empty_cache()
                print(f"[{index}/{len(grid)}] S={size}, B={batch}", flush=True)
                row, samples, kernels = measure_point(model, size, batch, nvml, handle)
                row["is_validation"] = validation
                writer.writerow(row)
                kernel_writer.writerows((size, batch, layer, kernel) for layer, kernel in kernels)
                sample_writer.writerows((size, batch, i, t) for i, t in enumerate(samples, start=1))
                for output in (csv_file, kernel_file, sample_file):
                    output.flush()
                print(row, flush=True)
        protocol["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        del model
        torch.cuda.empty_cache()
    finally:
        nvml.nvmlShutdown()
    print(f"Saved: {results}")


if __name__ == "__main__":
    main()
