import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

try:
    from .models import MLP, PatchTransformer
except ImportError:
    from models import MLP, PatchTransformer


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def load_data(args):
    from torchvision import datasets, transforms
    transform = transforms.ToTensor()
    train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    for name, dataset, limit in (("train", train, args.train_limit), ("test", test, args.test_limit)):
        if not limit:
            continue
        if not 0 < limit <= len(dataset):
            raise ValueError(f"{name}-limit must be in 1..{len(dataset)} or 0 for all")
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(2026)).tolist()[:limit]
        if name == "train":
            train = Subset(dataset, indices)
        else:
            test = Subset(dataset, indices)
    return train, test


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return {"test_loss": loss / count, "test_accuracy": correct / count}


@torch.no_grad()
def spectrum_metrics(matrix, tolerances):
    # Double precision SVD: absolute thresholds, not default relative matrix_rank tolerances.
    # MPS does not support float64; rank diagnostics use CPU there.
    rank_device = matrix.device if matrix.device.type == "cuda" else torch.device("cpu")
    singular = torch.linalg.svdvals(matrix.to(device=rank_device, dtype=torch.float64))
    largest = singular[0].item() if len(singular) else 0.0
    result = {f"rank_tol_{tol:g}": int((singular > tol).sum()) for tol in tolerances}
    relative_threshold = max(matrix.shape) * torch.finfo(matrix.dtype).eps * largest
    result["rank_default_float32"] = int((singular > relative_threshold).sum())
    result["default_float32_threshold"] = relative_threshold
    result["stable_rank"] = (singular.square().sum().item() / largest**2) if largest else 0.0
    result["spectral_norm"] = largest
    return result


@torch.no_grad()
def rank_snapshot(model, probe, step, epoch, tolerances):
    model.eval()
    residual = model.hidden.weight - torch.eye(
        model.hidden.weight.shape[0], device=model.hidden.weight.device,
        dtype=model.hidden.weight.dtype,
    )
    features = model.features(probe)
    rows = []
    for name, matrix in (("W2_minus_I", residual), ("hidden_activations", features)):
        rows.append({"step": step, "epoch": epoch, "matrix": name,
                     "rows": matrix.shape[0], "columns": matrix.shape[1],
                     **spectrum_metrics(matrix, tolerances)})
    return rows


def gradient_diagnostics(model, images, labels, step):
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    gradients = {name: parameter.grad.norm().item() if parameter.grad is not None else None
                 for name, parameter in model.named_parameters()}
    with torch.no_grad():
        features = model.features(images)
        row = {"step": step, "probe_loss": loss.item(), "gradient_norms": gradients,
               "feature_zero_fraction": (features == 0).float().mean().item(),
               "feature_rms": features.square().mean().sqrt().item()}
        if isinstance(model, PatchTransformer):
            row["attention"] = model.attention_diagnostics(images)
    model.zero_grad(set_to_none=True)
    return row


def run(args, method, seed, train_data, test_data):
    set_seed(seed)
    device = torch.device(args.device)
    if args.model == "mlp":
        model = MLP(784, args.width, method)
    else:
        model = PatchTransformer(28, args.patch_size, args.width, args.heads,
                                 args.depth, args.ff_width, method)
    model.to(device)
    directory = Path(args.output) / f"{args.model}_{method}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed + 100000), num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    probe_items = [train_data[index] for index in range(min(args.probe_size, len(train_data)))]
    probe = torch.stack([item[0] for item in probe_items]).to(device)
    probe_labels = torch.tensor([item[1] for item in probe_items], device=device)
    diagnostic_probe, diagnostic_labels = probe[:args.batch_size], probe_labels[:args.batch_size]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epoch, args.lr_decay_gamma)
                 if args.lr_decay_epoch else None)
    start = time.perf_counter()
    metrics = [{"epoch": 0, "step": 0, "train_loss": None, "train_accuracy": None,
                **evaluate(model, test_loader, device), "lr": args.lr, "seconds": 0.0}]
    ranks, diagnostics = [], []
    if args.model == "mlp" and 0 in args.rank_epochs:
        ranks.extend(rank_snapshot(model, probe, 0, 0, args.rank_tolerances))
    if 0 in args.diagnostic_steps:
        diagnostics.append(gradient_diagnostics(model, diagnostic_probe, diagnostic_labels, 0))
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        used_lr = optimizer.param_groups[0]["lr"]
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {method}, seed {seed}, step {step}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
            step += 1
            if step in args.diagnostic_steps:
                diagnostics.append(gradient_diagnostics(model, diagnostic_probe, diagnostic_labels, step))
                model.train()
        result = {"epoch": epoch, "step": step, "train_loss": total_loss / count,
                  "train_accuracy": correct / count, **evaluate(model, test_loader, device),
                  "lr": used_lr, "seconds": time.perf_counter() - start}
        metrics.append(result)
        if args.model == "mlp" and epoch in args.rank_epochs:
            ranks.extend(rank_snapshot(model, probe, step, epoch, args.rank_tolerances))
        write_csv(directory / "metrics.csv", metrics)
        write_csv(directory / "rank.csv", ranks)
        write_json(directory / "diagnostics.json", diagnostics)
        print(f"{method} seed={seed} epoch={epoch}/{args.epochs} "
              f"train_loss={result['train_loss']:.4f} test_accuracy={100 * result['test_accuracy']:.2f}%", flush=True)
        if scheduler:
            scheduler.step()
    return {"model": args.model, "method": method, "seed": seed,
            "epochs": args.epochs, "steps": step, "train_samples": len(train_data),
            "test_samples": len(test_data), "width": args.width,
            "test_accuracy": metrics[-1]["test_accuracy"], "test_loss": metrics[-1]["test_loss"],
            "train_loss": metrics[-1]["train_loss"], "seconds": time.perf_counter() - start,
            "directory": str(directory)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--lr-decay-epoch", type=int, default=0,
                        help="0: constant appendix LR; 12 reproduces author notebook schedule")
    parser.add_argument("--lr-decay-gamma", type=float, default=0.7)
    parser.add_argument("--train-limit", type=int, default=0, help="0: all 60000 training samples")
    parser.add_argument("--test-limit", type=int, default=0, help="0: all 10000 test samples")
    parser.add_argument("--rank-epochs", nargs="*", type=int, default=[0, 1, 7, 14],
                        help="MLP snapshot epochs; an empty list disables rank measurements")
    parser.add_argument("--rank-tolerances", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4, 1e-5])
    parser.add_argument("--probe-size", type=int, default=1024,
                        help="activation rank <= min(probe-size, width); samples fixed across runs")
    parser.add_argument("--diagnostic-steps", nargs="+", type=int, default=[0, 1, 10, 100])
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--ff-width", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-dir", default="hw2/data")
    parser.add_argument("--output", default="hw2/results/paper")
    args = parser.parse_args()
    args.epochs = args.epochs if args.epochs is not None else (14 if args.model == "mlp" else 10)
    args.width = args.width if args.width is not None else (2048 if args.model == "mlp" else 64)
    args.lr = args.lr if args.lr is not None else (0.1 if args.model == "mlp" else 1e-3)
    args.optimizer = args.optimizer or ("sgd" if args.model == "mlp" else "adamw")
    args.methods = args.methods or (["zero_paper", "zero_author", "identity", "kaiming", "xavier", "zero_all"]
                                   if args.model == "mlp" else ["standard", "identity", "zero_inspired", "zero_inspired_author"])
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if min(args.epochs, args.width, args.batch_size, args.test_batch_size, args.probe_size) < 1:
        parser.error("epochs, dimensions, batch sizes, and probe-size must be positive")
    if any(epoch < 0 for epoch in args.rank_epochs) or any(tol <= 0 for tol in args.rank_tolerances):
        parser.error("rank epochs must be nonnegative and rank tolerances positive")
    allowed = ({"zero_paper", "zero_author", "identity", "kaiming", "xavier", "zero_all"}
               if args.model == "mlp" else {"standard", "identity", "zero_inspired", "zero_inspired_author"})
    if set(args.methods) - allowed:
        parser.error(f"methods for {args.model} must be in {sorted(allowed)}")
    return args


def main():
    import torchvision
    args = parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Existing results in {output}; choose a new --output directory")
    torch.set_num_threads(args.threads)
    train_data, test_data = load_data(args)
    metadata_path = output.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata[output.name] = {
        "config": vars(args), "train_samples": len(train_data), "test_samples": len(test_data),
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__},
        "device": args.device, "device_name": (torch.cuda.get_device_name(args.device)
                                               if args.device == "cuda" else platform.machine()),
        "deterministic_algorithms": True, "tf32": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, metadata)
    summaries = []
    for method in args.methods:
        for seed in args.seeds:
            summaries.append(run(args, method, seed, train_data, test_data))
            write_csv(Path(args.output) / "summary.csv", summaries)


if __name__ == "__main__":
    main()
