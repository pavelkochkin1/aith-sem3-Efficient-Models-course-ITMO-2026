"""Regenerate figures and aggregates from saved, completed experiment logs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LABELS = {
    "zero_paper": "ZerO: статья", "zero_author": "ZerO: код",
    "identity": "Identity", "kaiming": "Kaiming", "xavier": "Xavier",
    "zero_all": "Все нули", "standard": "Xavier",
    "zero_inspired": "ZerO-inspired: статья",
    "zero_inspired_author": "ZerO-inspired: код",
}
COLORS = {key: value for key, value in zip(LABELS, ["#168a77", "#2858bc", "#c78027", "#bc486e", "#7663b7", "#777777", "#7663b7", "#168a77", "#2858bc"])}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def collect(group):
    runs = read_csv(RESULTS / group / "summary.csv")
    config = json.loads((RESULTS / "metadata.json").read_text())[group]["config"]
    expected = {(method, str(seed)) for method in config["methods"] for seed in config["seeds"]}
    assert len(runs) == len(expected), f"{group}: не все запуски завершены"
    assert {(run["method"], run["seed"]) for run in runs} == expected, group
    for run in runs:
        directory = RESULTS / group / Path(run["directory"]).name
        metrics = read_csv(directory / "metrics.csv")
        assert [int(row["epoch"]) for row in metrics] == list(range(config["epochs"] + 1)), directory
        for metric in ("test_accuracy", "test_loss", "train_loss"):
            assert float(metrics[-1][metric]) == float(run[metric]), (directory, metric)
    return runs


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(float(row["test_accuracy"]) * 100)
    return {method: {"n": len(values), "accuracy_mean_pct": float(np.mean(values)),
                     "accuracy_std_pct": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                     "accuracies_pct": values}
            for method, values in groups.items()}


def curves(ax, rows, group, file, metric, *, matrix=None, scale=1, min_epoch=0):
    methods = list(dict.fromkeys(row["method"] for row in rows))
    for method in methods:
        points = defaultdict(list)
        for run in (row for row in rows if row["method"] == method):
            directory = RESULTS / group / Path(run["directory"]).name
            for row in read_csv(directory / file):
                if int(row["epoch"]) < min_epoch:
                    continue
                if matrix and row["matrix"] != matrix:
                    continue
                if row.get(metric) not in ("", None):
                    points[int(row["epoch"])].append(float(row[metric]) * scale)
        epochs = sorted(points)
        mean = [np.mean(points[epoch]) for epoch in epochs]
        low = [min(points[epoch]) for epoch in epochs]
        high = [max(points[epoch]) for epoch in epochs]
        ax.plot(epochs, mean, "o-", ms=4, lw=2, color=COLORS[method], label=LABELS[method])
        ax.fill_between(epochs, low, high, alpha=0.09, color=COLORS[method])
    ax.set_xlabel("Эпоха")
    ax.grid(alpha=0.2)


def main():
    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    paper, controls, transformer = collect("paper"), collect("controls"), collect("transformer")
    aggregates = {"paper": aggregate(paper), "controls": aggregate(controls),
                  "transformer": aggregate(transformer), "spread_definition": "sample std, ddof=1; bands: min/max over seeds"}
    if paper:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        curves(axes[0], paper, "paper", "metrics.csv", "test_accuracy", scale=100, min_epoch=1)
        axes[0].set_ylabel("Test accuracy, %")
        axes[0].set_title("Полная MNIST: точность")
        curves(axes[1], paper, "paper", "metrics.csv", "train_loss")
        axes[1].set_ylabel("Train cross-entropy")
        axes[1].set_title("Обучение на 60 000 изображений")
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / "mnist_learning.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        curves(axes[0], paper, "paper", "rank.csv", "rank_tol_0.0001", matrix="W2_minus_I")
        # Control logs are in a separate directory; draw them without relocating raw data.
        if controls:
            curves(axes[0], [row for row in controls if row["method"] == "identity"], "controls", "rank.csv", "rank_tol_0.0001", matrix="W2_minus_I")
        axes[0].axhline(784, color="black", ls="--", lw=1, label="Размер входа: 784")
        axes[0].set_ylabel("Численный ранг (порог 10⁻⁴)")
        axes[0].set_title("Изменения весов W₂ − I")
        axes[0].legend(fontsize=8)
        curves(axes[1], paper, "paper", "rank.csv", "rank_tol_0.0001", matrix="hidden_activations")
        if controls:
            curves(axes[1], [row for row in controls if row["method"] == "identity"], "controls", "rank.csv", "rank_tol_0.0001", matrix="hidden_activations")
        axes[1].axhline(784, color="black", ls="--", lw=1)
        axes[1].set_ylabel("Ранг признаков на 1 024 примерах")
        axes[1].set_title("Нелинейные признаки; максимум 1 024")
        fig.tight_layout()
        fig.savefig(figures / "mnist_rank.png")
        plt.close(fig)

        rank_end = defaultdict(lambda: defaultdict(list))
        for group, rows in (("paper", paper), ("controls", controls)):
            for run in rows:
                ranks = read_csv(RESULTS / group / Path(run["directory"]).name / "rank.csv")
                final = [row for row in ranks if row["matrix"] == "W2_minus_I" and int(row["epoch"]) == int(run["epochs"])]
                for row in final:
                    for metric in ("rank_tol_0.01", "rank_tol_0.001", "rank_tol_0.0001", "rank_tol_1e-05", "stable_rank"):
                        rank_end[run["method"]][metric].append(float(row[metric]))
        aggregates["final_rank"] = {method: {metric: {"mean": float(np.mean(values)), "values": values}
                                           for metric, values in metrics.items()} for method, metrics in rank_end.items()}
        selected = [method for method in ("zero_paper", "zero_author", "identity") if method in rank_end]
        fig, ax = plt.subplots(figsize=(7, 4.1))
        for method in selected:
            means = [np.mean(rank_end[method][key]) for key in ("rank_tol_0.01", "rank_tol_0.001", "rank_tol_0.0001", "rank_tol_1e-05")]
            ax.plot([0, 1, 2, 3], means, "o-", color=COLORS[method], label=LABELS[method])
        ax.axhline(784, color="black", ls="--", lw=1, label="Размер входа: 784")
        ax.set_xticks([0, 1, 2, 3], ["10⁻²", "10⁻³", "10⁻⁴", "10⁻⁵"])
        ax.set_xlabel("Абсолютный порог сингулярного числа")
        ax.set_ylabel("Ранг W₂ − I после 14 эпох")
        ax.set_title("Чувствительность вывода к порогу ранга")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / "rank_tolerance.png")
        plt.close(fig)
    if transformer:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        curves(axes[0], transformer, "transformer", "metrics.csv", "test_accuracy", scale=100, min_epoch=1)
        axes[0].set_ylabel("Test accuracy, %")
        axes[0].set_title("Собственный опыт: patch Transformer / MNIST")
        curves(axes[1], transformer, "transformer", "metrics.csv", "train_loss")
        axes[1].set_ylabel("Train cross-entropy")
        axes[1].set_title("ReLU, 2 блока, ширина 64")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "transformer_learning.png")
        plt.close(fig)
    print(f"Saved {figures}; runs: MLP={len(paper)}, controls={len(controls)}, Transformer={len(transformer)}")
    return aggregates


if __name__ == "__main__":
    main()
