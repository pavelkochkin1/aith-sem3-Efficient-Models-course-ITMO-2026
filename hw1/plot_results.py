"""Графики и разбор сохранённого эксперимента; theta не подбирается повторно."""

import csv
import hashlib
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

from equations import energy, latency, memory


SOURCE = Path(__file__).resolve().parent
RESULTS = SOURCE / "results"
FIGURES = RESULTS / "figures"
METRICS = {
    "latency_s": ("Latency", "ms", 1000),
    "energy_j": ("Energy", "J / forward", 1),
    "memory_bytes": ("Peak memory", "MiB", 1 / 2**20),
}
SPLITS = ((False, "Calibration", "#2166AC", "o"),
          (True, "Validation", "#BB6B16", "s"))


def read_csv(name):
    with (RESULTS / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def predict(field, sizes, batches, parameters):
    if field == "latency_s":
        return latency(sizes, batches, parameters["latency"])
    if field == "energy_j":
        return energy(sizes, batches, parameters["energy"])
    return memory(sizes, batches)


def load_data(parameters):
    # Не смешиваем старые предсказания с новыми исходниками или измерениями.
    for name, expected in parameters["provenance"]["sha256"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Изменён {name}; сначала запусти calibrate.py.")
    rows = read_csv("predictions.csv")
    measured_rows = read_csv("measurements.csv")
    if [{key: row[key] for key in measured_rows[0]} for row in rows] != measured_rows:
        raise ValueError("predictions.csv не соответствует measurements.csv.")
    data = {key: np.array([int(row[key]) for row in rows]) for key in ("S", "B")}
    data["validation"] = np.array([row["is_validation"] == "True" for row in rows])
    for field in METRICS:
        data[field] = np.array([float(row[field]) if row["status"] == "ok" and row[field]
                               else np.nan for row in rows])
        data[field + "_predicted"] = predict(field, data["S"], data["B"], parameters)
        np.testing.assert_allclose(data[field + "_predicted"],
                                   [float(row[field + "_predicted"]) for row in rows])
    samples = defaultdict(list)
    for row in read_csv("latency_samples.csv"):
        samples[int(row["S"]), int(row["B"])].append(float(row["latency_s"]))
    quantiles = np.array([np.quantile(samples[s, b], [0.25, 0.5, 0.75])
                         if samples[s, b] else [np.nan] * 3
                         for s, b in zip(data["S"], data["B"])])
    np.testing.assert_allclose(quantiles[:, 1], data["latency_s"], equal_nan=True)
    data["latency_q25"], data["latency_q75"] = quantiles[:, 0], quantiles[:, 2]
    return data


def measured_points(ax, data, field, selected, x):
    scale = METRICS[field][2]
    for validation, label, color, marker in SPLITS:
        mask = selected & (data["validation"] == validation) & np.isfinite(data[field])
        y = data[field][mask] * scale
        spread = None
        if field == "latency_s":
            spread = np.array([data[field][mask] - data["latency_q25"][mask],
                               data["latency_q75"][mask] - data[field][mask]]) * scale
        ax.errorbar(x[mask], y, yerr=spread, fmt=marker, color=color, ms=4.5,
                    mfc="white" if validation else color, mew=1.2, capsize=2,
                    elinewidth=0.8, label=label, zorder=3)


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=180, facecolor="white")
    plt.close(fig)


def comparison_legend(fig):
    handles = [Line2D([], [], color="#303030", lw=1.6, label="Prediction (frozen model)")]
    handles += [Line2D([], [], color=c, marker=m, linestyle="none", ms=5,
                       markerfacecolor="white" if v else c, label=label + " measurements")
                for v, label, c, m in SPLITS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.054),
               ncol=3, frameon=False)


def grid_plot(data, field, parameters):
    title, unit, scale = METRICS[field]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharex=True, sharey=True)
    curve_batches = np.arange(data["B"].min(), data["B"].max() + 1)
    for ax, size in zip(axes.flat, sorted(set(data["S"]))):
        selected = data["S"] == size
        ax.plot(curve_batches, predict(field, size, curve_batches, parameters) * scale,
                color="#303030", lw=1.6)
        measured_points(ax, data, field, selected, data["B"])
        ax.set_title(f"S = {size} px", fontsize=12)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64, 256])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Batch B")
        ax.grid(which="major", color="#E3E6E9", lw=0.7)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{title} [{unit}]")
    axes.flat[-1].axis("off")
    note = "Latency bars: 25th–75th percentiles\nof 30 forwards. These are not\nconfidence intervals."
    if field != "latency_s":
        note = "No uncertainty interval from\nrepeated runs is available\nfor this metric."
    axes.flat[-1].text(0.02, 0.7, "Shared logarithmic axes\n11 sizes × 12 batches\n\n" + note,
                       transform=axes.flat[-1].transAxes, va="top", linespacing=1.6)
    fig.suptitle(f"{title}: measurements and analytical predictions", fontsize=17, y=0.98)
    comparison_legend(fig)
    fig.text(0.5, 0.025, "Tesla T4 · FP32 · 2026-09-25 · 63 calibration / 69 validation points",
             ha="center", color="#555555")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.92,
                        hspace=0.4, wspace=0.13)
    save(fig, field.split("_")[0] + "_grid.png")


def validation_plot(data):
    sizes, batches = sorted(set(data["S"])), sorted(set(data["B"]))
    cmap = LinearSegmentedColormap.from_list("signed_error", ["#2166AC", "#FFFFFF", "#BB6B16"])
    cmap.set_bad("#E4E7EB")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), width_ratios=[1, 1.55], layout="constrained")
    for (field, (title, unit, scale)), (parity, heatmap) in zip(METRICS.items(), axes):
        mask = data["validation"] & np.isfinite(data[field])
        actual, predicted = data[field][mask] * scale, data[field + "_predicted"][mask] * scale
        error = 100 * (predicted / actual - 1)
        limits = [min(actual.min(), predicted.min()) / 1.25,
                  max(actual.max(), predicted.max()) * 1.25]
        parity.plot(limits, limits, "--", color="#303030", lw=1.2, label="Perfect prediction")
        parity.scatter(actual, predicted, marker="s", facecolors="none", edgecolors="#BB6B16",
                       s=26, linewidths=1, label=f"Validation (n={mask.sum()})")
        parity.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
                   xlabel=f"Measured [{unit}]", ylabel=f"Predicted [{unit}]",
                   title=f"{title}: MAPE = {np.mean(np.abs(error)):.2f}%")
        parity.set_aspect("equal", adjustable="box")
        parity.grid(color="#E3E6E9", lw=0.7)
        parity.legend(loc="upper left", fontsize=9, frameon=False)
        matrix = np.full((len(sizes), len(batches)), np.nan)
        for s, b, value in zip(data["S"][mask], data["B"][mask], error):
            matrix[sizes.index(s), batches.index(b)] = value
        rendered = heatmap.imshow(matrix, cmap=cmap, vmin=-100, vmax=100, aspect="auto")
        heatmap.set(xticks=range(len(batches)), xticklabels=batches,
                    yticks=range(len(sizes)), yticklabels=sizes, xlabel="Batch B", ylabel="S [px]",
                    title="Signed relative error [%] · grey = not evaluated")
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                heatmap.text(j, i, f"{value:+.0f}", ha="center", va="center", fontsize=9,
                             color="white" if abs(value) > 60 else "#222222")
        fig.colorbar(rendered, ax=heatmap, ticks=[-100, -50, 0, 50, 100], shrink=0.85)
    fig.suptitle("Validation only · 100 × (prediction − measurement) / measurement\n"
                 "Logarithmic parity axes · heatmap cells are discrete configurations · Tesla T4 / FP32",
                 fontsize=15)
    save(fig, "validation_errors.png")


def transition_plot(data, parameters):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    selected = (data["S"] == 512) & (data["B"] >= 64) & (data["B"] <= 78)
    batches = np.arange(64, 79)
    for ax, (field, (title, unit, scale)) in zip(axes, METRICS.items()):
        ax.plot(batches, predict(field, 512, batches, parameters) * scale,
                color="#303030", lw=1.6)
        measured_points(ax, data, field, selected, data["B"])
        ax.set(xlabel="Batch B", ylabel=f"{title} [{unit}]", xticks=[64, 65, 70, 78])
        ax.set_ylim(bottom=0)
        ax.grid(color="#E3E6E9", lw=0.7)
    fig.suptitle("S = 512 px: measured configurations around a Conv7×7 kernel change",
                 fontsize=15, y=0.98)
    comparison_legend(fig)
    fig.text(0.5, 0.014, "B=65: scudnn kernel names; B=70: FFT + GEMM names. Exact switch point was not measured.\n"
             "Latency bars: observed IQR, not CI. Kernel names alone do not establish layer timing. Tesla T4 · FP32.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.87, bottom=0.25, wspace=0.27)
    save(fig, "kernel_transition.png")


def analysis_summary(data, parameters):
    summary = {"metrics": {}, "transition_S512_B65_to_B70": {}}
    for field in METRICS:
        error = 100 * (data[field + "_predicted"] / data[field] - 1)
        valid = data["validation"] & np.isfinite(error)
        worst = np.flatnonzero(valid)[np.argsort(-np.abs(error[valid]))[:5]]
        info = {"worst_validation": [
            {"S": int(data["S"][i]), "B": int(data["B"][i]),
             "measured": float(data[field][i]), "predicted": float(data[field + "_predicted"][i]),
             "signed_error_percent": float(error[i])} for i in worst]}
        for dimension in ("S", "B"):
            info["validation_by_" + dimension] = [
                {dimension: int(value), "n": int((valid & (data[dimension] == value)).sum()),
                 "mape_percent": float(np.mean(np.abs(error[valid & (data[dimension] == value)])))}
                for value in sorted(set(data[dimension][valid]))]
        summary["metrics"][field] = info
        indices = [np.flatnonzero((data["S"] == 512) & (data["B"] == b))[0] for b in (65, 70)]
        a, b = indices
        summary["transition_S512_B65_to_B70"][field] = {
            "measured": data[field][indices].tolist(),
            "predicted": data[field + "_predicted"][indices].tolist(),
            "measured_change_percent": float(100 * (data[field][b] / data[field][a] - 1)),
            "predicted_change_percent": float(100 * (data[field + "_predicted"][b]
                                                    / data[field + "_predicted"][a] - 1))}
    kernels = defaultdict(set)
    for row in read_csv("kernels.csv"):
        kernels[int(row["S"]), int(row["B"]), row["layer"]].add(row["kernel_name"])
    summary["kernel_changes_B65_to_B70"] = [
        {"S": int(s), "layer": layer,
         "removed": sorted(kernels[s, 65, layer] - kernels[s, 70, layer]),
         "added": sorted(kernels[s, 70, layer] - kernels[s, 65, layer])}
        for s in sorted(set(data["S"])) for layer in sorted({key[2] for key in kernels})
        if kernels[s, 65, layer] != kernels[s, 70, layer]]
    summary["roofline_threshold_FLOP_per_byte"] = parameters["latency"]["C"] / parameters["latency"]["beta"]
    sources = ["results/" + name for name in ("measurements.csv", "predictions.csv", "theta.json",
               "latency_samples.csv", "kernels.csv", "environment.json", "protocol.json")]
    summary["provenance"] = {
        "numpy": version("numpy"), "matplotlib": version("matplotlib"),
        "sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                   for name in sources + ["plot_results.py", "equations.py", "requirements-plots.txt"]}}
    (RESULTS / "analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                     allow_nan=False) + "\n", encoding="utf-8")


def main():
    parameters = json.loads((RESULTS / "theta.json").read_text())
    data = load_data(parameters)
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#303030", "text.color": "#303030"})
    for field in METRICS:
        grid_plot(data, field, parameters)
    validation_plot(data)
    transition_plot(data, parameters)
    analysis_summary(data, parameters)
    print(f"Verified {len(data['S'])} predictions and latency medians; saved 5 PNGs and analysis.json.")


if __name__ == "__main__":
    main()
