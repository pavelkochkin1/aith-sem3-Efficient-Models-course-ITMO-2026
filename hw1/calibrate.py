"""Калибровка на CPU: сначала latency, затем energy; validation только для оценки."""

import csv
import hashlib
import json
import platform
from importlib.metadata import version
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, memory


def observations(rows, field, validation):
    """OOM и отсутствующие измерения не заменяются нулями."""
    selected = [row for row in rows
                if row["status"] == "ok"
                and row["is_validation"] == str(validation)
                and row[field] != ""]
    sizes = np.array([float(row["S"]) for row in selected])
    batches = np.array([float(row["B"]) for row in selected])
    measured = np.array([float(row[field]) for row in selected])
    if not np.all(np.isfinite(measured) & (measured > 0)):
        raise ValueError(f"Некорректные измерения {field}: нужны конечные положительные числа.")
    return sizes, batches, measured


def conditioning(matrix):
    """Зависимость столбцов после нормировки: единицы не влияют на результат."""
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.where(norms > 0, norms, 1.0)
    condition = float(np.linalg.cond(normalized))
    return {
        "column_normalized_rank": int(np.linalg.matrix_rank(normalized)),
        "column_normalized_condition_number": condition if np.isfinite(condition) else None,
    }


def fit_latency(sizes, batches, measured):
    if len(measured) < 3:
        raise ValueError("Для калибровки latency нужны хотя бы три точки.")

    def unpack(parameters):
        # Подбираем обратные скорости, масштабируя переменные для оптимизатора.
        return {"t0": float(parameters[0] * 1e-3),
                "C": float(1e12 / parameters[1]),
                "beta": float(1e11 / parameters[2])}

    def residuals(parameters):
        return (latency(sizes, batches, unpack(parameters)) - measured) / measured

    solutions = []
    for inverse_compute, inverse_bandwidth in product((0.1, 1.0, 10.0), repeat=2):
        solution = least_squares(
            residuals, [0.5 * measured.min() / 1e-3, inverse_compute, inverse_bandwidth],
            bounds=([0.0, 1e-12, 1e-12], np.inf), x_scale="jac",
            ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=2000,
        )
        if solution.success:
            solutions.append(solution)
    if not solutions:
        raise RuntimeError("Оптимизатор latency не сошёлся ни из одной начальной точки.")
    best = min(solutions, key=lambda result: result.cost)
    diagnostics = {
        "n": len(measured), "method": "scipy.optimize.least_squares",
        "starts": 9, "converged_starts": len(solutions),
        "mean_squared_relative_error": float(np.mean(best.fun**2)),
        "termination": best.message,
        **conditioning(best.jac),
    }
    return unpack(best.x), diagnostics


def fit_energy(sizes, batches, measured, theta):
    if len(measured) < 3:
        raise ValueError("Для калибровки energy нужны хотя бы три точки.")
    # Используем ПРЕДСКАЗАННОЕ время с уже зафиксированной theta.
    design = np.column_stack((latency(sizes, batches, theta),
                              flops(sizes, batches), bytes_moved(sizes, batches)))
    weighted = design / measured[:, None]
    scales = np.linalg.norm(weighted, axis=0)
    coefficients, residual_norm = nnls(weighted / scales, np.ones(len(measured)))
    coefficients /= scales
    parameters = {"latency": dict(theta), **dict(zip(("P0", "eps_F", "eps_Q"),
                                                    map(float, coefficients)))}
    diagnostics = {
        "n": len(measured), "method": "scipy.optimize.nnls",
        "mean_squared_relative_error": float(residual_norm**2 / len(measured)),
        "zero_coefficients": [name for name in ("P0", "eps_F", "eps_Q")
                              if parameters[name] == 0.0],
        **conditioning(weighted),
    }
    return parameters, diagnostics


def fit_parameters(rows):
    """Обе функции fit получают только calibration; validation сюда не попадает."""
    theta, latency_info = fit_latency(*observations(rows, "latency_s", False))
    theta_energy, energy_info = fit_energy(*observations(rows, "energy_j", False), theta)
    return {"latency": theta, "energy": theta_energy,
            "fit_diagnostics": {"latency": latency_info, "energy": energy_info}}


def error_metrics(measured, predicted):
    if len(measured) == 0:
        return {"n": 0}
    relative = (predicted - measured) / measured
    absolute = np.abs(relative)
    return {
        "n": len(measured),
        "mape_percent": float(100 * np.mean(absolute)),
        "median_ape_percent": float(100 * np.median(absolute)),
        "p95_ape_percent": float(100 * np.percentile(absolute, 95)),
        "relative_rmse_percent": float(100 * np.sqrt(np.mean(relative**2))),
        "rmse": float(np.sqrt(np.mean((predicted - measured)**2))),
    }


def evaluate(rows, parameters):
    metrics = {}
    for field, unit, predict in (
        ("latency_s", "s", lambda s, b: latency(s, b, parameters["latency"])),
        ("energy_j", "J", lambda s, b: energy(s, b, parameters["energy"])),
        ("memory_bytes", "byte", memory),
    ):
        metrics[field] = {"rmse_unit": unit}
        for split, validation in (("calibration", False), ("validation", True)):
            sizes, batches, measured = observations(rows, field, validation)
            metrics[field][split] = error_metrics(measured, predict(sizes, batches))
    return metrics


def write_predictions(path, rows, parameters):
    predictions = []
    for row in rows:
        size, batch = float(row["S"]), float(row["B"])
        predicted = {
            "latency_s": float(latency(size, batch, parameters["latency"])),
            "energy_j": float(energy(size, batch, parameters["energy"])),
            "memory_bytes": float(memory(size, batch)),
        }
        result = {**row, "flops_predicted": float(flops(size, batch))}
        for field, value in predicted.items():
            result[f"{field}_predicted"] = value
            result[f"{field}_ape_percent"] = (
                100 * abs(value / float(row[field]) - 1)
                if row["status"] == "ok" and row[field] else ""
            )
        predictions.append(result)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


def main():
    source = Path(__file__).resolve().parent
    results = source / "results"
    measurements = results / "measurements.csv"
    with measurements.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    parameters = fit_parameters(rows)
    parameters["units"] = {"t0": "s", "C": "FLOP/s", "beta": "byte/s",
                           "P0": "W", "eps_F": "J/FLOP", "eps_Q": "J/byte"}
    parameters["method"] = {
        "objective": "mean squared relative error; equal weight per configuration",
        "split": "is_validation from measurements.csv; no fitting on validation",
        "constraints": "t0>=0, C>0, beta>0, P0>=0, eps_F>=0, eps_Q>=0",
        "latency_variables": "[t0/1e-3, 1e12/C, 1e11/beta]; lower bounds [0, 1e-12, 1e-12]",
        "energy_time": "predicted latency with frozen calibrated parameters",
    }
    parameters["provenance"] = {
        "python": platform.python_version(), "numpy": version("numpy"), "scipy": version("scipy"),
        "sha256": {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (measurements, results / "environment.json", source / "equations.py",
                                source / "calibrate.py", source / "requirements-calibration.txt")},
    }
    metrics = evaluate(rows, parameters)
    for name, data in (("theta.json", parameters), ("calibration_metrics.json", metrics)):
        (results / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
                                    + "\n", encoding="utf-8")
    write_predictions(results / "predictions.csv", rows, parameters)
    print(json.dumps({"latency": parameters["latency"], "energy": parameters["energy"]}, indent=2))
    for field, splits in metrics.items():
        for split in ("calibration", "validation"):
            values = splits[split]
            print(f"{field} {split}: n={values['n']}, MAPE={values.get('mape_percent', float('nan')):.2f}%")
    print(f"Saved: {results / 'theta.json'}, calibration_metrics.json, predictions.csv")


if __name__ == "__main__":
    main()
