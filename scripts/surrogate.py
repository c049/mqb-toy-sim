"""Gaussian-process surrogate modeling for MQB toy simulations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from mqb_toy.analysis.metrics import mean_absolute_error
from mqb_toy.analysis.sweeps import grid_scan


def load_table(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("Empty dataset.")
    fields = rows[0].keys()
    data_columns: Dict[str, List[float]] = {field: [] for field in fields}
    for row in rows:
        for field in fields:
            data_columns[field].append(float(row[field]))
    arrays = {k: np.asarray(v, dtype=float) for k, v in data_columns.items()}
    X = np.column_stack([arrays["g"], arrays["Delta"]])
    return X, arrays


def fit_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    noise: float,
) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=(0.2, 0.2), nu=2.5) + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-6, 1e-1))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gpr.fit(X, y)
    return gpr


def learning_curve(
    X: np.ndarray,
    y: np.ndarray,
    n_points: int = 6,
    seed: int = 0,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    splits = np.linspace(max(5, len(X) // n_points), len(X), n_points, dtype=int)
    curves: List[Dict[str, float]] = []
    for size in splits:
        train_idx = order[:size]
        test_idx = order[size:]
        if test_idx.size == 0 or train_idx.size < 3:
            continue
        model = fit_surrogate(X[train_idx], y[train_idx], noise=1e-4)
        pred = model.predict(X[test_idx])
        curves.append({"train_size": float(size), "mae": float(mean_absolute_error(y[test_idx], pred))})
    return curves


def save_learning_curve(path: str, curve: List[Dict[str, float]]) -> None:
    if not curve:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["train_size", "mae"])
        writer.writeheader()
        for row in curve:
            writer.writerow(row)


def plot_surface(
    g_values: Sequence[float],
    delta_values: Sequence[float],
    predictions: np.ndarray,
    target: str,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    G, D = np.meshgrid(g_values, delta_values, indexing="ij")
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    c = ax.contourf(G, D, predictions, levels=20, cmap="viridis")
    ax.set_xlabel("g")
    ax.set_ylabel("Delta")
    ax.set_title(f"Surrogate prediction of {target}")
    fig.colorbar(c, ax=ax, label=target)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GP surrogate to emulate MQB toy simulations.")
    ap.add_argument("--data", help="CSV from sweeps.grid_scan(). If omitted, a fresh grid is generated.")
    ap.add_argument("--target", default="purity_final", help="Metric column to model.")
    ap.add_argument("--g-range", nargs=3, type=float, metavar=("min", "max", "num"), default=[0.05, 0.4, 12])
    ap.add_argument("--delta-range", nargs=3, type=float, metavar=("min", "max", "num"), default=[0.05, 0.4, 12])
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--noise", type=float, default=1e-3, help="Observation noise level for GP.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.data:
        X, arrays = load_table(args.data)
    else:
        g_values = np.linspace(args.g_range[0], args.g_range[1], int(args.g_range[2]))
        delta_values = np.linspace(args.delta_range[0], args.delta_range[1], int(args.delta_range[2]))
        sweep = grid_scan(
            g_values=g_values,
            delta_values=delta_values,
            base_config={"N": 14, "steps": 180, "dt": 0.04},
            save_path=os.path.join(args.out, "grid_generated.csv"),
        )
        arrays = {key: np.asarray([row[key] for row in sweep], dtype=float) for key in sweep[0].keys()}
        X = np.column_stack([arrays["g"], arrays["Delta"]])

    if args.target not in arrays:
        raise ValueError(f"Target metric '{args.target}' not found in dataset.")

    y = arrays[args.target]
    surrogate = fit_surrogate(X, y, noise=args.noise)
    y_pred = surrogate.predict(X)
    mae_value = mean_absolute_error(y, y_pred)

    report = {
        "target": args.target,
        "n_points": int(X.shape[0]),
        "mae_train": float(mae_value),
        "kernel": str(surrogate.kernel_),
    }
    with open(os.path.join(args.out, "surrogate_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    curve = learning_curve(X, y)
    save_learning_curve(os.path.join(args.out, "surrogate_learning_curve.csv"), curve)

    g_grid = np.linspace(args.g_range[0], args.g_range[1], 50)
    delta_grid = np.linspace(args.delta_range[0], args.delta_range[1], 50)
    GG, DD = np.meshgrid(g_grid, delta_grid, indexing="ij")
    grid_points = np.column_stack([GG.ravel(), DD.ravel()])
    predictions = surrogate.predict(grid_points).reshape(GG.shape)
    plot_surface(g_grid, delta_grid, predictions, args.target, os.path.join(args.out, "surrogate_surface.png"))

    if curve:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([row["train_size"] for row in curve], [row["mae"] for row in curve], marker="o")
            ax.set_xlabel("training size")
            ax.set_ylabel("MAE")
            ax.set_title("Surrogate accuracy vs samples")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "surrogate_learning_curve.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            pass

    print("Surrogate model saved to", args.out)


if __name__ == "__main__":
    main()
