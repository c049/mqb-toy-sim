"""Parameter inversion helper for MQB toy simulations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from mqb_toy.analysis.metrics import mean_absolute_error, residual_rms
from mqb_toy.simulate import run_model


@dataclass
class Dataset:
    time: np.ndarray
    pop1: np.ndarray
    x_expect: Optional[np.ndarray]


def load_dataset(path: str) -> Dataset:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        times: List[float] = []
        pop1: List[float] = []
        x_expect: List[float] = []
        has_x = "x_expect" in reader.fieldnames if reader.fieldnames else False
        for row in reader:
            times.append(float(row["time"]))
            pop1.append(float(row["pop1"]))
            if has_x:
                x_expect.append(float(row["x_expect"]))
    time_arr = np.asarray(times, dtype=float)
    pop_arr = np.asarray(pop1, dtype=float)
    x_arr = np.asarray(x_expect, dtype=float) if x_expect else None
    return Dataset(time=time_arr, pop1=pop_arr, x_expect=x_arr)


def build_residual_function(
    data: Dataset,
    base_config: Dict[str, float],
    fit_gamma: bool,
) -> callable:
    steps = len(data.time) - 1
    dt = float(np.mean(np.diff(data.time))) if len(data.time) > 1 else base_config.get("dt", 0.05)

    def residuals(params: Sequence[float]) -> np.ndarray:
        g, Delta = params[0], params[1]
        cfg = dict(base_config)
        cfg.update({"g": g, "Delta": Delta, "steps": steps, "dt": dt})
        if fit_gamma:
            cfg["gamma_dephase"] = max(params[2], 0.0)
        sim = run_model(**cfg)

        sim_pop = sim.populations[: len(data.pop1), 1]
        diff_pop = sim_pop - data.pop1

        contributions = [diff_pop]

        if data.x_expect is not None:
            sim_x = sim.x_expect[: len(data.x_expect)]
            contributions.append(sim_x - data.x_expect)

        return np.concatenate(contributions)

    return residuals


def main() -> None:
    ap = argparse.ArgumentParser(description="Estimate g/Delta (and optionally dephasing) from experimental observables.")
    ap.add_argument("--data", required=True, help="CSV file with columns time,pop1[,x_expect].")
    ap.add_argument("--out", required=True, help="Output directory for reports and plots.")
    ap.add_argument("--N", type=int, default=16, help="Oscillator truncation for fitting.")
    ap.add_argument("--omega", type=float, default=1.0, help="Oscillator frequency used during fitting.")
    ap.add_argument("--g0", type=float, default=0.2, help="Initial guess for g.")
    ap.add_argument("--Delta0", type=float, default=0.1, help="Initial guess for Delta.")
    ap.add_argument("--gamma0", type=float, default=0.02, help="Initial guess for dephasing rate.")
    ap.add_argument("--fit-gamma", action="store_true", help="Also fit qudit dephasing.")
    ap.add_argument("--bounds", nargs=4, type=float, metavar=("g_min", "g_max", "Delta_min", "Delta_max"), default=[0.0, 0.5, 0.0, 0.5])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = load_dataset(args.data)

    base_cfg: Dict[str, float] = {
        "N": args.N,
        "omega": args.omega,
        "steps": len(dataset.time) - 1,
        "dt": float(np.mean(np.diff(dataset.time))) if len(dataset.time) > 1 else 0.05,
    }

    residual_fn = build_residual_function(dataset, base_cfg, args.fit_gamma)

    x0 = [args.g0, args.Delta0]
    bounds_lower = [args.bounds[0], args.bounds[2]]
    bounds_upper = [args.bounds[1], args.bounds[3]]
    if args.fit_gamma:
        x0.append(args.gamma0)
        bounds_lower.append(0.0)
        bounds_upper.append(max(args.gamma0 * 5, 0.5))

    result = least_squares(residual_fn, x0=x0, bounds=(bounds_lower, bounds_upper))
    fitted_params = result.x

    cfg_final = dict(base_cfg)
    cfg_final.update({"g": fitted_params[0], "Delta": fitted_params[1]})
    if args.fit_gamma:
        cfg_final["gamma_dephase"] = fitted_params[2]
    sim_final = run_model(**cfg_final)

    obs_pop = sim_final.populations[: len(dataset.pop1), 1]
    metrics = {
        "rms_pop1": residual_rms(dataset.pop1, obs_pop),
        "mae_pop1": mean_absolute_error(dataset.pop1, obs_pop),
    }
    if dataset.x_expect is not None:
        obs_x = sim_final.x_expect[: len(dataset.x_expect)]
        metrics.update(
            {
                "rms_x": residual_rms(dataset.x_expect, obs_x),
                "mae_x": mean_absolute_error(dataset.x_expect, obs_x),
            }
        )

    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "params": {
            "g": float(fitted_params[0]),
            "Delta": float(fitted_params[1]),
            "gamma_dephase": float(fitted_params[2]) if args.fit_gamma else 0.0,
        },
        "metrics": metrics,
        "iterations": int(result.nfev),
        "cost": float(result.cost),
    }

    with open(os.path.join(args.out, "fit_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    time = dataset.time
    np.savetxt(
        os.path.join(args.out, "fit_vs_data.csv"),
        np.column_stack(
            [
                time,
                dataset.pop1,
                obs_pop,
                dataset.x_expect if dataset.x_expect is not None else np.zeros_like(time),
                sim_final.x_expect[: len(time)],
            ]
        ),
        delimiter=",",
        header="time,pop1_data,pop1_model,x_data,x_model",
        comments="",
    )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2 if dataset.x_expect is not None else 1, 1, figsize=(8, 6), sharex=True)
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])

        ax[0].plot(time, dataset.pop1, "o", label="pop1 data")
        ax[0].plot(time, obs_pop, "-", label="pop1 model")
        ax[0].set_ylabel("Population |1⟩")
        ax[0].legend(loc="best")

        if dataset.x_expect is not None:
            ax[1].plot(time, dataset.x_expect, "o", label="<x> data")
            ax[1].plot(time, sim_final.x_expect[: len(time)], "-", label="<x> model")
            ax[1].set_ylabel("<x>")
            ax[1].legend(loc="best")

        ax[-1].set_xlabel("time")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fit_diagnostics.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass

    print("Fit complete. Report saved to", args.out)


if __name__ == "__main__":
    main()
