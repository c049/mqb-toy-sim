"""Parameter sweep utilities for MQB toy simulations."""

from __future__ import annotations

import csv
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..simulate import SimulationResult, run_model
from .metrics import mean_absolute_error, mape, residual_rms

MetricFunc = Callable[[SimulationResult], float]


def _default_metrics() -> Dict[str, MetricFunc]:
    return {
        "purity_final": lambda res: float(res.purity[-1]),
        "coherence_final": lambda res: float(res.qudit_coherence[-1]),
        "pop1_mean": lambda res: float(np.mean(res.populations[:, 1])),
        "energy_span": lambda res: float(np.max(res.energy) - np.min(res.energy)),
    }


def _write_csv(path: str, rows: List[Dict[str, float]], field_order: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def grid_scan(
    g_values: Sequence[float],
    delta_values: Sequence[float],
    base_config: Optional[Dict[str, float]] = None,
    metric_funcs: Optional[Dict[str, MetricFunc]] = None,
    target_observables: Optional[Dict[str, np.ndarray]] = None,
    save_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Evaluate a grid of (g, Delta) pairs and compute metrics."""
    base_config = dict(base_config or {})
    metric_funcs = dict(metric_funcs or _default_metrics())
    results: List[Dict[str, float]] = []

    for g in g_values:
        for delta in delta_values:
            cfg = {**base_config, "g": g, "Delta": delta}
            sim_result = run_model(**cfg)
            row: Dict[str, float] = {"g": g, "Delta": delta}

            for name, func in metric_funcs.items():
                row[name] = func(sim_result)

            if target_observables:
                if "x_expect" in target_observables:
                    ref = target_observables["x_expect"]
                    row["rms_x"] = residual_rms(ref, sim_result.x_expect[: len(ref)])
                    row["mae_x"] = mean_absolute_error(ref, sim_result.x_expect[: len(ref)])
                    row["mape_x"] = mape(ref, sim_result.x_expect[: len(ref)])
                if "pop1" in target_observables:
                    ref = target_observables["pop1"]
                    row["rms_pop1"] = residual_rms(ref, sim_result.populations[: len(ref), 1])

            results.append(row)

    if save_path:
        fieldnames = list(results[0].keys()) if results else ["g", "Delta"]
        _write_csv(save_path, results, fieldnames)

    return results


def random_scan(
    num_samples: int,
    g_range: Tuple[float, float],
    delta_range: Tuple[float, float],
    base_config: Optional[Dict[str, float]] = None,
    metric_funcs: Optional[Dict[str, MetricFunc]] = None,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Randomly sample parameter space; used to contrast with Bayesian or grid methods."""
    rng = np.random.default_rng(seed)
    base_config = dict(base_config or {})
    metric_funcs = dict(metric_funcs or _default_metrics())

    rows: List[Dict[str, float]] = []
    for _ in range(num_samples):
        g = float(rng.uniform(*g_range))
        delta = float(rng.uniform(*delta_range))
        cfg = {**base_config, "g": g, "Delta": delta}
        sim_result = run_model(**cfg)
        row: Dict[str, float] = {"g": g, "Delta": delta}
        for name, func in metric_funcs.items():
            row[name] = func(sim_result)
        rows.append(row)

    if save_path:
        fieldnames = list(rows[0].keys()) if rows else ["g", "Delta"]
        _write_csv(save_path, rows, fieldnames)

    return rows
