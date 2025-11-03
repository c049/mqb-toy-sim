"""Quantitative metrics used to benchmark MQB toy simulations against experiment."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import sqrtm


def residual_rms(reference: Sequence[float], prediction: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    diff = ref - pred
    return float(np.sqrt(np.mean(diff**2)))


def mean_absolute_error(reference: Sequence[float], prediction: Sequence[float]) -> float:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    return float(np.mean(np.abs(ref - pred)))


def mape(reference: Sequence[float], prediction: Sequence[float], eps: float = 1e-9) -> float:
    ref = np.asarray(reference, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    safe = np.where(np.abs(ref) < eps, eps, np.abs(ref))
    return float(np.mean(np.abs((ref - pred) / safe)))


def fidelity_overlap(rho_reference: np.ndarray, rho_candidate: np.ndarray) -> float:
    """Uhlmann fidelity for mixed states."""
    sqrt_ref = sqrtm(rho_reference)
    mid = sqrt_ref @ rho_candidate @ sqrt_ref
    sqrt_mid = sqrtm(mid)
    fidelity = np.trace(sqrt_mid)
    return float(np.real(fidelity) ** 2)


def purity_loss(purity_trace: Sequence[float]) -> float:
    purity = np.asarray(purity_trace, dtype=float)
    if purity.size == 0:
        return 0.0
    return float(1.0 - purity[-1])


def lead_time_gain(
    times: Sequence[float],
    baseline_series: Sequence[float],
    surrogate_series: Sequence[float],
    threshold: float,
) -> float:
    """Compute lead-time improvement (positive -> surrogate predicts earlier)."""
    t = np.asarray(times, dtype=float)
    base = np.asarray(baseline_series, dtype=float)
    surrogate = np.asarray(surrogate_series, dtype=float)

    def _first_cross(series: np.ndarray) -> float:
        idx = np.where(series >= threshold)[0]
        return float(t[idx[0]]) if idx.size > 0 else float("nan")

    tb = _first_cross(base)
    ts = _first_cross(surrogate)
    if np.isnan(tb) or np.isnan(ts):
        return float("nan")
    return tb - ts


def confidence_interval_width(samples: Iterable[float], alpha: float = 0.95) -> float:
    data = np.asarray(list(samples), dtype=float)
    if data.size == 0:
        return 0.0
    lower = 0.5 * (1 - alpha)
    upper = 1 - lower
    q_low, q_high = np.quantile(data, [lower, upper])
    return float(q_high - q_low)


def aic(residuals: Sequence[float], num_params: int) -> float:
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    if n == 0:
        return float("nan")
    rss = np.sum(residuals**2)
    if rss <= 0:
        rss = 1e-12
    return float(n * np.log(rss / n) + 2 * num_params)


def bic(residuals: Sequence[float], num_params: int) -> float:
    residuals = np.asarray(residuals, dtype=float)
    n = residuals.size
    if n == 0:
        return float("nan")
    rss = np.sum(residuals**2)
    if rss <= 0:
        rss = 1e-12
    return float(n * np.log(rss / n) + num_params * np.log(n))
