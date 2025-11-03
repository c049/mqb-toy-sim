import numpy as np

import pytest

from mqb_toy.analysis.metrics import (
    aic,
    bic,
    fidelity_overlap,
    lead_time_gain,
    mean_absolute_error,
    mape,
    purity_loss,
    residual_rms,
)


def test_scalar_metrics_basic():
    ref = np.array([0.1, 0.2, 0.3])
    pred = np.array([0.1, 0.25, 0.28])
    assert residual_rms(ref, pred) > 0
    assert mean_absolute_error(ref, pred) > 0
    assert mape(ref, pred) > 0


def test_aic_bic_monotonic():
    residuals = np.array([0.1, -0.05, 0.02])
    assert aic(residuals, num_params=2) < bic(residuals, num_params=5)


def test_fidelity_and_purity_loss():
    rho_ref = np.array([[1, 0], [0, 0]], dtype=complex)
    rho_candidate = np.array([[0.9, 0], [0, 0.1]], dtype=complex)
    fid = fidelity_overlap(rho_ref, rho_candidate)
    assert 0 <= fid <= 1
    trace = [1.0, 0.95, 0.9]
    assert purity_loss(trace) == pytest.approx(0.1, abs=1e-8)


def test_lead_time_gain_positive_when_surrogate_faster():
    times = np.linspace(0, 1, 6)
    base = np.array([0.1, 0.2, 0.3, 0.55, 0.7, 0.9])
    surrogate = np.array([0.1, 0.35, 0.5, 0.7, 0.85, 0.95])
    gain = lead_time_gain(times, base, surrogate, threshold=0.5)
    assert gain > 0
