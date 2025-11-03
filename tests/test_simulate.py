import numpy as np

from mqb_toy.simulate import run_model


def test_run_model_shapes():
    result = run_model(N=6, g=0.15, Delta=0.12, steps=40, dt=0.05)
    assert result.populations.shape[1] == 2
    assert result.x_expect.shape[0] == result.times.shape[0]
    assert np.isclose(result.purity[0], 1.0, atol=1e-6)


def test_noise_reduces_purity():
    clean = run_model(N=6, g=0.18, Delta=0.14, steps=30, dt=0.04)
    noisy = run_model(N=6, g=0.18, Delta=0.14, steps=30, dt=0.04, gamma_dephase=0.05)
    assert noisy.purity[-1] < clean.purity[-1]
