from mqb_toy.physics.ops import a_op, adag_op, x_op, p_op
import numpy as np
import pytest

def test_commutator_xp_approx():
    N=12
    x = x_op(N)
    p = p_op(N)
    comm = x@p - p@x
    vac = np.zeros((N,), dtype=complex)
    vac[0] = 1.0
    val = vac.conj() @ comm @ vac
    assert np.real(val) == pytest.approx(0.0, abs=1e-8)
    assert np.imag(val) == pytest.approx(1.0, rel=1e-2)
