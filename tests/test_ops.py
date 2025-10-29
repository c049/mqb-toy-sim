from mqb_toy.physics.ops import a_op, adag_op, x_op, p_op
import numpy as np

def test_commutator_xp_approx():
    N=12
    x = x_op(N)
    p = p_op(N)
    comm = x@p - p@x
    # In truncated space, [x,p] ≈ iI for low-lying subspace
    trace_imag = np.imag(np.trace(comm))/N
    assert trace_imag > 0.5  # loose check
