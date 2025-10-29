import numpy as np
from .ops import number_op, x_op

def kron(A,B):
    return np.kron(A,B)

def ham_osc(N, omega):
    n = number_op(N)
    # H_osc = omega*(n + 1/2)
    return omega*(n + 0.5*np.eye(N))

def ham_total(N, omega=1.0, g=0.2, Delta=0.1):
    # Oscillator ⊗ identity_2
    Hosc = kron(ham_osc(N, omega), np.eye(2))
    x = x_op(N)
    # Pauli matrices (electronic space)
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    # Couplings
    Hc1 = kron(x, sz) * g
    Hc2 = kron(np.eye(N), sx) * Delta
    return Hosc + Hc1 + Hc2
