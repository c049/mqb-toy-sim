import numpy as np

def a_op(N):
    a = np.zeros((N,N), dtype=complex)
    for n in range(1, N):
        a[n-1, n] = np.sqrt(n)
    return a

def adag_op(N):
    return a_op(N).conj().T

def x_op(N):
    a = a_op(N)
    adag = a.conj().T
    return (a + adag) / np.sqrt(2.0)

def p_op(N):
    a = a_op(N)
    adag = a.conj().T
    return 1j*(adag - a)/np.sqrt(2.0)

def number_op(N):
    a = a_op(N)
    adag = a.conj().T
    return adag @ a
