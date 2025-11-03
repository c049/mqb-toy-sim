import numpy as np


def _lindblad_rhs(
    H: np.ndarray,
    rho: np.ndarray,
    jumps: list,
) -> np.ndarray:
    comm = H @ rho - rho @ H
    drho = -1j * comm

    for gamma, c in jumps:
        cd = c.conj().T
        cdc = cd @ c
        drho += gamma * (c @ rho @ cd - 0.5 * (cdc @ rho + rho @ cdc))

    return drho


def _rk4_step(H: np.ndarray, rho: np.ndarray, dt: float, jumps: list) -> np.ndarray:
    k1 = _lindblad_rhs(H, rho, jumps)
    k2 = _lindblad_rhs(H, rho + 0.5 * dt * k1, jumps)
    k3 = _lindblad_rhs(H, rho + 0.5 * dt * k2, jumps)
    k4 = _lindblad_rhs(H, rho + dt * k3, jumps)
    return rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def evolve_lindblad(
    H: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    steps: int,
    jumps: list,
) -> np.ndarray:
    rho = rho0.copy()
    traj = [rho.copy()]
    for _ in range(steps):
        rho = _rk4_step(H, rho, dt, jumps)
        rho = 0.5 * (rho + rho.conj().T)
        trace = np.real(np.trace(rho))
        if trace > 0:
            rho /= trace
        traj.append(rho.copy())
    return np.array(traj)
