import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, expm

from .physics.ham import ham_total
from .physics.lindblad import evolve_lindblad
from .physics.ops import (
    a_op,
    adag_op,
    number_op,
    p_op,
    x_op,
)


@dataclass
class SimulationResult:
    times: np.ndarray
    populations: np.ndarray  # shape (T, 2)
    x_expect: np.ndarray
    p_expect: np.ndarray
    energy: np.ndarray
    oscillator_n: np.ndarray
    purity: np.ndarray
    qudit_coherence: np.ndarray
    metadata: Dict[str, float]
    density_traj: np.ndarray  # (T, dim, dim)
    state_traj: Optional[np.ndarray] = None  # (T, dim)


def initial_state(N: int) -> np.ndarray:
    e0 = np.array([1.0, 0.0], dtype=complex)
    osc0 = np.zeros(N, dtype=complex)
    osc0[0] = 1.0
    return np.kron(osc0, e0)


def evolve_unitary(H: np.ndarray, psi0: np.ndarray, dt: float, steps: int) -> np.ndarray:
    U = expm(-1j * H * dt)
    psi = psi0.copy()
    traj: List[np.ndarray] = [psi0.copy()]
    for _ in range(steps):
        psi = U @ psi
        psi = psi / np.linalg.norm(psi)
        traj.append(psi.copy())
    return np.array(traj)


def states_to_density(traj: np.ndarray) -> np.ndarray:
    return np.array([np.outer(psi, psi.conj()) for psi in traj])


def partial_trace_qudit(rho: np.ndarray, N: int) -> np.ndarray:
    reshaped = rho.reshape(N, 2, N, 2)
    return np.einsum("iajb->jb", reshaped)


def compute_observables(
    H: np.ndarray,
    rho_traj: np.ndarray,
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    P0 = np.kron(np.eye(N), np.array([[1, 0], [0, 0]], dtype=complex))
    P1 = np.kron(np.eye(N), np.array([[0, 0], [0, 1]], dtype=complex))
    x = np.kron(x_op(N), np.eye(2))
    p = np.kron(p_op(N), np.eye(2))
    n_op = np.kron(number_op(N), np.eye(2))

    pops = []
    x_vals = []
    p_vals = []
    energy = []
    n_vals = []
    purity = []
    coherence = []

    for rho in rho_traj:
        pop0 = np.real(np.trace(rho @ P0))
        pop1 = np.real(np.trace(rho @ P1))
        pops.append([pop0, pop1])
        x_vals.append(np.real(np.trace(rho @ x)))
        p_vals.append(np.real(np.trace(rho @ p)))
        energy.append(np.real(np.trace(rho @ H)))
        n_vals.append(np.real(np.trace(rho @ n_op)))
        purity.append(np.real(np.trace(rho @ rho)))
        rho_qudit = partial_trace_qudit(rho, N)
        coherence.append(np.abs(rho_qudit[0, 1]))

    return (
        np.array(pops),
        np.array(x_vals),
        np.array(p_vals),
        np.array(energy),
        np.array(n_vals),
        np.array(purity),
        np.array(coherence),
    )


def build_jump_ops(
    N: int,
    gamma_dephase: float,
    gamma_relax: float,
    gamma_damp: float,
    gamma_heat: float,
) -> List[Tuple[float, np.ndarray]]:
    ops: List[Tuple[float, np.ndarray]] = []
    eye_osc = np.eye(N)

    if gamma_dephase > 0.0:
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        ops.append((gamma_dephase, np.kron(eye_osc, sz)))

    if gamma_relax > 0.0:
        sm = np.array([[0, 0], [1, 0]], dtype=complex)
        ops.append((gamma_relax, np.kron(eye_osc, sm)))

    if gamma_damp > 0.0:
        a = a_op(N)
        ops.append((gamma_damp, np.kron(a, np.eye(2))))

    if gamma_heat > 0.0:
        adag = adag_op(N)
        ops.append((gamma_heat, np.kron(adag, np.eye(2))))

    return ops


def save_figures(out_dir: str, times: np.ndarray, result: SimulationResult) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(times, result.populations[:, 0], label="pop |0⟩")
    ax.plot(times, result.populations[:, 1], label="pop |1⟩")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend(loc="best")
    fig.savefig(os.path.join(out_dir, "populations.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig2 = plt.figure(figsize=(8, 5))
    ax2 = plt.gca()
    ax2.plot(times, result.x_expect, label="<x>")
    ax2.plot(times, result.p_expect, label="<p>")
    ax2.set_xlabel("time")
    ax2.set_ylabel("oscillator quadrature")
    ax2.legend(loc="best")
    fig2.savefig(os.path.join(out_dir, "quadratures.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    fig3 = plt.figure(figsize=(8, 5))
    ax3 = plt.gca()
    ax3.plot(times, result.purity, label="purity Tr(ρ²)")
    ax3.plot(times, result.qudit_coherence, label="|ρ_ge|")
    ax3.set_xlabel("time")
    ax3.legend(loc="best")
    ax3.set_ylabel("coherence metrics")
    fig3.savefig(os.path.join(out_dir, "coherence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig3)


def save_spectrum(out_dir: str, H: np.ndarray) -> None:
    evals, _ = eigh(H)
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ax.plot(evals[: min(80, len(evals))], marker="o", linestyle="None")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    fig.savefig(os.path.join(out_dir, "spectrum.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def persist_observables(out_dir: str, result: SimulationResult) -> None:
    np.savez_compressed(
        os.path.join(out_dir, "observables.npz"),
        times=result.times,
        populations=result.populations,
        x_expect=result.x_expect,
        p_expect=result.p_expect,
        energy=result.energy,
        oscillator_n=result.oscillator_n,
        purity=result.purity,
        qudit_coherence=result.qudit_coherence,
    )

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(result.metadata, f, indent=2)


def maybe_save_trajectories(out_dir: str, result: SimulationResult, save_states: bool) -> None:
    traj_path = os.path.join(out_dir, "trajectories.npz")
    if save_states and result.state_traj is not None:
        np.savez_compressed(
            traj_path,
            density=result.density_traj,
            states=result.state_traj,
        )
    else:
        np.savez_compressed(traj_path, density=result.density_traj)


def simulate(args: argparse.Namespace) -> SimulationResult:
    N, om, g, Delta = args.N, args.omega, args.g, args.Delta
    H = ham_total(N, om, g, Delta)
    dim = H.shape[0]

    psi0 = initial_state(N)
    jumps = build_jump_ops(N, args.gamma_dephase, args.gamma_relax, args.gamma_damp, args.gamma_heat)
    noise_enabled = len(jumps) > 0

    if noise_enabled:
        rho0 = np.outer(psi0, psi0.conj())
        rho_traj = evolve_lindblad(H, rho0, args.dt, args.steps, jumps)
        state_traj = None
    else:
        state_traj = evolve_unitary(H, psi0, args.dt, args.steps)
        rho_traj = states_to_density(state_traj)

    (
        pops,
        x_vals,
        p_vals,
        energy,
        n_vals,
        purity,
        coherence,
    ) = compute_observables(H, rho_traj, N)

    metadata = {
        "N": N,
        "omega": om,
        "g": g,
        "Delta": Delta,
        "steps": args.steps,
        "dt": args.dt,
        "dim": dim,
        "gamma_dephase": args.gamma_dephase,
        "gamma_relax": args.gamma_relax,
        "gamma_damp": args.gamma_damp,
        "gamma_heat": args.gamma_heat,
        "noise_enabled": noise_enabled,
    }

    times = np.arange(rho_traj.shape[0]) * args.dt
    return SimulationResult(
        times=times,
        populations=pops,
        x_expect=x_vals,
        p_expect=p_vals,
        energy=energy,
        oscillator_n=n_vals,
        purity=purity,
        qudit_coherence=coherence,
        metadata=metadata,
        density_traj=rho_traj,
        state_traj=state_traj,
    )


def run_model(
    N: int = 20,
    omega: float = 1.0,
    g: float = 0.2,
    Delta: float = 0.1,
    steps: int = 300,
    dt: float = 0.05,
    gamma_dephase: float = 0.0,
    gamma_relax: float = 0.0,
    gamma_damp: float = 0.0,
    gamma_heat: float = 0.0,
) -> SimulationResult:
    args = argparse.Namespace(
        N=N,
        omega=omega,
        g=g,
        Delta=Delta,
        steps=steps,
        dt=dt,
        gamma_dephase=gamma_dephase,
        gamma_relax=gamma_relax,
        gamma_damp=gamma_damp,
        gamma_heat=gamma_heat,
    )
    return simulate(args)


def write_summary(out_dir: str, H: np.ndarray, result: SimulationResult) -> None:
    lines = [
        f"N={result.metadata['N']}, omega={result.metadata['omega']}, g={result.metadata['g']}, Delta={result.metadata['Delta']}",
        f"steps={result.metadata['steps']}, dt={result.metadata['dt']}, dim={result.metadata['dim']}",
        "Noise parameters:",
        f"  gamma_dephase={result.metadata['gamma_dephase']}",
        f"  gamma_relax={result.metadata['gamma_relax']}",
        f"  gamma_damp={result.metadata['gamma_damp']}",
        f"  gamma_heat={result.metadata['gamma_heat']}",
        f"H Hermitian? {np.allclose(H, H.conj().T)}",
        f"Final purity={result.purity[-1]:.6f}",
        f"Final coherence={result.qudit_coherence[-1]:.6f}",
        f"Final oscillator n={result.oscillator_n[-1]:.6f}",
    ]
    if result.state_traj is not None:
        lines.append(f"Final state norm={np.linalg.norm(result.state_traj[-1]):.6f}")
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid qudit-boson toy model aligned with MQB experiments.")
    ap.add_argument("--N", type=int, default=20, help="Oscillator truncation dimension.")
    ap.add_argument("--omega", type=float, default=1.0, help="Oscillator frequency.")
    ap.add_argument("--g", type=float, default=0.2, help="Coupling strength to σ_z.")
    ap.add_argument("--Delta", type=float, default=0.1, help="Qudit transverse driving.")
    ap.add_argument("--steps", type=int, default=300, help="Number of simulation steps.")
    ap.add_argument("--dt", type=float, default=0.05, help="Time increment.")
    ap.add_argument("--gamma-dephase", dest="gamma_dephase", type=float, default=0.0, help="Qudit σ_z dephasing rate.")
    ap.add_argument("--gamma-relax", dest="gamma_relax", type=float, default=0.0, help="Qudit relaxation rate (σ-).")
    ap.add_argument("--gamma-damp", dest="gamma_damp", type=float, default=0.0, help="Oscillator damping rate (a).")
    ap.add_argument("--gamma-heat", dest="gamma_heat", type=float, default=0.0, help="Oscillator heating rate (a†).")
    ap.add_argument("--save-trajectories", action="store_true", help="Persist state/density trajectories for analysis.")
    ap.add_argument("--out", required=True, help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    result = simulate(args)
    H = ham_total(args.N, args.omega, args.g, args.Delta)

    save_figures(args.out, result.times, result)
    save_spectrum(args.out, H)
    persist_observables(args.out, result)
    maybe_save_trajectories(args.out, result, args.save_trajectories)
    write_summary(args.out, H, result)

    print(f"Done. Outputs saved to {args.out}")


if __name__ == "__main__":
    main()
