import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh
from .physics.ham import ham_total

def evolve(H, psi0, dt, steps):
    U = expm(-1j*H*dt)
    psi = psi0.copy()
    traj = [psi0]
    for _ in range(steps):
        psi = U @ psi
        # renormalize to avoid numeric drift
        psi = psi/np.linalg.norm(psi)
        traj.append(psi)
    return np.array(traj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--g", type=float, default=0.2)
    ap.add_argument("--Delta", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    N, om, g, De = args.N, args.omega, args.g, args.Delta
    H = ham_total(N, om, g, De)

    # Initial state: oscillator |0>, electronic |0>
    e0 = np.array([1,0], dtype=complex)
    osc0 = np.zeros(N, dtype=complex); osc0[0] = 1.0
    psi0 = np.kron(osc0, e0)

    traj = evolve(H, psi0, args.dt, args.steps)

    # Observables
    # Projectors on electronic |0>, |1>
    P0 = np.kron(np.eye(N), np.array([[1,0],[0,0]], dtype=complex))
    P1 = np.kron(np.eye(N), np.array([[0,0],[0,1]], dtype=complex))

    pops0 = [np.real(np.vdot(psi, P0@psi)) for psi in traj]
    pops1 = [np.real(np.vdot(psi, P1@psi)) for psi in traj]
    t = np.arange(len(traj))*args.dt

    # Plots
    fig = plt.figure(figsize=(8,6))
    ax = plt.gca()
    ax.plot(t, pops0, label='pop |0>')
    ax.plot(t, pops1, label='pop |1>')
    ax.set_xlabel('time')
    ax.set_ylabel('population')
    ax.legend(loc='best')
    fig.savefig(os.path.join(args.out, "populations.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Expectation of x (oscillator)
    # x ⊗ I
    from .physics.ops import x_op
    x = np.kron(x_op(N), np.eye(2))
    xexp = [np.real(np.vdot(psi, x@psi)) for psi in traj]
    fig2 = plt.figure(figsize=(8,6))
    ax2 = plt.gca()
    ax2.plot(t, xexp)
    ax2.set_xlabel('time')
    ax2.set_ylabel('<x>')
    fig2.savefig(os.path.join(args.out, "x_expect.png"), dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # Spectrum (first few eigenvalues)
    evals, _ = eigh(H)
    fig3 = plt.figure(figsize=(8,6))
    ax3 = plt.gca()
    ax3.plot(evals[:50], marker='o', linestyle='None')
    ax3.set_xlabel('index')
    ax3.set_ylabel('eigenvalue')
    fig3.savefig(os.path.join(args.out, "spectrum.png"), dpi=200, bbox_inches='tight')
    plt.close(fig3)

    # Summary
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"N={N}, omega={om}, g={g}, Delta={De}, steps={args.steps}, dt={args.dt}\n")
        f.write(f"H Hermitian? {np.allclose(H, H.conj().T)}\n")
        f.write(f"Final norm: {np.linalg.norm(traj[-1])}\n")

    print("Done. Outputs saved to", args.out)

if __name__ == "__main__":
    main()
