# mqb-toy-sim

A **hybrid qudit–bosonic toy model** that mimics features of molecular non-adiabatic dynamics used in trapped-ion **MQB** simulations.

## What it demonstrates
- A truncated harmonic oscillator (bosonic mode, dimension `N`) coupled to a two-level "electronic" system (qudit).
- Hamiltonian: `H = H_osc + g * x ⊗ σ_z + Δ * I ⊗ σ_x`, showing an avoided crossing–like behaviour.
- Unitary time evolution, populations vs time, and oscillator quadrature snapshots.

> This is **not** a full chemistry model; it’s a compact, didactic demo that aligns with MQB mapping ideas and runs in seconds.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m mqb_toy.simulate --N 24 --omega 1.0 --g 0.25 --Delta 0.2 --steps 400 --dt 0.05 --out out
```

Outputs:
- `populations.png` – electronic populations over time
- `x_expect.png` – oscillator ⟨x⟩ vs time
- `spectrum.png` – eigen-spectrum sample
- `summary.txt` – parameters and checks

## Files
```
mqb-toy-sim/
  mqb_toy/
    __init__.py
    simulate.py       # CLI entry
    physics/
      ops.py          # ladder operators, x, p
      ham.py          # Hamiltonian builder
  tests/
    test_ops.py
  out/                # generated figures
  requirements.txt
  README.md
```

## Extensions (ideas)
- Add dissipation/dephasing via Lindblad (requires scipy.linalg expm for Liouvillians).
- Add multi-mode oscillator; scan `g, Δ` to show resource/fidelity tradeoffs.
- Export wavefunction snapshots for comparison with experiment.
