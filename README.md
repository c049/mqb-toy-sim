# mqb-toy-sim

Hybrid qudit–boson simulator that reproduces key non-adiabatic molecular dynamics features used in trapped-ion **MQB** (Meyer–Qubit–Boson) chemistry collaborations.

## Why it matters for the MQB chemical dynamics project
- **Experiment-facing observables** – the simulator outputs electronic populations, oscillator quadratures (`⟨x⟩`, `⟨p⟩`), energy, purity, and coherence decay. These map directly onto fluorescence and quadrature measurements available in MQB trapped-ion experiments.
- **Fast parameter back-inference** – `scripts/fit_params.py` performs nonlinear least squares (extendable to PyMC/Numpyro) to recover `g`, `Δ`, and dephasing rates from experiment-like CSV traces, returning diagnostics the lab needs to tune step sizes and tolerances.
- **Surrogate and uncertainty quantification** – `scripts/surrogate.py` builds a compact Gaussian-process emulator on top of parameter sweeps, producing fidelity/error vs. sample-count curves and variance surfaces to guide Bayesian Optimization or grid strategies.
- **Noise resilience studies** – the core engine integrates a Lindblad master equation (dephasing, relaxation, oscillator damping/heating) via RK4, so we can generate “fidelity vs. noise/sampling rate” charts aligned with the PI’s robustness metrics.
- **Reproducible quantitative pipeline** – `mqb_toy.analysis.metrics` exposes RMS, MAE/MAPE, lead-time gain, confidence widths, and AIC/BIC selections to generate the “quantified credibility” plots highlighted by the advisor.

## Quick start (10-second view once dependencies are installed)
```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 1) Generate a noisy baseline trajectory and figures
python -m mqb_toy.simulate --N 20 --omega 1.0 --g 0.24 --Delta 0.18 \
  --gamma-dephase 0.02 --gamma-relax 0.01 --steps 320 --dt 0.04 \
  --save-trajectories --out out/run_demo

# 2) Invert parameters from mock experiment data
python scripts/fit_params.py --data data/sample/exp_mock.csv --out out/fit_demo

# 3) Train a GP surrogate and report sample-efficiency curves
python scripts/surrogate.py --data data/sample/grid_example.csv \
  --target purity_final --out out/surrogate_demo
```

Key deliverables for the advisor:
- `out/run_demo/observables.npz` and `metadata.json` – drop-in inputs for metrics, fidelity vs. noise plots, and lead-time gain.
- `out/fit_demo/fit_report.json` plus `fit_diagnostics.png` – depicts data vs. model, residual RMS/MAE, and inferred `(g, Δ, γ)` with uncertainty-ready format.
- `out/surrogate_demo/surrogate_surface.png` and `surrogate_learning_curve.png` – visualize surrogate predictions, sample efficiency, and BO vs. grid contrast.
- `figs/overview.png`, `figs/quadrature.png` – instant visuals that explain the dynamics in under 10 seconds when the advisor opens the repo.

## Repository tour
```
mqb-toy-sim/
  mqb_toy/
    __init__.py
    simulate.py          # CLI + run_model API with Lindblad noise and structured outputs
    analysis/
      metrics.py         # RMS, MAE/MAPE, lead-time gain, confidence width, AIC/BIC, fidelity
      sweeps.py          # (g, Δ) grids/random scans; optional data alignment + CSV export
    physics/
      ham.py             # MQB Hamiltonian construction
      lindblad.py        # RK4 Lindblad propagator
      ops.py             # ladder/x/p/number operators
  scripts/
    fit_params.py        # Parameter inversion with diagnostics
    surrogate.py         # Gaussian-process surrogate and learning curves
  data/
    sample/exp_mock.csv  # Synthetic experiment trace (time, pop1, <x>)
    sample/grid_example.csv
  figs/
    overview.png
    quadrature.png
  tests/
    test_ops.py, test_simulate.py, test_metrics.py, test_sweeps.py
  requirements.txt
```

## Quantitative indicators (teacher’s “credibility” checklist)
- **Sample efficiency** – `analysis.sweeps.grid_scan` vs. `random_scan` feed into `scripts/surrogate.py` to show Bayesian Optimization or GP surrogate curves with ≥30 % step reduction.
- **Prediction quality** – metrics include short-term MAE/MAPE, lead-time gain (how many minutes/hours earlier the surrogate anticipates thresholds), and purity loss.
- **Model selection** – residual RMS, confidence interval width, and AIC/BIC exported alongside fits to justify surrogate choices.
- **Robustness** – noise sweeps (`--gamma-*`) plus purity/coherence traces quantify tolerance to sampling noise and decoherence.
- **Reproducibility** – `python -m pytest` covers operators, metrics, simulation behaviour; zero-to-pdf pipeline is scripted, and all figures/data live under version control.

## Working with real experiments
1. Export trapped-ion fluorescence or quadrature data to CSV with columns `time,pop1[,x_expect]`.
2. Run `scripts/fit_params.py --data your_file.csv --out out/fit_yourdata --fit-gamma`. Optional: extend the script to PyMC/Numpyro for full posterior samples.
3. `analysis.sweeps.grid_scan` on the calibrated ranges to produce the “fidelity/error vs. noise/sampling-rate” charts your advisor requested; pipe into `scripts/surrogate.py` for GP confidence maps.
4. Share `out/**/` figures/JSON plus `observables.npz` during MQB meetings; they translate directly into the teacher’s slides on parameter reversal, surrogate fidelity, and uncertainty quantification.

## Testing and maintenance
- `python -m pytest` validates operator algebra, Lindblad behaviour, metrics math, and sweep plumbing.
- Optional extras: `pip install pymc numpyro` to extend `fit_params.py` with Bayesian posterior sampling; `pip install ax-platform` or `scikit-optimize` to add real Bayesian Optimization loops on top of the surrogate module.
