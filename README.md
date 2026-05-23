# Phononic Frequency Comb Reservoir Computer

Code accompanying the paper. Implements a reservoir computer whose physical
substrate is the coupled-mode phononic system from Qi et al. (2020), operating
in the parametric-resonance regime where a frequency comb naturally forms.

## References

- **Qi et al.** "Nonlinear phononic frequency comb via parametric resonance",
  *APL* **117**, 183503 (2020) — coupled-mode model, Eqs. (5)–(6) and Fig. 3
  parameters.
- **Shaabani Shishavan et al.** "Reservoir computing with frequency-comb
  dynamics", *Phys. Rev. Research* **7**, L042008 (2025) — RC framework.

---

## Physics

The reservoir is governed by two nonlinearly coupled phonon modes:

```
∂w₁/∂s = −iF − (1 + iΔ₁)w₁ + iw₂²         (Eq. 5)
∂w₂/∂s = −(γ₂₁ + iΔ₂)w₂ + 2iw₁w₂*         (Eq. 6)
```

where `s = γ₁ t` is normalised time, `w₁, w₂` are complex mode amplitudes,
`F` is the drive amplitude, `Δ₁, Δ₂` are detunings, and `γ₂₁ = γ₂/γ₁`.

When the detuning condition `2Δ₁Δ₂ ≤ −(1 + Δ₁² + 2γ₂₁)` is satisfied and
`F` is above the Arnold-tongue threshold, the fixed point undergoes a Hopf
bifurcation and the mode envelope oscillates periodically — forming a
frequency comb. The comb lines provide a high-dimensional feature space for
reservoir computing.

Default parameters (Qi 2020 Fig. 3): `Δ₁ = 4`, `κ = −9`, `Δ₂ = Δ₁/2 + κ`,
`γ₂₁ = 1`, `F = 20`, physical mode at `ω₁/2π = 3.86 MHz`, `Q₁ = 1000`.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All model and RC hyperparameters |
| `phononic_solver.py` | ODE integrator for Eqs. (5)–(6); extracts virtual-node and FFT features |
| `reservoir.py` | End-to-end RC pipeline: encoding → reservoir → Ridge regression |
| `chaotic_systems.py` | Mackey-Glass, Lorenz, and Rössler time-series generators |
| `metrics.py` | NMSE, MASE, Pearson correlation |
| `comb_numeric.py` | Analytical (Qi 2020) and IVP-based comb existence checks |
| `main.py` | CLI: single-task prediction, heatmap sweeps, combined figures |
| `sweep.py` | Multi-panel (a–d) parameter sweep with live heatmap updates |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. All numerics use standard scientific Python; no GPU
required.

---

## Reproducing Paper Figures

### Single-task prediction (Mackey-Glass, Lorenz, Rössler)

```bash
# Default parameters — Mackey-Glass
python main.py --task mackey_glass --combine --outdir results/combined-comb

# All three tasks in one combined figure
python main.py --combine --outdir results/combined-comb
```

The `--combine` flag produces a 5-panel figure:
- (a) ψ₂ time-domain power oscillation
- (b) ψ₂ frequency spectrum (comb teeth)
- (c–e) RC test predictions for Mackey-Glass, Rössler, Lorenz

### NMSE vs. drive parameters (heatmap)

```bash
# F_AVG × Δ₁ heatmap at a single data rate
python main.py --task mackey_glass --sweep_f_delta1 \
    --favg_min 0 --favg_max 50 --favg_n 50 \
    --delta1_min -8 --delta1_max 8 --delta1_n 40 \
    --data_rate 2400 --outdir results/mackey-glass

# Multi-panel sweep (panels a–d: unmodulated, modulated, F×Δf, Δ₁×Δf)
python sweep.py --jobs 2000 --panel all --task mackey_glass
```

### γ₂₁ sweep

```bash
python main.py --task mackey_glass --sweep_gamma21 \
    --gamma21_vals 0.2 0.6 1.0 1.4 1.8 2.2 \
    --data_rate 2400 --outdir results/mackey-glass/gamma21-sweep
```

### Data-rate sweep

```bash
python main.py --task mackey_glass --sweep_data_rate \
    --data_rates '[1000,5000,10000,50000,100000]' \
    --outdir results/mackey-glass/data-rates
```

---

## Key Configuration (`config.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `DELTA1` | 4.0 | Drive detuning Δ₁ |
| `KAPPA` | −9.0 | Offset κ; sets Δ₂ = Δ₁/2 + κ |
| `GAMMA21` | 1.0 | Damping ratio γ₂/γ₁ |
| `F_AVG` | 20.0 | Mean drive amplitude |
| `DELTA_F` | 0.0 | Drive modulation depth |
| `N_VIRTUAL` | 512 | Virtual nodes per symbol |
| `N_FFT_FEATURES` | 512 | FFT bins per mode |
| `SYMBOL_DURATION_TAU` | 5.0 | Symbol length in normalised time |
| `WARMUP_SYMBOLS` | 200 | Transient symbols to discard |
| `REG_LAMBDA` | 1e-3 | Ridge regularisation λ |
| `N_SYMBOLS` | 4000 | Total time-series length |

Override Δ₁ and F at runtime:

```bash
python main.py --task lorenz --delta1_val 5.0 --favg_val 20.0 --combine \
    --outdir results/lorenz-d1-5
```

---

## Feature Extraction

For each input symbol, the solver integrates Eqs. (5)–(6) over one symbol
period `T_symbol = γ₁/data_rate` and samples the state at `N_virtual`
equally-spaced times. The feature vector concatenates:

- `|ψ₁(tₖ)|²` for k = 1…N_virtual  (mode-1 power time trace)
- `|ψ₂(tₖ)|²` for k = 1…N_virtual  (mode-2 power time trace)
- `|FFT(ψ₁)|[0:N_fft]`             (comb line magnitudes, mode 1)
- `|FFT(ψ₂)|[0:N_fft]`             (comb line magnitudes, mode 2)

Features are log-compressed (`log₁₀(|·| + ε)`) and standardised before
Ridge regression.

---

## Input Encoding

The input time series `s(n) ∈ [0, 1]` is mapped to drive amplitude:

- `δf = 0` (default): `F(n) = F_avg · (0.5 + s(n))` — multiplicative, drive
  always positive.
- `δf > 0`: `F(n) = F_avg + δf · (s(n) − 0.5)` — additive modulation.

---

## Comb Existence (`comb_numeric.py`)

`comb_qi_analytical(f, Δ₁, Δ₂, γ₂₁)` checks four necessary conditions from
Qi 2020:
1. Drive exceeds Arnold-tongue minimum (`f ≥ f_min_arnold`)
2. Eq. (7) quadratic has a positive root (`|w₂P|² > 0`)
3. Detuning boundary: `2Δ₁Δ₂ ≤ −(1 + Δ₁² + 2γ₂₁)`
4. Eq. (10) lower bound on `|w₂P|²` satisfied

`evaluate_comb_at_point(...)` additionally integrates the ODEs and checks
whether `|ψ₂|²` oscillates with modulation index > 1% (numerical pulsing),
used to overlay the RC NMSE heatmaps with the comb existence region.
