# Results

Pre-computed results from the phononic frequency-comb reservoir computer.
Default parameters throughout: Delta1=4, kappa=-9, Delta2=Delta1/2+kappa=-7,
gamma21=1, F_AVG=20 (Qi 2020 Fig. 3), ridge regression, unless noted otherwise.

---

## Structure

```
results/
├── combined/                   5-panel figures across four operating regimes
│   ├── regime_comb/            Frequency-comb regime (paper main result)
│   ├── regime_bistable/        Bistable regime
│   ├── regime_chaotic_comb/    Chaotic-comb regime
│   ├── regime_param_res/       Parametric-resonance regime
│   ├── multi_panel.png         Full multi-panel paper figure
│   └── multi_panel.pdf
├── f_avg_delta1_sweep/         F_AVG x Delta1 NMSE heatmaps (ridge)
│   ├── mackey_glass/
│   ├── lorenz/
│   └── rossler/
├── datarate_sweep/             NMSE vs physical data rate (Mackey-Glass)
│   ├── summary.png/.csv
│   ├── nmse_vs_datarate.csv
│   └── per_rate/               Per-rate F_AVG x Delta1 grids
├── gamma21_sweep/              Effect of damping ratio gamma21
│   ├── range_0.2-2.2/          Fine sweep (paper figure)
│   └── range_0.01-100/         Wide sweep
├── kappa_sweep/                Effect of kappa on comb regime and NMSE
│   ├── summary.png/.csv
│   └── per_kappa/
└── lasso_comparison/           Lasso regression vs Ridge comparison
```

---

## `combined/`

Each regime subfolder contains the same six files:

| File | Description |
|------|-------------|
| `combined.png` | 5-panel figure: (a) psi2 time domain, (b) psi2 spectrum, (c-e) RC predictions |
| `time_domain.png` | psi2 power oscillation vs normalised time tau |
| `frequency_spectrum.png` | FFT of psi2 showing comb teeth |
| `mackey_glass_prediction.png` | Test set prediction for Mackey-Glass (tau=17) |
| `lorenz_prediction.png` | Test set prediction for Lorenz x-component |
| `rossler_prediction.png` | Test set prediction for Rossler x-component |

`regime_comb` is the main paper result. The other three show the same RC
pipeline under different dynamical regimes of the coupled-mode system.

Reproduced by:
```bash
python main.py --combine --outdir results/combined/regime_comb
```

---

## `f_avg_delta1_sweep/`

NMSE heatmaps over the (F_AVG, Delta1) parameter space.
Each cell is log10(NMSE) for single-step prediction.
Delta2 = Delta1/2 + kappa throughout.

### `mackey_glass/`

| File | Description |
|------|-------------|
| `ridge_dr500.{csv,png}` | Heatmap at 500 Sa/s |
| `ridge_dr100.{csv,png}` | Heatmap at 100 Sa/s |
| `panel_a_delta_f0.csv` | sweep.py panel (a): delta_f=0, multiplicative encoding f(n)=F_avg*(0.5+s(n)) |
| `panel_b_delta_f5.csv` | sweep.py panel (b): additive modulation delta_f=5 |
| `panel_b_delta_f10.csv` | sweep.py panel (b): additive modulation delta_f=10 |
| `panel_c_favg_vs_deltaf.csv` | sweep.py panel (c): F_AVG vs delta_f at fixed Delta1 |
| `panel_d_deltaf_vs_delta1.csv` | sweep.py panel (d): delta_f vs Delta1 at fixed F_AVG |

CSV columns for ridge_dr* files: `f_avg, delta1, log10_nmse`

CSV columns for panel_a/b/c/d files: `F_AVG, DELTA1, DELTA2, NMSE, log10_NMSE, COMB_QI, COMB_PULSING, COMB_MOD_RATIO`
- `COMB_QI`: 1 if all Qi 2020 analytical comb conditions are satisfied
- `COMB_PULSING`: 1 if IVP shows pulsing (std/mean > 0.01 and mean |w2|^2 > 1e-4)
- `COMB_MOD_RATIO`: std(|w2|^2) / mean(|w2|^2)

### `lorenz/` and `rossler/`

| File | Description |
|------|-------------|
| `ridge.{csv,png}` | F_AVG x Delta1 heatmap, single data rate |
| `ridge_sweep.{csv,png}` | Same sweep from a separate run |

Reproduced by:
```bash
python main.py --task mackey_glass --sweep_f_delta1 \
    --favg_min 0 --favg_max 50 --favg_n 50 \
    --delta1_min -8 --delta1_max 8 --delta1_n 40 \
    --data_rate 500 --outdir results/f_avg_delta1_sweep/mackey_glass
```

---

## `datarate_sweep/`

NMSE vs physical data rate for Mackey-Glass prediction.

| File | Description |
|------|-------------|
| `summary.png` | Multi-panel F_AVG x Delta1 heatmap at each data rate |
| `summary.csv` | Combined table: `data_rate_Sa, f_avg, delta1, log10_nmse` |
| `nmse_vs_datarate.csv` | Summary: log10(NMSE) at optimal params vs data rate |
| `per_rate/dr_{N}.csv` | F_AVG x Delta1 grid at N Sa/s; columns: `favg_\ delta1, {delta1_vals...}` |

Data rates: 2000-20000 Sa/s (step 1000) and 96000-104000 Sa/s.

Reproduced by:
```bash
python main.py --task mackey_glass --sweep_data_rate \
    --data_rates '[2000,5000,10000,50000,100000]' \
    --outdir results/datarate_sweep
```

---

## `gamma21_sweep/`

Effect of the damping ratio gamma21 = gamma2/gamma1 on RC performance.
All runs: Mackey-Glass task, same F_AVG x Delta1 grid.

### `range_0.2-2.2/`  (used for paper figure)

| File | Description |
|------|-------------|
| `nmse_heatmap.png` | F_AVG x Delta1 heatmap for each gamma21 in {0.2, 0.6, 1.0, 1.4, 1.8, 2.2} |
| `nmse_and_comb.png` | NMSE + Qi analytical comb overlay, 3x3 grid |
| `comb_existence.png` | Analytical comb existence region (Qi 2020 conditions only) |
| `data.csv` | Raw data: `gamma21, f_avg, delta1, log10_nmse` |

### `range_0.01-100/`

| File | Description |
|------|-------------|
| `nmse_heatmap.png` | Heatmaps for gamma21 in {0.01, 0.1, 1, 10, 100} |
| `nmse_and_comb.png` | NMSE + comb overlay |
| `data.csv` | Raw data |

Reproduced by:
```bash
python main.py --task mackey_glass --sweep_gamma21 \
    --gamma21_vals 0.2 0.6 1.0 1.4 1.8 2.2 \
    --data_rate 2400 --outdir results/gamma21_sweep/range_0.2-2.2
```

---

## `kappa_sweep/`

Effect of kappa (which sets Delta2 = Delta1/2 + kappa) on the comb region and NMSE.
Kappa values: -9 (paper default), -4, 0, 4, 9.

| File | Description |
|------|-------------|
| `summary.png` | F_AVG x Delta1 heatmap for each kappa |
| `summary.csv` | Combined: `KAPPA, F_AVG, DELTA1, DELTA2, NMSE, log10_NMSE, COMB_QI, ...` |
| `per_kappa/kappa_{val}.csv` | Per-kappa F_AVG x Delta1 grid |

Reproduced via:
```bash
python sweep.py --panel a --kappa '-9,-4,0,4,9' --jobs 2000
```

---

## `lasso_comparison/`

Lasso regression output layer compared against Ridge (same RC pipeline, same parameters).

| File | Description |
|------|-------------|
| `f_avg_delta1.{csv,png}` | F_AVG x Delta1 heatmap with Lasso at default data rate |
| `dr_1000.{csv,png}` | Lasso heatmap at 1000 Sa/s |
| `dr_10000.{csv,png}` | Lasso heatmap at 10000 Sa/s |
| `dr_100000.{csv,png}` | Lasso heatmap at 100000 Sa/s |

Reproduced by setting `REGRESSION = "lasso"` in `config.py` then running `main.py --sweep_f_delta1`.
