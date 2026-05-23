#!/usr/bin/env python3
"""
Main entry point for the Phononic Frequency Comb Reservoir Computer.

Uses the coupled-mode equations from Qi et al. (2020) as the reservoir
physics, with the reservoir computing framework adapted from
Shaabani Shishavan et al. (2025).

Usage:
    python main.py --task mackey_glass
    python main.py --task lorenz --n_symbols 3000
    python main.py --combine --outdir results/combined
"""

import argparse
import csv
from tqdm import tqdm as _tqdm
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def _get_max_workers():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(slurm_cpus) if slurm_cpus else os.cpu_count()

import config
from chaotic_systems import get_time_series
from reservoir import ReservoirComputer
from metrics import nmse, correlation
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", message="findfont:")
from datetime import datetime

from scipy.integrate import solve_ivp

plt.style.use("default")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

FONT_MULTIPLIER = 0.8

plt.rcParams.update({
    "font.size": 18 * FONT_MULTIPLIER,
    "axes.titlesize": 20 * FONT_MULTIPLIER,
    "axes.labelsize": 18 * FONT_MULTIPLIER,
    "xtick.labelsize": 16 * FONT_MULTIPLIER,
    "ytick.labelsize": 16 * FONT_MULTIPLIER,
    "legend.fontsize": 16 * FONT_MULTIPLIER,
})

SIZE_RED = 0.8


def _solve_comb_steady(f_val, d1, d2, g21, t_max=500, n_pts=50000, n_transient=10000):
    """Integrate the coupled-mode ODEs at constant drive and return steady-state traces."""
    lin1 = -(1.0 + 1j * d1)
    lin2 = -(g21 + 1j * d2)

    def rhs(tau, y):
        psi1 = y[0] + 1j * y[1]
        psi2 = y[2] + 1j * y[3]
        dp1 = -1j * f_val + lin1 * psi1 + 1j * psi2**2
        dp2 = lin2 * psi2 + 2j * psi1 * np.conj(psi2)
        return [dp1.real, dp1.imag, dp2.real, dp2.imag]

    y0 = np.random.default_rng().normal(0, 0.01, size=4)
    t_eval = np.linspace(0, t_max, n_pts)
    sol = solve_ivp(rhs, (0, t_max), y0, method='RK45', t_eval=t_eval, rtol=1e-10, atol=1e-12)

    psi1 = sol.y[0] + 1j * sol.y[1]
    psi2 = sol.y[2] + 1j * sol.y[3]
    t_steady = t_eval[n_transient:]
    psi1_steady = psi1[n_transient:]
    psi2_steady = psi2[n_transient:]

    mean_power = float(np.mean(np.abs(psi2_steady) ** 2))
    comb_active = mean_power > 1e-10
    return t_steady, psi1_steady, psi2_steady, comb_active, mean_power


def plot_time_domain(outdir, f_val, d1, d2, g21):
    """Save time-domain plot of |psi1|^2 and |psi2|^2 in the parametric-resonance regime."""
    t_steady, psi1_steady, psi2_steady, comb_active, mean_power = \
        _solve_comb_steady(f_val, d1, d2, g21)

    n_show = 2000
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_steady[:n_show], np.abs(psi1_steady[:n_show]) ** 2,
            label=r"$|\psi_1|^2$", lw=1.2, color="steelblue")
    ax.plot(t_steady[:n_show], np.abs(psi2_steady[:n_show]) ** 2,
            label=r"$|\psi_2|^2$", lw=1.2, color="darkorange")
    ax.set_xlabel(r"$\tau$ (normalised time)")
    ax.set_ylabel("Power")
    ax.set_title(
        rf"Time domain -- F={f_val}, $\Delta_1$={d1}, $\Delta_2$={d2:.3g}, $\gamma_{{21}}$={g21}"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "time_domain.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_frequency_spectrum(outdir, f_val, d1, d2, g21):
    """Save frequency-domain (FFT) plot of psi1 and psi2 showing comb teeth."""
    t_steady, psi1_steady, psi2_steady, comb_active, mean_power = \
        _solve_comb_steady(f_val, d1, d2, g21)

    dt = float(t_steady[1] - t_steady[0])

    def shifted_spectrum(psi):
        freqs = np.fft.fftshift(np.fft.fftfreq(len(psi), d=dt))
        spec = np.fft.fftshift(np.abs(np.fft.fft(psi)))
        return freqs, spec

    freqs1, spec1 = shifted_spectrum(psi1_steady)
    freqs2, spec2 = shifted_spectrum(psi2_steady)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].semilogy(freqs1, spec1, lw=0.8, color="steelblue")
    axes[0].set_ylabel(r"$|\mathrm{FFT}(\psi_1)|$ (log)")
    axes[0].set_title(
        rf"Frequency spectrum -- F={f_val}, $\Delta_1$={d1}, $\Delta_2$={d2:.3g}, $\gamma_{{21}}$={g21}"
    )
    axes[0].set_xlim(-5, 5)
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(freqs2, spec2, lw=0.8, color="darkorange")
    axes[1].set_ylabel(r"$|\mathrm{FFT}(\psi_2)|$ (log)")
    axes[1].set_xlabel(r"Frequency offset $(1/\tau)$")
    axes[1].set_xlim(-5, 5)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "frequency_spectrum.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def run_single_step(task_name, n_symbols, n_virtual, n_fft, outdir, reg_lambda,
                    n_test=None, t_symbol=None, delta_f=0.0):
    """Run single-step prediction for a given chaotic system."""
    if t_symbol is None:
        t_symbol = config.SYMBOL_DURATION_TAU

    phys_data_rate = config.GAMMA1_RAD_S / t_symbol

    n_test_actual = n_test if n_test is not None else config.N_TEST_SYMBOLS
    n_eff = n_symbols - config.WARMUP_SYMBOLS - 1
    n_train_est = n_eff - n_test_actual

    print("=" * 60)
    print("Phononic RC -- Single-step prediction")
    print(f"  Task:            {task_name}")
    print(f"  Symbols:         {n_symbols}")
    print(f"  Virtual nodes:   {n_virtual}")
    print(f"  FFT bins:        {n_fft}")
    print(f"  reg_lambda:      {reg_lambda:.2e}")
    print(f"  T_symbol:        {t_symbol} tau  ({t_symbol / config.GAMMA1_RAD_S * 1e6:.2f} us physical)")
    print(f"  Data rate:       {phys_data_rate:.4g} Sa/s  ({phys_data_rate/1e3:.4g} kSa/s)")
    print(f"  Warmup:          {config.WARMUP_SYMBOLS}")
    print(f"  Precondition:    {config.PRECONDITION_SYMBOLS}")
    print(f"  Train symbols:   ~{n_train_est}")
    print(f"  Test symbols:    {n_test_actual}")
    print(f"  F_AVG:           {config.F_AVG}")
    print(f"  delta_f:         {config.DELTA_F}")
    print(f"  Delta1:          {config.DELTA1}")
    print(f"  Delta2:          {config.DELTA2:.4g}")
    print(f"  kappa:           {config.KAPPA}")
    print(f"  gamma21:         {config.GAMMA21}")
    print("=" * 60)

    print("\n[0/3] Generating parametric-resonance plots...")
    plot_time_domain(outdir, config.F_AVG, config.DELTA1, config.DELTA2, config.GAMMA21)
    plot_frequency_spectrum(outdir, config.F_AVG, config.DELTA1, config.DELTA2, config.GAMMA21)

    print("\n[1/3] Generating time series...")
    series = get_time_series(task_name, n_symbols)
    print(f"  Generated {len(series)} samples (range [{series.min():.3f}, {series.max():.3f}])")

    print("\n[2/3] Running reservoir computer...")
    t0 = time.time()
    rc = ReservoirComputer(
        n_virtual=n_virtual,
        n_fft=n_fft,
        reg_lambda=reg_lambda,
        t_symbol=t_symbol,
        delta_f=delta_f,
    )
    results = rc.run_pipeline(series, n_test=n_test, progress=True)
    elapsed = time.time() - t0

    print(f"\n  Elapsed time:       {elapsed:.1f}s")
    print(f"  Active features:    {results['n_features_active']}")

    test_corr = correlation(results["y_test_true"], results["y_test_pred"])
    print(f"  Test correlation: {test_corr:.6f}")

    print("\n[3/3] Saving plots...")
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    ax = axes[0]
    n_show = len(results["y_test_true"])
    ax.plot(results["y_test_true"][:n_show], "b-", lw=1.0, label="True", alpha=0.8)
    ax.plot(results["y_test_pred"][:n_show], "r--", lw=1.0, label="Predicted", alpha=0.8)
    ax.set_title(
        f"{task_name} -- Test ({n_show} symbols): NMSE = {results['test_nmse']:.4e}, "
        f"log10(NMSE) = {np.log10(results['test_nmse'] + 1e-30):.2f}",
    )
    ax.set_ylabel("s(n)")
    ax.legend()

    ax = axes[1]
    error = results["y_test_true"] - results["y_test_pred"]
    ax.plot(error[:n_show], "g-", lw=0.5)
    ax.set_ylabel("Error")
    ax.set_title(f"Prediction error (std = {np.std(error):.4e})")

    ax = axes[2]
    n_show_train = min(len(results["y_train_true"]), 1000)
    ax.plot(results["y_train_true"][:n_show_train], "b-", lw=1.0, label="True", alpha=0.8)
    ax.plot(results["y_train_pred"][:n_show_train], "r--", lw=1.0, label="Predicted", alpha=0.8)
    ax.set_title(f"Training fit: NMSE = {results['train_nmse']:.4e}")
    ax.set_ylabel("s(n)")
    ax.set_xlabel("Time Step")
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(outdir, f"{task_name}_prediction.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {plot_path}")

    print("\n" + "=" * 60)
    print(f"RESULTS -- {task_name}")
    print(f"  Train NMSE:  {results['train_nmse']:.6e}")
    print(f"  Test  NMSE:  {results['test_nmse']:.6e}")
    print(f"  log10(NMSE): {np.log10(results['test_nmse'] + 1e-30):.3f}")
    print(f"  Correlation:  {test_corr:.6f}")
    print(f"  Time:         {elapsed:.1f}s")
    print("=" * 60)

    return results


def run_combined(outdir, n_symbols, n_virtual, n_fft, reg_lambda, n_test, t_symbol, delta_f=0.0):
    """
    Single combined figure: psi2 time domain + psi2 frequency spectrum +
    RC test predictions for mackey_glass, rossler, and lorenz.
    """
    tasks = ["mackey_glass", "rossler", "lorenz"]
    task_labels = {"mackey_glass": "Mackey-Glass", "rossler": "Rossler", "lorenz": "Lorenz"}

    print("=" * 60)
    print("Phononic RC -- Combined figure")
    print(f"  Tasks:         {', '.join(tasks)}")
    print(f"  Symbols:       {n_symbols}")
    print(f"  Virtual nodes: {n_virtual}")
    print(f"  FFT points:    {n_fft}")
    print(f"  Regularisation: {reg_lambda}")
    print(f"  Test points:   {n_test}")
    print(f"  Symbol time:   {t_symbol}")
    print(f"  F_AVG:         {config.F_AVG}")
    print(f"  Delta1:        {config.DELTA1}")
    print("=" * 60)

    print("\n[0/2] Simulating phononic dynamics...")
    t_steady, _, psi2_steady, _, _ = _solve_comb_steady(
        config.F_AVG, config.DELTA1, config.DELTA2, config.GAMMA21
    )
    n_show = 2000
    dt = float(t_steady[1] - t_steady[0])
    freqs2 = np.fft.fftshift(np.fft.fftfreq(len(psi2_steady), d=dt))
    spec2 = np.fft.fftshift(np.abs(np.fft.fft(psi2_steady)))

    print("\n[1/2] Running RC for all tasks...")
    predictions = {}
    for task in tasks:
        print(f"  {task}...")
        series = get_time_series(task, n_symbols)
        rc = ReservoirComputer(
            f_avg=config.F_AVG,
            n_virtual=n_virtual, n_fft=n_fft,
            reg_lambda=reg_lambda, t_symbol=t_symbol,
            delta1=config.DELTA1,
            delta2=config.DELTA2,
            gamma21=config.GAMMA21,
            delta_f=delta_f,
        )
        predictions[task] = rc.run_pipeline(series, n_test=n_test, progress=False)

    os.makedirs(outdir, exist_ok=True)

    print("\n[2/2] Saving plots...")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t_steady[:n_show], np.abs(psi2_steady[:n_show]) ** 2, lw=1.2, color="darkorange")
    ax.set_xlabel(r"$\tau$ (normalised time)")
    ax.set_ylabel(r"$|\psi_2|^2$")
    ax.set_title(
        rf"Time domain  (F={config.F_AVG}, $\Delta_1$={config.DELTA1}, $\gamma_{{21}}$={config.GAMMA21})"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "time_domain.png"), dpi=300)
    plt.close()
    print(f"  Saved: {os.path.join(outdir, 'time_domain.png')}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(freqs2, spec2, lw=0.8, color="darkorange")
    ax.set_xlabel(r"Frequency offset $(1/\tau)$")
    ax.set_ylabel(r"$|\mathrm{FFT}(\psi_2)|$ (log)")
    ax.set_title(r"Frequency spectrum ($\psi_2$)")
    ax.set_xlim(-5, 5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "frequency_spectrum.png"), dpi=300)
    plt.close()
    print(f"  Saved: {os.path.join(outdir, 'frequency_spectrum.png')}")

    for task in tasks:
        res = predictions[task]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(res["y_test_true"], "b-", lw=1.0, label="True", alpha=0.8)
        ax.plot(res["y_test_pred"], "r--", lw=1.0, label="Predicted", alpha=0.8)
        ax.set_title(
            rf"{task_labels[task]}  $\log_{{10}}$(NMSE) = {np.log10(res['test_nmse'] + 1e-30):.2f}",
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("s(n)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        pred_path = os.path.join(outdir, f"{task}_prediction.png")
        plt.savefig(pred_path, dpi=300)
        plt.close()
        print(f"  Saved: {pred_path}")

    # Combined 5-panel figure
    fig, axes = plt.subplots(5, 1, figsize=(8, 18 * SIZE_RED))
    fig.subplots_adjust(hspace=0.55)

    def _label(ax, letter):
        ax.text(-0.08, 1.05, f"({letter})", transform=ax.transAxes,
                fontweight="bold", va="bottom", ha="left")

    axes[0].plot(t_steady[:n_show], np.abs(psi2_steady[:n_show]) ** 2, lw=1.2, color="darkorange")
    axes[0].set_xlabel(r"$\tau$ (normalised time)")
    axes[0].set_ylabel(r"$|\psi_2|^2$")
    axes[0].set_title(
        rf"(a) Time domain  ($\Delta_1$={config.DELTA1}, F={config.F_AVG}, $\gamma_{{21}}$={config.GAMMA21})"
    )
    axes[0].grid(alpha=0.3)
    _label(axes[0], "a")

    axes[1].semilogy(freqs2, spec2, lw=0.8, color="darkorange")
    axes[1].set_xlabel(r"Frequency offset $(1/\tau)$")
    axes[1].set_ylabel(r"$|\mathrm{FFT}(\psi_2)|$ (log)")
    axes[1].set_title(r"(b) Frequency spectrum ($\psi_2$)")
    axes[1].set_xlim(-5, 5)
    axes[1].grid(alpha=0.3)
    _label(axes[1], "b")

    for ax, task, letter in zip(axes[2:], tasks, "cde"):
        res = predictions[task]
        ax.plot(res["y_test_true"], "b-", lw=0.8, label="True", alpha=0.8)
        ax.plot(res["y_test_pred"], "r--", lw=0.8, label="Predicted", alpha=0.8)
        ax.set_title(
            rf"({letter}) {task_labels[task]}  --  $\log_{{10}}$(NMSE) = {np.log10(res['test_nmse'] + 1e-30):.2f}",
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("s(n)")
        ax.legend()
        ax.grid(alpha=0.3)
        _label(ax, letter)

    out_path = os.path.join(outdir, "combined.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def run_data_rate_sweep(task_name, n_symbols, n_virtual, n_fft, outdir, reg_lambda,
                        rate_start, rate_stop, rate_step, n_test=None, delta_f=0.0):
    """
    Sweep data rate from rate_start to rate_stop (inclusive) in steps of rate_step (Sa/s).
    """
    data_rates = list(range(rate_start, rate_stop + 1, rate_step))

    print("=" * 60)
    print("Phononic RC -- Data Rate Sweep")
    print(f"  Task:       {task_name}")
    print(f"  Rates:      {data_rates[0]} - {data_rates[-1]} Sa/s  (step {rate_step})")
    print(f"  Points:     {len(data_rates)}")
    print(f"  delta_f:    {delta_f}")
    print("=" * 60)

    series = get_time_series(task_name, n_symbols)

    sweep_rates = []
    sweep_t_sym = []
    sweep_nmse = []
    sweep_log10 = []

    for i, dr in enumerate(data_rates):
        t_sym = config.GAMMA1_RAD_S / dr
        print(f"\n[{i+1}/{len(data_rates)}] data_rate={dr} Sa/s  "
              f"T_symbol={t_sym:.4f} tau  ({t_sym/config.GAMMA1_RAD_S*1e6:.1f} us)")

        rc = ReservoirComputer(
            n_virtual=n_virtual,
            n_fft=n_fft,
            reg_lambda=reg_lambda,
            delta_f=delta_f,
            t_symbol=t_sym,
        )
        results = rc.run_pipeline(series, n_test=n_test, progress=False)
        test_nmse = results["test_nmse"]
        log10_nmse = np.log10(test_nmse + 1e-30)

        sweep_rates.append(dr)
        sweep_t_sym.append(t_sym)
        sweep_nmse.append(test_nmse)
        sweep_log10.append(log10_nmse)

        print(f"  Test NMSE: {test_nmse:.4e}  log10(NMSE): {log10_nmse:.3f}")

    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(sweep_rates, sweep_nmse, 'bo-', lw=1.5, ms=6)
    axes[0].set_xlabel("Data Rate (Sa/s)")
    axes[0].set_ylabel("Test NMSE")
    axes[0].set_title(f"{task_name} -- NMSE vs Data Rate")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sweep_rates, sweep_log10, 'ro-', lw=1.5, ms=6)
    axes[1].set_xlabel("Data Rate (Sa/s)")
    axes[1].set_ylabel("log10(Test NMSE)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(outdir, f"{task_name}_data_rate_sweep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {plot_path}")

    print("\n" + "=" * 60)
    print(f"{'Data Rate (Sa/s)':>18}  {'T_symbol (tau)':>14}  {'T_symbol (us)':>14}  {'log10(NMSE)':>12}")
    print("-" * 60)
    for dr, ts, ln in zip(sweep_rates, sweep_t_sym, sweep_log10):
        print(f"{dr:>18}  {ts:>14.4f}  {ts/config.GAMMA1_RAD_S*1e6:>13.1f}  {ln:>12.3f}")
    print("=" * 60)


def _run_one_cell(args):
    """Worker: run one RC cell."""
    (ri, di, fi, d1, fav, t_sym, task_name, n_symbols,
     n_virtual, n_fft, reg_lambda, n_test, gamma21, delta_f) = args
    series = get_time_series(task_name, n_symbols)
    d2 = d1 / 2.0 + config.KAPPA
    rc = ReservoirComputer(
        f_avg=fav,
        delta_f=delta_f,
        n_virtual=n_virtual,
        n_fft=n_fft,
        reg_lambda=reg_lambda,
        t_symbol=t_sym,
        delta1=d1,
        delta2=d2,
        gamma21=gamma21,
    )
    results = rc.run_pipeline(series, n_test=n_test, progress=False)
    return ri, di, fi, np.log10(results["test_nmse"] + 1e-30)


def _subplot_layout(n):
    """Return (nrows, ncols) for a compact grid fitting n subplots."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def run_f_delta1_single(task_name, n_symbols, n_virtual, n_fft, outdir, reg_lambda,
                        favg_min, favg_max, favg_n,
                        delta1_min, delta1_max, delta1_n,
                        t_symbol=None, n_test=None, delta_f=0.0):
    """Single F_AVG x Delta1 heatmap at a fixed data rate / t_symbol."""
    if t_symbol is None:
        t_symbol = config.SYMBOL_DURATION_TAU
    dr = config.GAMMA1_RAD_S / t_symbol

    favg_vals = np.linspace(favg_min, favg_max, favg_n)
    delta1_vals = np.linspace(delta1_min, delta1_max, delta1_n)

    print("=" * 60)
    print("Phononic RC -- F_AVG x Delta1 heatmap")
    print(f"  Task:      {task_name}")
    print(f"  F_AVG:     {favg_min} - {favg_max}  ({favg_n} pts)")
    print(f"  Delta1:    {delta1_min} - {delta1_max}  ({delta1_n} pts)")
    print(f"  Data rate: {dr:.1f} Sa/s  (T_symbol={t_symbol:.4f} tau)")
    print(f"  delta_f:   {delta_f}")
    print("=" * 60)

    os.makedirs(outdir, exist_ok=True)
    plot_path = os.path.join(outdir, f"{task_name}_f_delta1.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        np.full((favg_n, delta1_n), np.nan),
        origin="lower", aspect="auto",
        extent=[delta1_min, delta1_max, favg_min, favg_max],  # type: ignore
        vmax=0,
        cmap="jet",
    )
    plt.colorbar(im, ax=ax, label="log10(NMSE)")
    ax.set_xlabel("Delta1")
    ax.set_ylabel("F_AVG")
    ax.set_title(f"{task_name} -- log10(NMSE): F_AVG x Delta1  ({dr:.1f} Sa/s, delta_f={delta_f})")
    plt.tight_layout()

    grid = np.full((favg_n, delta1_n), np.nan)
    total = favg_n * delta1_n
    done = 0
    save_every = 5

    csv_path = os.path.join(outdir, f"{task_name}_f_delta1.csv")
    csv_existed = os.path.exists(csv_path)
    if csv_existed:
        df = pd.read_csv(csv_path)
        loaded = 0
        for _, row in df.iterrows():
            fi = int(np.argmin(np.abs(favg_vals - float(row["f_avg"]))))
            di = int(np.argmin(np.abs(delta1_vals - float(row["delta1"]))))
            grid[fi, di] = float(row["log10_nmse"])
            loaded += 1
        print(f"  Resumed: loaded {loaded}/{total} existing cells from {csv_path}")

    csv_file = open(csv_path, "a" if csv_existed else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_existed:
        csv_writer.writerow(["f_avg", "delta1", "log10_nmse"])

    jobs = [
        (0, di, fi, d1, fav, t_symbol, task_name, n_symbols,
         n_virtual, n_fft, reg_lambda, n_test, config.GAMMA21, delta_f)
        for di, d1 in enumerate(delta1_vals)
        for fi, fav in enumerate(favg_vals)
        if np.isnan(grid[fi, di])
    ]
    print(f"  Jobs remaining: {len(jobs)} / {total}")

    with ProcessPoolExecutor(max_workers=_get_max_workers()) as executor:
        futures = {executor.submit(_run_one_cell, j): j for j in jobs}
        for fut in as_completed(futures):
            _, di, fi, log_nmse = fut.result()
            grid[fi, di] = log_nmse
            done += 1

            csv_writer.writerow([f"{favg_vals[fi]:.4f}", f"{delta1_vals[di]:.4f}", f"{log_nmse:.6f}"])
            csv_file.flush()

            if done % save_every == 0 or done == total:
                valid = grid[~np.isnan(grid)]
                im.set_data(grid)
                if len(valid):
                    im.set_clim(valid.min(), 0)
                fig.savefig(plot_path, dpi=150)

    csv_file.close()
    plt.close()
    print(f"\n  Saved: {plot_path}")
    print(f"  Saved: {csv_path}")


def load_existing_grids(csv_path, data_rates, favg_vals, delta1_vals, favg_n, delta1_n):
    """Load a combined sweep CSV and restore grids. Returns list of grids (one per data rate)."""
    grids = [np.full((favg_n, delta1_n), np.nan) for _ in data_rates]

    if not os.path.exists(csv_path):
        return grids

    df = pd.read_csv(csv_path)
    loaded = 0
    for _, row in df.iterrows():
        dr = int(round(row["data_rate_Sa"]))
        fav = float(row["f_avg"])
        d1 = float(row["delta1"])
        val = float(row["log10_nmse"])

        if dr not in data_rates:
            continue

        ri = data_rates.index(dr)
        fi = int(np.argmin(np.abs(favg_vals - fav)))
        di = int(np.argmin(np.abs(delta1_vals - d1)))
        grids[ri][fi, di] = val
        loaded += 1

    total = favg_n * delta1_n * len(data_rates)
    print(f"  Resumed: loaded {loaded}/{total} existing cells from {csv_path}")
    return grids


def run_f_delta1_sweep(task_name, n_symbols, n_virtual, n_fft, outdir, reg_lambda,
                       favg_min, favg_max, favg_n,
                       delta1_min, delta1_max, delta1_n,
                       rate_start, rate_stop, rate_step,
                       n_test=None, init_grids=None, data_rates=None, delta_f=0.0):
    """
    Multi-subplot heatmap: F_AVG vs Delta1, one subplot per data rate.
    Each cell = log10(NMSE). Delta2 tracks Delta1 via: Delta2 = Delta1/2 + kappa.
    """
    if data_rates is None:
        data_rates = list(range(rate_start, rate_stop + 1, rate_step))
    n_plots = len(data_rates)
    nrows, ncols = _subplot_layout(n_plots)

    favg_vals = np.linspace(favg_min, favg_max, favg_n)
    delta1_vals = np.linspace(delta1_min, delta1_max, delta1_n)

    total_runs = n_plots * favg_n * delta1_n
    print("=" * 60)
    print(f"Phononic RC -- F_AVG x Delta1 sweep  ({n_plots} data rates)")
    print(f"  Task:     {task_name}")
    print(f"  F_AVG:    {favg_min} - {favg_max}  ({favg_n} pts)")
    print(f"  Delta1:   {delta1_min} - {delta1_max}  ({delta1_n} pts)")
    print(f"  Rates:    {data_rates[0]} - {data_rates[-1]} Sa/s  (step {rate_step})")
    print(f"  delta_f:  {delta_f}")
    print(f"  Total RC runs: {total_runs}")
    print("=" * 60)

    os.makedirs(outdir, exist_ok=True)
    plot_path = os.path.join(outdir, f"{task_name}_f_delta1_sweep.png")

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows * SIZE_RED))
    axes = np.array(axes).flatten()
    ims = []
    for idx, (ax, dr) in enumerate(zip(axes, data_rates)):
        im = ax.imshow(
            np.full((favg_n, delta1_n), np.nan),
            origin="lower", aspect="auto",
            extent=[delta1_min, delta1_max, favg_min, favg_max],
            vmax=0,
            cmap="viridis_r",
        )
        plt.colorbar(im, ax=ax, label="log10(NMSE)")
        ax.set_title(f"{dr} Sa/s", fontsize=10)
        ax.set_xlabel("Delta1", fontsize=8)
        ax.set_ylabel("F_AVG", fontsize=8)
        ims.append(im)
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    fig.suptitle(f"{task_name} -- log10(NMSE): F_AVG x Delta1  (delta_f={delta_f}, Delta2=Delta1/2+kappa)", fontsize=13)
    plt.tight_layout()

    grids = init_grids if init_grids is not None else \
            [np.full((favg_n, delta1_n), np.nan) for _ in data_rates]
    all_jobs = []
    for ri, dr in enumerate(data_rates):
        t_sym = config.GAMMA1_RAD_S / dr
        for di, d1 in enumerate(delta1_vals):
            for fi, fav in enumerate(favg_vals):
                if np.isnan(grids[ri][fi, di]):
                    all_jobs.append(
                        (ri, di, fi, d1, fav, t_sym, task_name, n_symbols,
                         n_virtual, n_fft, reg_lambda, n_test, config.GAMMA21, delta_f)
                    )
    np.random.default_rng().shuffle(all_jobs)
    print(f"  Jobs remaining: {len(all_jobs)} / {len(data_rates) * favg_n * delta1_n}")

    total = len(all_jobs)
    done = 0
    save_every = 10

    combined_path = os.path.join(outdir, f"{task_name}_f_delta1_sweep.csv")

    csv_existed = os.path.exists(combined_path)
    combined_file = open(combined_path, "a", newline="")
    combined_writer = csv.writer(combined_file)
    if not csv_existed:
        combined_writer.writerow(["data_rate_Sa", "f_avg", "delta1", "log10_nmse"])

    with ProcessPoolExecutor(max_workers=_get_max_workers()) as executor:
        futures = {executor.submit(_run_one_cell, j): j for j in all_jobs}
        pbar = _tqdm(as_completed(futures), total=total, desc="RC Sweep")
        for fut in pbar:
            ri, di, fi, log_nmse = fut.result()
            grids[ri][fi, di] = log_nmse
            done += 1

            dr = data_rates[ri]
            fav = favg_vals[fi]
            d1 = delta1_vals[di]
            combined_writer.writerow([dr, f"{fav:.4f}", f"{d1:.4f}", f"{log_nmse:.6f}"])
            combined_file.flush()

            if done % save_every == 0 or done == total:
                for ri_, im in enumerate(ims):
                    valid = grids[ri_][~np.isnan(grids[ri_])]
                    im.set_data(grids[ri_])
                    if len(valid):
                        im.set_clim(valid.min(), 0)
                fig.savefig(plot_path, dpi=150)

                for ri_, dr_ in enumerate(data_rates):
                    csv_path = os.path.join(outdir, f"{task_name}_data_rate_{dr_}Sa.csv")
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["favg_\\ delta1"] + [f"{v:.4f}" for v in delta1_vals])
                        for fi_, fav_ in enumerate(favg_vals):
                            w.writerow([f"{fav_:.4f}"] +
                                       [f"{grids[ri_][fi_, di_]:.6f}" for di_ in range(delta1_n)])

                pbar.set_postfix(rate=f"{data_rates[ri]}Sa/s", log10_nmse=f"{log_nmse:.2f}")

    combined_file.close()
    plt.close()
    print(f"\n  Saved: {plot_path}")
    print(f"  Saved: {combined_path}")


def run_gamma21_sweep(task_name, n_symbols, n_virtual, n_fft, outdir, reg_lambda,
                      favg_min, favg_max, favg_n,
                      delta1_min, delta1_max, delta1_n,
                      gamma21_vals, data_rate, n_test=None, delta_f=0.0):
    """F_AVG x Delta1 heatmap for each gamma21 value at a fixed data rate."""
    t_sym = config.GAMMA1_RAD_S / data_rate
    n_plots = len(gamma21_vals)
    nrows, ncols = _subplot_layout(n_plots)

    favg_vals = np.linspace(favg_min, favg_max, favg_n)
    delta1_vals = np.linspace(delta1_min, delta1_max, delta1_n)

    print("=" * 60)
    print(f"Phononic RC -- gamma21 sweep  ({n_plots} values)")
    print(f"  Task:      {task_name}")
    print(f"  gamma21:   {gamma21_vals}")
    print(f"  Data rate: {data_rate} Sa/s  (T_symbol={t_sym:.4f} tau)")
    print(f"  F_AVG:     {favg_min} - {favg_max}  ({favg_n} pts)")
    print(f"  Delta1:    {delta1_min} - {delta1_max}  ({delta1_n} pts)")
    print(f"  Total RC runs: {n_plots * favg_n * delta1_n}")
    print("=" * 60)

    os.makedirs(outdir, exist_ok=True)
    plot_path = os.path.join(outdir, f"{task_name}_gamma21_sweep.png")
    combined_path = os.path.join(outdir, f"{task_name}_gamma21_sweep.csv")

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows * SIZE_RED))
    axes = np.array(axes).flatten()
    ims = []
    for ax, g21 in zip(axes, gamma21_vals):
        im = ax.imshow(
            np.full((favg_n, delta1_n), np.nan),
            origin="lower", aspect="auto",
            extent=[delta1_min, delta1_max, favg_min, favg_max],
            vmax=0, cmap="viridis_r",
        )
        plt.colorbar(im, ax=ax, label="log10(NMSE)")
        ax.set_title(f"gamma21 = {g21}", fontsize=10)
        ax.set_xlabel("Delta1", fontsize=8)
        ax.set_ylabel("F_AVG", fontsize=8)
        ims.append(im)
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    fig.suptitle(
        f"{task_name} -- log10(NMSE): F_AVG x Delta1  "
        f"(data_rate={data_rate} Sa/s, delta_f={delta_f}, kappa={config.KAPPA})",
        fontsize=13,
    )
    plt.tight_layout()

    grids = [np.full((favg_n, delta1_n), np.nan) for _ in gamma21_vals]
    all_jobs = [
        (ri, di, fi, d1, fav, t_sym, task_name, n_symbols,
         n_virtual, n_fft, reg_lambda, n_test, g21, delta_f)
        for ri, g21 in enumerate(gamma21_vals)
        for di, d1 in enumerate(delta1_vals)
        for fi, fav in enumerate(favg_vals)
    ]
    np.random.default_rng().shuffle(all_jobs)

    total = len(all_jobs)
    done = 0

    combined_file = open(combined_path, "w", newline="")
    combined_writer = csv.writer(combined_file)
    combined_writer.writerow(["gamma21", "f_avg", "delta1", "log10_nmse"])

    with ProcessPoolExecutor(max_workers=_get_max_workers()) as executor:
        futures = {executor.submit(_run_one_cell, j): j for j in all_jobs}
        with _tqdm(total=total, miniters=10, desc="gamma21 sweep") as pbar:
            for fut in as_completed(futures):
                ri, di, fi, log_nmse = fut.result()
                grids[ri][fi, di] = log_nmse
                done += 1

                combined_writer.writerow([
                    gamma21_vals[ri], f"{favg_vals[fi]:.4f}",
                    f"{delta1_vals[di]:.4f}", f"{log_nmse:.6f}",
                ])
                combined_file.flush()

                pbar.set_postfix({"gamma21": gamma21_vals[ri], "log10NMSE": f"{log_nmse:.2f}"})
                pbar.update(1)

                if done % 10 == 0 or done == total:
                    for ri_, im in enumerate(ims):
                        valid = grids[ri_][~np.isnan(grids[ri_])]
                        im.set_data(grids[ri_])
                        if len(valid):
                            im.set_clim(valid.min(), 0)
                    fig.savefig(plot_path, dpi=150)

    combined_file.close()
    plt.close()
    print(f"\n  Saved: {plot_path}")
    print(f"  Saved: {combined_path}")


def _parse_rate_list(arg):
    """
    Parse --data_rates: accepts bracket list, comma-separated, or a single int.
    Examples: '[1000,5000,10000]', '1000,5000,10000', '2000'.
    """
    if arg is None:
        return None
    import ast
    s = str(arg).strip()
    if not s:
        return None
    try:
        if s.startswith("["):
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return sorted(int(x) for x in v)
            return [int(v)]
    except (SyntaxError, ValueError, TypeError):
        pass
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return sorted(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="Phononic Frequency Comb Reservoir Computer"
    )
    parser.add_argument(
        "--task", type=str, default="mackey_glass",
        choices=["mackey_glass", "lorenz", "rossler"],
        help="Chaotic system benchmark (default: mackey_glass)",
    )
    parser.add_argument(
        "--n_symbols", type=int, default=config.N_SYMBOLS,
        help=f"Total number of symbols (default: {config.N_SYMBOLS})",
    )
    parser.add_argument(
        "--n_virtual", type=int, default=config.N_VIRTUAL,
        help=f"Virtual nodes per symbol (default: {config.N_VIRTUAL})",
    )
    parser.add_argument(
        "--n_fft", type=int, default=config.N_FFT_FEATURES,
        help=f"FFT features per mode (default: {config.N_FFT_FEATURES})",
    )
    parser.add_argument(
        "--reg_lambda", type=float, default=config.REG_LAMBDA,
        help=f"Ridge regularisation (default: {config.REG_LAMBDA})",
    )
    parser.add_argument(
        "--n_test", type=int, default=config.N_TEST_SYMBOLS,
        help="Number of test symbols",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory. If set and a prior sweep CSV exists, resumes from it.",
    )
    parser.add_argument(
        "--data_rate", type=float, default=None,
        help="Target physical data rate in Sa/s. Overrides --t_symbol.",
    )
    parser.add_argument(
        "--t_symbol", type=float, default=None,
        help=f"Symbol duration in normalised time (default: {config.SYMBOL_DURATION_TAU}). "
             "Ignored if --data_rate is set.",
    )
    parser.add_argument(
        "--sweep_data_rate", action="store_true",
        help="Sweep data rate from --sweep_start to --sweep_stop.",
    )
    parser.add_argument(
        "--sweep_f_delta1", action="store_true",
        help="Heatmap sweep: F_AVG x Delta1.",
    )
    parser.add_argument("--sweep_start", type=int, default=2000)
    parser.add_argument("--sweep_stop",  type=int, default=10000)
    parser.add_argument("--sweep_step",  type=int, default=1000)
    parser.add_argument(
        "--data_rates", type=str, default=None,
        help="Custom list of data rates (Sa/s): '[2000,10000,96000]' or '2000,10000,96000'.",
    )
    parser.add_argument("--favg_min",    type=float, default=0.0)
    parser.add_argument("--favg_max",    type=float, default=50.0)
    parser.add_argument("--favg_n",      type=int,   default=50)
    parser.add_argument("--delta1_min",  type=float, default=-8.0, help="Delta1 sweep min")
    parser.add_argument("--delta1_max",  type=float, default=8.0,  help="Delta1 sweep max")
    parser.add_argument("--delta1_n",    type=int,   default=40,   help="Delta1 grid points")
    parser.add_argument(
        "--sweep_gamma21", action="store_true",
        help="Sweep gamma21 values at a fixed data rate (set via --data_rate).",
    )
    parser.add_argument(
        "--gamma21_vals", type=float, nargs="+",
        default=[0.01, 0.1, 1, 10, 100],
        help="gamma21 values for --sweep_gamma21",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Combined figure: psi2 time domain + frequency spectrum + RC predictions.",
    )
    parser.add_argument(
        "--delta_f", type=float, default=0.0,
        help="Drive modulation depth delta_f (default: 0.0).",
    )
    parser.add_argument(
        "--delta1_val", type=float, default=None,
        help="Override config DELTA1 (Delta2 recomputed as delta1/2 + kappa).",
    )
    parser.add_argument(
        "--favg_val", type=float, default=None,
        help="Override config F_AVG.",
    )
    parser.add_argument(
        "--regression", type=str, default=None, choices=["ridge", "lasso"],
        help="Regression method for the output layer.",
    )

    args = parser.parse_args()

    _workers = _get_max_workers()
    _src = f"SLURM_CPUS_PER_TASK={os.environ['SLURM_CPUS_PER_TASK']}" if os.environ.get("SLURM_CPUS_PER_TASK") else "os.cpu_count()"
    print(f"[workers] using {_workers} CPU cores ({_src})")

    if args.delta1_val is not None:
        config.DELTA1 = float(args.delta1_val)
        config.DELTA2 = config.DELTA1 / 2 + config.KAPPA
    if args.favg_val is not None:
        config.F_AVG = float(args.favg_val)
    if args.regression is not None:
        config.REGRESSION = args.regression

    outdir = args.outdir if args.outdir is not None else f"results/{args.task}"

    if args.combine:
        if args.data_rate is not None:
            t_symbol = config.GAMMA1_RAD_S / args.data_rate
        elif args.t_symbol is not None:
            t_symbol = args.t_symbol
        else:
            t_symbol = config.SYMBOL_DURATION_TAU
        run_combined(
            outdir=outdir,
            n_symbols=args.n_symbols,
            n_virtual=args.n_virtual,
            n_fft=args.n_fft,
            reg_lambda=args.reg_lambda,
            n_test=args.n_test,
            t_symbol=t_symbol,
            delta_f=args.delta_f,
        )
    elif args.sweep_gamma21:
        dr = args.data_rate if args.data_rate is not None else 2400.0
        run_gamma21_sweep(
            task_name=args.task,
            n_symbols=args.n_symbols,
            n_virtual=args.n_virtual,
            n_fft=args.n_fft,
            outdir=outdir,
            reg_lambda=args.reg_lambda,
            favg_min=args.favg_min,
            favg_max=args.favg_max,
            favg_n=args.favg_n,
            delta1_min=args.delta1_min,
            delta1_max=args.delta1_max,
            delta1_n=args.delta1_n,
            gamma21_vals=args.gamma21_vals,
            data_rate=dr,
            n_test=args.n_test,
            delta_f=args.delta_f,
        )
    elif args.sweep_data_rate:
        custom_rates = _parse_rate_list(args.data_rates)
        if custom_rates is not None:
            data_rates = custom_rates
            print(f"  Using --data_rates: {data_rates}")
        else:
            data_rates = list(range(args.sweep_start, args.sweep_stop + 1, args.sweep_step))
        favg_vals = np.linspace(args.favg_min, args.favg_max, args.favg_n)
        delta1_vals = np.linspace(args.delta1_min, args.delta1_max, args.delta1_n)
        combined_csv = os.path.join(outdir, f"{args.task}_f_delta1_sweep.csv")
        init_grids = None
        if args.outdir is not None:
            init_grids = load_existing_grids(
                combined_csv, data_rates, favg_vals, delta1_vals,
                args.favg_n, args.delta1_n,
            )
        run_f_delta1_sweep(
            task_name=args.task,
            n_symbols=args.n_symbols,
            n_virtual=args.n_virtual,
            n_fft=args.n_fft,
            outdir=outdir,
            reg_lambda=args.reg_lambda,
            favg_min=args.favg_min,
            favg_max=args.favg_max,
            favg_n=args.favg_n,
            delta1_min=args.delta1_min,
            delta1_max=args.delta1_max,
            delta1_n=args.delta1_n,
            rate_start=args.sweep_start,
            rate_stop=args.sweep_stop,
            rate_step=args.sweep_step,
            n_test=args.n_test,
            init_grids=init_grids,
            data_rates=data_rates,
            delta_f=args.delta_f,
        )
    elif args.sweep_f_delta1:
        if args.data_rate is not None:
            t_symbol = config.GAMMA1_RAD_S / args.data_rate
        elif args.t_symbol is not None:
            t_symbol = args.t_symbol
        else:
            t_symbol = config.SYMBOL_DURATION_TAU
        run_f_delta1_single(
            task_name=args.task,
            n_symbols=args.n_symbols,
            n_virtual=args.n_virtual,
            n_fft=args.n_fft,
            outdir=outdir,
            reg_lambda=args.reg_lambda,
            favg_min=args.favg_min,
            favg_max=args.favg_max,
            favg_n=args.favg_n,
            delta1_min=args.delta1_min,
            delta1_max=args.delta1_max,
            delta1_n=args.delta1_n,
            t_symbol=t_symbol,
            n_test=args.n_test,
            delta_f=args.delta_f,
        )
    else:
        if args.data_rate is not None:
            t_symbol = config.GAMMA1_RAD_S / args.data_rate
        elif args.t_symbol is not None:
            t_symbol = args.t_symbol
        else:
            t_symbol = config.SYMBOL_DURATION_TAU

        run_single_step(
            task_name=args.task,
            n_symbols=args.n_symbols,
            n_virtual=args.n_virtual,
            n_fft=args.n_fft,
            outdir=outdir,
            reg_lambda=args.reg_lambda,
            n_test=args.n_test,
            t_symbol=t_symbol,
            delta_f=args.delta_f,
        )


if __name__ == "__main__":
    main()
