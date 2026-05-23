"""
Multi-panel parameter sweep for the Phononic Frequency Comb Reservoir Computer.

Physics: Coupled-mode equations Eqs. (5)-(6) from Qi et al., APL 117, 183503 (2020)
RC framework: Shaabani Shishavan et al., Phys. Rev. Research 7, L042008 (2025)

Panel (a): F_AVG x Delta1  with  delta_f = 0  (multiplicative encoding: f(n)=F_avg*(0.5+s(n)))
Panel (b): F_AVG x Delta1  with  delta_f = config.DELTA_F  (additive modulation)
Panel (c): F_AVG x delta_f  with  Delta1 = config.DELTA1
Panel (d): Delta1 x delta_f  with  F_AVG = config.F_AVG

With --panel all, one step is shared so F_AVG (and Delta1 where used) grids align.
Fixed: gamma21 from config; kappa from config except panel (a), which can sweep kappa via --kappa.

Outputs:
    results/sweep_favg_vs_delta1.csv
    results/sweep_favg_vs_delta1_modulated.csv
    results/sweep_favg_vs_deltaf.csv
    results/sweep_deltaf_vs_delta1.csv
    results/sweep.png   -- 2xN grid: log10(NMSE) + Qi comb map

Usage:
    python sweep.py --jobs 1000 [--workers N] [--panel a|b|c|d|all] [--task mackey_glass]
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import math
import ast
import numpy as np
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from comb_numeric import (
    comb_qi_analytical,
    evaluate_comb_at_point,
)

F_AVG_RANGE = (0.0, 50.0)
DELTA1_RANGE = (-8.0, 8.0)
DELTA_F_RANGE = (0.0, 5.0)


def parse_kappa_list(arg):
    """
    Parse --kappa for panel (a): list syntax, comma-separated, or a single float.
    Examples: '[-9, -4, 0, 4, 9]', '-9,-4,0,4,9', '-9'.
    """
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        return None
    try:
        if s.startswith("["):
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [float(x) for x in v]
            return [float(v)]
    except (SyntaxError, ValueError, TypeError):
        pass
    inner = s.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    return [float(p) for p in parts]


def compute_step(total_jobs, panels, n_kappa_a=1):
    """Binary-search for the step size that produces ~total_jobs across selected panels."""
    favg_span = F_AVG_RANGE[1] - F_AVG_RANGE[0]
    d1_span = DELTA1_RANGE[1] - DELTA1_RANGE[0]
    df_span = DELTA_F_RANGE[1] - DELTA_F_RANGE[0]
    n_kappa_a = max(int(n_kappa_a), 1)

    def count_jobs(step):
        n_favg = int(round(favg_span / step)) + 1
        n_d1 = int(round(d1_span / step)) + 1
        n_df = int(round(df_span / step)) + 1
        n = 0
        if "a" in panels:
            n += n_favg * n_d1 * n_kappa_a
        if "b" in panels:
            n += n_favg * n_d1
        if "c" in panels:
            n += n_favg * n_df
        if "d" in panels:
            n += n_d1 * n_df
        return n

    lo, hi = 0.01, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if count_jobs(mid) > total_jobs:
            lo = mid
        else:
            hi = mid

    return max(round((lo + hi) / 2, 4), 0.01)


def make_axis(lo, hi, step):
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def compute_n_symbols(n_test, warmup, train_frac):
    n_eff = math.ceil(n_test / (1.0 - train_frac) - 1e-9)
    return n_eff + warmup + 1


def run_one(job):
    """
    RC evaluation plus numerical comb check for one (f_avg, Delta1, ...) point.
    """
    import config as cfg
    from chaotic_systems import get_time_series
    from reservoir import ReservoirComputer
    import traceback

    f_avg = job["f_avg"]
    delta_f = job["delta_f"]
    delta1 = job["delta1"]
    gamma21 = job["gamma21"]
    kappa = job["kappa"]
    delta2 = delta1 / 2.0 + kappa

    err = None
    series = get_time_series(job["task"], job["n_symbols"])
    try:
        rc = ReservoirComputer(
            f_avg=f_avg,
            delta_f=delta_f,
            n_virtual=cfg.N_VIRTUAL,
            n_fft=cfg.N_FFT_FEATURES,
            reg_lambda=cfg.REG_LAMBDA,
            warmup=min(cfg.WARMUP_SYMBOLS, job["n_symbols"] // 5),
            precondition=cfg.PRECONDITION_SYMBOLS,
            t_symbol=cfg.SYMBOL_DURATION_TAU,
            delta1=delta1,
            delta2=delta2,
            gamma21=gamma21,
        )
        res = rc.run_pipeline(series, n_test=job["n_test"], progress=False)
        nmse = res["test_nmse"]
    except Exception:
        nmse = float("nan")
        err = traceback.format_exc()

    comb_qi, _qi_diag = comb_qi_analytical(f_avg, delta1, delta2, gamma21)

    # IVP pulsing: bracket drive when delta_f > 0 (same envelope as RC encoding s in [0,1])
    if delta_f > 0.0:
        f_lo = max(f_avg - 0.5 * delta_f, 1e-9)
        f_hi = f_avg + 0.5 * delta_f
        ev_lo = evaluate_comb_at_point(f_lo, delta1, delta2, gamma21)
        ev_hi = evaluate_comb_at_point(f_hi, delta1, delta2, gamma21)
        comb_pulsing = ev_lo.comb_pulsing or ev_hi.comb_pulsing
        comb_mod_ratio = max(ev_lo.mod_ratio, ev_hi.mod_ratio)
    else:
        ev = evaluate_comb_at_point(f_avg, delta1, delta2, gamma21)
        comb_pulsing = ev.comb_pulsing
        comb_mod_ratio = ev.mod_ratio

    out = {
        **job,
        "nmse": nmse,
        "comb_qi": comb_qi,
        "comb_pulsing": comb_pulsing,
        "comb_mod_ratio": comb_mod_ratio,
    }
    if err is not None:
        out["error"] = err
    return out


def build_panel_a(favg_vals, d1_vals, n_symbols, n_test, task, kappa_vals):
    """F_AVG x Delta1 x kappa. Fixed: delta_f=0, gamma21."""
    import config as cfg
    jobs = []
    for kappa in kappa_vals:
        k = round(float(kappa), 4)
        for f in favg_vals:
            for d1 in d1_vals:
                jobs.append({
                    "panel": "a",
                    "f_avg": round(float(f), 4),
                    "delta1": round(float(d1), 4),
                    "delta_f": 0.0,
                    "gamma21": cfg.GAMMA21,
                    "kappa": k,
                    "n_symbols": n_symbols,
                    "n_test": n_test,
                    "task": task,
                })
    return jobs


def build_panel_b(favg_vals, d1_vals, n_symbols, n_test, task):
    """F_AVG x Delta1. Fixed: delta_f=config.DELTA_F, gamma21, kappa."""
    import config as cfg
    jobs = []
    dfixed = round(float(cfg.DELTA_F), 4)
    for f in favg_vals:
        for d1 in d1_vals:
            jobs.append({
                "panel": "b",
                "f_avg": round(float(f), 4),
                "delta1": round(float(d1), 4),
                "delta_f": dfixed,
                "gamma21": cfg.GAMMA21,
                "kappa": cfg.KAPPA,
                "n_symbols": n_symbols,
                "n_test": n_test,
                "task": task,
            })
    return jobs


def build_panel_c(favg_vals, df_vals, n_symbols, n_test, task):
    """F_AVG x delta_f. Fixed: Delta1, gamma21, kappa from config."""
    import config as cfg
    jobs = []
    for f in favg_vals:
        for df in df_vals:
            jobs.append({
                "panel": "c",
                "f_avg": round(float(f), 4),
                "delta_f": round(float(df), 4),
                "delta1": cfg.DELTA1,
                "gamma21": cfg.GAMMA21,
                "kappa": cfg.KAPPA,
                "n_symbols": n_symbols,
                "n_test": n_test,
                "task": task,
            })
    return jobs


def build_panel_d(d1_vals, df_vals, n_symbols, n_test, task):
    """Delta1 x delta_f. Fixed: F_AVG, gamma21, kappa from config."""
    import config as cfg
    jobs = []
    f_fix = round(float(cfg.F_AVG), 4)
    for d1 in d1_vals:
        for df in df_vals:
            jobs.append({
                "panel": "d",
                "f_avg": f_fix,
                "delta1": round(float(d1), 4),
                "delta_f": round(float(df), 4),
                "gamma21": cfg.GAMMA21,
                "kappa": cfg.KAPPA,
                "n_symbols": n_symbols,
                "n_test": n_test,
                "task": task,
            })
    return jobs


def _csv_row_for_result(r, panel_name):
    nm = r["nmse"]
    logn = float(np.log10(nm + 1e-30)) if not np.isnan(nm) else float("nan")
    cq = r.get("comb_qi")
    cp = r.get("comb_pulsing")
    cq_s = "" if cq is None else str(int(bool(cq)))
    cp_s = "" if cp is None else str(int(bool(cp)))
    cr = r.get("comb_mod_ratio")
    cr_s = "" if cr is None else f"{float(cr):.6f}"
    if panel_name == "a":
        d2 = r["delta1"] / 2.0 + r["kappa"]
        return [
            f"{r['kappa']:.4f}",
            f"{r['f_avg']:.4f}",
            f"{r['delta1']:.4f}",
            f"{d2:.4f}",
            f"{nm:.6e}",
            f"{logn:.4f}",
            cq_s, cp_s, cr_s,
        ]
    if panel_name == "b":
        d2 = r["delta1"] / 2.0 + r["kappa"]
        return [
            f"{r['f_avg']:.4f}",
            f"{r['delta1']:.4f}",
            f"{d2:.4f}",
            f"{nm:.6e}",
            f"{logn:.4f}",
            cq_s, cp_s, cr_s,
        ]
    if panel_name == "d":
        d2 = r["delta1"] / 2.0 + r["kappa"]
        return [
            f"{r['f_avg']:.4f}",
            f"{r['delta1']:.4f}",
            f"{d2:.4f}",
            f"{r['delta_f']:.4f}",
            f"{nm:.6e}",
            f"{logn:.4f}",
            cq_s, cp_s, cr_s,
        ]
    return [
        f"{r['f_avg']:.4f}",
        f"{r['delta_f']:.4f}",
        f"{nm:.6e}",
        f"{logn:.4f}",
        cq_s, cp_s, cr_s,
    ]


def write_csv(path, rows, panel_name):
    """Write full CSV (overwrites path)."""
    extra = ["COMB_QI", "COMB_PULSING", "COMB_MOD_RATIO"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if panel_name == "a":
            w.writerow(["KAPPA", "F_AVG", "DELTA1", "DELTA2", "NMSE", "log10_NMSE", *extra])
        elif panel_name == "b":
            w.writerow(["F_AVG", "DELTA1", "DELTA2", "NMSE", "log10_NMSE", *extra])
        elif panel_name == "d":
            w.writerow(["F_AVG", "DELTA1", "DELTA2", "DELTA_F", "NMSE", "log10_NMSE", *extra])
        else:
            w.writerow(["F_AVG", "DELTA_F", "NMSE", "log10_NMSE", *extra])
        for r in rows:
            w.writerow(_csv_row_for_result(r, panel_name))


def open_live_csv(path, panel_name):
    """Open CSV, write header, return (file, writer)."""
    extra = ["COMB_QI", "COMB_PULSING", "COMB_MOD_RATIO"]
    f = open(path, "w", newline="")
    w = csv.writer(f)
    if panel_name == "a":
        w.writerow(["KAPPA", "F_AVG", "DELTA1", "DELTA2", "NMSE", "log10_NMSE", *extra])
    elif panel_name == "b":
        w.writerow(["F_AVG", "DELTA1", "DELTA2", "NMSE", "log10_NMSE", *extra])
    elif panel_name == "d":
        w.writerow(["F_AVG", "DELTA1", "DELTA2", "DELTA_F", "NMSE", "log10_NMSE", *extra])
    else:
        w.writerow(["F_AVG", "DELTA_F", "NMSE", "log10_NMSE", *extra])
    f.flush()
    return f, w


def append_live_csv(writer, file_obj, result, panel_name):
    writer.writerow(_csv_row_for_result(result, panel_name))
    file_obj.flush()


SWEEP_PLOT_PX_W = 4050
SWEEP_PLOT_PX_H = 1500
SWEEP_PLOT_DPI = 150
SWEEP_PLOT_BASE_COLS = 3
NMSE_HEATMAP_MISSING_HEX = "#d4d4d4"


def _comb_cell_from_result(r):
    v = r.get("comb_qi")
    if v is None:
        return np.nan
    return 1.0 if v else 0.0


def _sweep_plot_columns(panels, kappas_a):
    """One matplotlib column per panel; (a) repeats once per kappa."""
    cols = []
    for p in panels:
        if p == "a":
            for k in kappas_a:
                cols.append(("a", float(k)))
        else:
            cols.append((p, None))
    return cols


def update_plot(
    results_a, results_b, results_c, results_d,
    panels, d1_vals, favg_vals, df_vals,
    plot_path, task_name, kappas_a,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": False,
    })

    import config as cfg
    ka = [float(x) for x in (kappas_a or [])]
    if "a" in panels and not ka:
        ka = [float(cfg.KAPPA)]
    cols = _sweep_plot_columns(panels, ka)
    n_panels = len(cols)
    fig_w = (SWEEP_PLOT_PX_W / SWEEP_PLOT_DPI) * (n_panels / float(SWEEP_PLOT_BASE_COLS))
    fig_h = SWEEP_PLOT_PX_H / SWEEP_PLOT_DPI
    fig, ax_rows = plt.subplots(
        2, n_panels,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout="constrained",
    )

    fig.suptitle(
        f"Phononic frequency-comb reservoir -- {task_name.replace('_', ' ')}",
        fontsize=12,
        fontweight="bold",
    )

    cmap_comb = ListedColormap(["#b71c1c", "#1b5e20"])
    cmap_comb.set_bad(color=NMSE_HEATMAP_MISSING_HEX)

    for ax_idx, (panel, kappa_col) in enumerate(cols):
        ax_top = ax_rows[0, ax_idx]
        ax_bot = ax_rows[1, ax_idx]

        if panel == "a":
            x_vals, y_vals = d1_vals, favg_vals
            k_round = round(float(kappa_col), 4)
            results = [r for r in results_a if round(float(r["kappa"]), 4) == k_round]
            xlabel = "Delta1 (drive detuning)"
            ylabel = "F_avg (drive amplitude)"
            subtitle = f"(a) F_avg vs Delta1  (delta_f=0; f(n)=F_avg*(0.5+s_n)), kappa={float(kappa_col):g}"
            subtitle_comb = f"(a) Qi analytical comb, kappa={float(kappa_col):g} (Arnold + Eq.(7) u>0 + Eq.(10) + detuning)"
            get_xy = lambda r: (r["delta1"], r["f_avg"])
        elif panel == "b":
            x_vals, y_vals = d1_vals, favg_vals
            results = results_b
            df_b = float(cfg.DELTA_F)
            xlabel = "Delta1 (drive detuning)"
            ylabel = "F_avg (drive amplitude)"
            subtitle = f"(b) F_avg vs Delta1  (fixed delta_f={df_b:g})"
            subtitle_comb = "(b) Same Qi comb map as (a); IVP pulsing uses f in [F_avg-df/2, F_avg+df/2]"
            get_xy = lambda r: (r["delta1"], r["f_avg"])
        elif panel == "c":
            x_vals, y_vals = df_vals, favg_vals
            results = results_c
            xlabel = "delta_f (modulation depth)"
            ylabel = "F_avg (drive amplitude)"
            d1_fix = float(cfg.DELTA1)
            subtitle = f"(c) F_avg vs delta_f  (fixed Delta1={d1_fix:g})"
            subtitle_comb = "(c) Qi comb vs F_avg (indep. of delta_f); IVP pulsing brackets F_avg +/- delta_f/2"
            get_xy = lambda r: (r["delta_f"], r["f_avg"])
        else:
            x_vals, y_vals = d1_vals, df_vals
            results = results_d
            xlabel = "Delta1 (drive detuning)"
            ylabel = "delta_f (modulation depth)"
            f_fix = float(cfg.F_AVG)
            subtitle = f"(d) delta_f vs Delta1  (fixed F_avg={f_fix:g})"
            subtitle_comb = "(d) Qi comb at fixed F_avg; IVP pulsing brackets F_avg +/- delta_f/2"
            get_xy = lambda r: (r["delta1"], r["delta_f"])

        grid = np.full((len(y_vals), len(x_vals)), np.nan)
        for r in results:
            xv, yv = get_xy(r)
            xi = int(np.argmin(np.abs(x_vals - xv)))
            yi = int(np.argmin(np.abs(y_vals - yv)))
            nm = r["nmse"]
            grid[yi, xi] = nm

        extent = [float(x_vals[0]), float(x_vals[-1]),
                  float(y_vals[0]), float(y_vals[-1])]

        with np.errstate(invalid="ignore"):
            grid_log = np.log10(np.clip(grid, 1e-30, None))
        masked = np.ma.masked_invalid(grid_log)
        valid = grid_log[~np.isnan(grid_log)]
        vmin_nmse = float(valid.min()) if len(valid) else -2.5
        vmax_nmse = 0.0
        if vmin_nmse >= vmax_nmse:
            vmin_nmse = vmax_nmse - 0.5

        cmap_nmse = plt.colormaps["viridis_r"].copy()
        cmap_nmse.set_bad(color=NMSE_HEATMAP_MISSING_HEX)

        im = ax_top.imshow(
            masked,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap_nmse,
            vmin=vmin_nmse,
            vmax=vmax_nmse,
            interpolation="nearest",
            filternorm=False,
        )
        cbar = fig.colorbar(im, ax=ax_top, shrink=0.82, aspect=24, pad=0.02)
        cbar.set_label("log10(NMSE)", fontsize=9)

        ax_top.set_xlabel(xlabel)
        ax_top.set_ylabel(ylabel)
        ax_top.set_title(subtitle, fontsize=11, pad=8)
        ax_top.tick_params(axis="both", which="major", labelsize=9)

        comb_grid = np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)
        for r in results:
            xv, yv = get_xy(r)
            xi = int(np.argmin(np.abs(x_vals - xv)))
            yi = int(np.argmin(np.abs(y_vals - yv)))
            comb_grid[yi, xi] = _comb_cell_from_result(r)

        masked_comb = np.ma.masked_invalid(comb_grid)
        im_c = ax_bot.imshow(
            masked_comb,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap_comb,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            filternorm=False,
        )
        cbar_c = fig.colorbar(im_c, ax=ax_bot, shrink=0.82, aspect=24, pad=0.02, ticks=[0.0, 1.0])
        cbar_c.ax.set_yticklabels(["No", "Yes"])
        cbar_c.set_label("Qi 2020 analytical comb", fontsize=9)

        ax_bot.set_xlabel(xlabel)
        ax_bot.set_ylabel(ylabel)
        ax_bot.set_title(subtitle_comb, fontsize=10, pad=8)
        ax_bot.tick_params(axis="both", which="major", labelsize=9)

    fig.savefig(plot_path, dpi=SWEEP_PLOT_DPI, facecolor="white", edgecolor="none")
    plt.close("all")


if __name__ == "__main__":
    import argparse
    import random
    import config as cfg

    parser = argparse.ArgumentParser(
        description="Phononic frequency-comb RC parameter sweep (Qi 2020)"
    )
    parser.add_argument("--jobs", type=int, required=True,
                        help="Target total job count")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Parallel workers (default: all CPU cores)")
    parser.add_argument("--n_test", type=int, default=250,
                        help="Number of test symbols per run (default: 250)")
    parser.add_argument("--task", type=str, default="mackey_glass",
                        choices=["mackey_glass", "lorenz", "rossler"])
    parser.add_argument("--panel", type=str, default="all",
                        choices=["a", "b", "c", "d", "all"],
                        help="Which panel(s) to sweep (default: all)")
    parser.add_argument(
        "--kappa", type=str, default=None,
        help="Panel (a) only: kappa values, e.g. '[-9,-4,0,4,9]'. Default: config KAPPA.",
    )
    args = parser.parse_args()

    panels = ["a", "b", "c", "d"] if args.panel == "all" else [args.panel]

    kappas_a = parse_kappa_list(args.kappa)
    if not kappas_a:
        kappas_a = [float(cfg.KAPPA)]
    n_kappa_a = len(kappas_a) if "a" in panels else 1

    step = compute_step(args.jobs, panels, n_kappa_a=n_kappa_a)
    d1_vals = make_axis(*DELTA1_RANGE, step)
    df_vals = make_axis(*DELTA_F_RANGE, step)
    favg_vals = make_axis(*F_AVG_RANGE, step)

    warmup = cfg.WARMUP_SYMBOLS
    n_symbols = compute_n_symbols(args.n_test, warmup, cfg.TRAIN_FRACTION)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    csv_a_path = os.path.join(out_dir, "sweep_favg_vs_delta1.csv")
    csv_b_path = os.path.join(out_dir, "sweep_favg_vs_delta1_modulated.csv")
    csv_c_path = os.path.join(out_dir, "sweep_favg_vs_deltaf.csv")
    csv_d_path = os.path.join(out_dir, "sweep_deltaf_vs_delta1.csv")
    plot_path = os.path.join(out_dir, "sweep.png")
    error_log = os.path.join(out_dir, "errors.log")

    all_jobs = []
    if "a" in panels:
        jobs_a = build_panel_a(favg_vals, d1_vals, n_symbols, args.n_test, args.task, kappas_a)
        all_jobs.extend(jobs_a)
        print(f"Panel (a): F_AVG x Delta1 x kappa (delta_f=0) -- {len(jobs_a)} jobs  "
              f"({len(favg_vals)} x {len(d1_vals)} x {len(kappas_a)}), kappa={kappas_a}, step={step}")

    if "b" in panels:
        jobs_b = build_panel_b(favg_vals, d1_vals, n_symbols, args.n_test, args.task)
        all_jobs.extend(jobs_b)
        print(f"Panel (b): F_AVG x Delta1 (delta_f={cfg.DELTA_F:g}) -- {len(jobs_b)} jobs  "
              f"({len(favg_vals)} x {len(d1_vals)}), step={step}")

    if "c" in panels:
        jobs_c = build_panel_c(favg_vals, df_vals, n_symbols, args.n_test, args.task)
        all_jobs.extend(jobs_c)
        print(f"Panel (c): F_AVG x delta_f -- {len(jobs_c)} jobs  "
              f"({len(favg_vals)} x {len(df_vals)}), step={step}")

    if "d" in panels:
        jobs_d = build_panel_d(d1_vals, df_vals, n_symbols, args.n_test, args.task)
        all_jobs.extend(jobs_d)
        print(f"Panel (d): Delta1 x delta_f (F_AVG={cfg.F_AVG:g}) -- {len(jobs_d)} jobs  "
              f"({len(d1_vals)} x {len(df_vals)}), step={step}")

    random.shuffle(all_jobs)
    n_total = len(all_jobs)
    n_train_approx = n_symbols - warmup - 1 - args.n_test

    print(f"\n{'=' * 60}")
    print(f"Phononic RC sweep -- panels: {', '.join(panels)}")
    print(f"  --jobs:     {args.jobs}  (actual: {n_total})")
    print(f"  Step:       {step}")
    print(f"  n_test:     {args.n_test}")
    print(f"  n_symbols:  {n_symbols}  ({warmup} warmup + ~{n_train_approx} train + {args.n_test} test)")
    print(f"  Workers:    {args.workers} / {os.cpu_count()} cores")
    print(f"  Task:       {args.task}")
    kappa_note = (
        f"kappa (a)={kappas_a}, kappa (b-d)={cfg.KAPPA}"
        if "a" in panels else f"kappa={cfg.KAPPA}"
    )
    print(f"  Fixed:      {kappa_note}, gamma21={cfg.GAMMA21}  |  "
          f"(a) delta_f=0  |  (b) delta_f={cfg.DELTA_F:g}  |  "
          f"(c) Delta1={cfg.DELTA1}  |  (d) F_AVG={cfg.F_AVG}")
    print(f"{'=' * 60}\n")

    with open(error_log, "w") as ef:
        ef.write(f"Sweep error log -- {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        ef.write(f"Total jobs: {n_total}\n{'=' * 60}\n\n")

    t_start = time.time()
    results_a = []
    results_b = []
    results_c = []
    results_d = []

    live_a = live_b = live_c = live_d = None
    live_a_w = live_b_w = live_c_w = live_d_w = None
    if "a" in panels:
        live_a, live_a_w = open_live_csv(csv_a_path, "a")
    if "b" in panels:
        live_b, live_b_w = open_live_csv(csv_b_path, "b")
    if "c" in panels:
        live_c, live_c_w = open_live_csv(csv_c_path, "c")
    if "d" in panels:
        live_d, live_d_w = open_live_csv(csv_d_path, "d")

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, job): job for job in all_jobs}
            t_first = time.time()
            print(f"All {n_total} jobs submitted. Waiting for first result...")

            with tqdm(total=n_total, unit="run", dynamic_ncols=True) as pbar:
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()

                    if result["panel"] == "a":
                        results_a.append(result)
                        if live_a_w is not None:
                            append_live_csv(live_a_w, live_a, result, "a")
                    elif result["panel"] == "b":
                        results_b.append(result)
                        if live_b_w is not None:
                            append_live_csv(live_b_w, live_b, result, "b")
                    elif result["panel"] == "c":
                        results_c.append(result)
                        if live_c_w is not None:
                            append_live_csv(live_c_w, live_c, result, "c")
                    else:
                        results_d.append(result)
                        if live_d_w is not None:
                            append_live_csv(live_d_w, live_d, result, "d")

                    if "error" in result:
                        with open(error_log, "a") as ef:
                            ef.write(
                                f"FAILED: panel={result['panel']} "
                                f"kappa={result.get('kappa', '')} "
                                f"f={result['f_avg']}, "
                                f"d1={result['delta1']}, df={result['delta_f']}\n"
                                f"{result['error']}\n{'-' * 40}\n\n"
                            )

                    if i == 1:
                        dt = time.time() - t_first
                        est = dt * (n_total - args.workers) / max(args.workers, 1)
                        print(f"\n  First job: {dt:.1f}s -- est. remaining: ~{est / 60:.0f} min\n")

                    if i % 20 == 0:
                        update_plot(
                            results_a, results_b, results_c, results_d, panels,
                            d1_vals, favg_vals, df_vals, plot_path, args.task, kappas_a,
                        )

                    nm = result["nmse"]
                    pbar.set_postfix({
                        "panel": result["panel"],
                        "NMSE": f"{nm:.2e}" if not np.isnan(nm) else "ERR",
                    })
                    pbar.update(1)
    finally:
        if live_a is not None:
            live_a.close()
        if live_b is not None:
            live_b.close()
        if live_c is not None:
            live_c.close()
        if live_d is not None:
            live_d.close()

    if "a" in panels:
        results_a.sort(key=lambda r: (r["kappa"], r["f_avg"], r["delta1"]))
        write_csv(csv_a_path, results_a, "a")

    if "b" in panels:
        results_b.sort(key=lambda r: (r["f_avg"], r["delta1"]))
        write_csv(csv_b_path, results_b, "b")

    if "c" in panels:
        results_c.sort(key=lambda r: (r["f_avg"], r["delta_f"]))
        write_csv(csv_c_path, results_c, "c")

    if "d" in panels:
        results_d.sort(key=lambda r: (r["delta1"], r["delta_f"]))
        write_csv(csv_d_path, results_d, "d")

    update_plot(
        results_a, results_b, results_c, results_d, panels,
        d1_vals, favg_vals, df_vals, plot_path, args.task, kappas_a,
    )

    n_errors = sum(
        1 for r in (results_a + results_b + results_c + results_d) if np.isnan(r["nmse"])
    )
    total_time = time.time() - t_start

    print(f"\nDone in {total_time / 60:.1f} min")
    print(f"Errors: {n_errors} / {n_total}")
    if "a" in panels:
        print(f"CSV (a): {csv_a_path}")
    if "b" in panels:
        print(f"CSV (b): {csv_b_path}")
    if "c" in panels:
        print(f"CSV (c): {csv_c_path}")
    if "d" in panels:
        print(f"CSV (d): {csv_d_path}")
    print(f"Plot:    {plot_path}")
