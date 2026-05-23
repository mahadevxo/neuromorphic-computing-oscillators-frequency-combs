"""
Chaotic time-series generators for reservoir computing benchmarks.

All return 1-D arrays normalised to [0, 1]:
  - Mackey-Glass (delay-differential equation)
  - Lorenz attractor
  - Rossler attractor
"""

import numpy as np
from scipy.integrate import solve_ivp
import config


def mackey_glass(n_points, tau=config.MG_TAU, beta=config.MG_BETA,
                 gamma=config.MG_GAMMA_MG, n_exp=config.MG_N_EXP,
                 dt=config.MG_DT, warmup=500):
    """
    Generate a Mackey-Glass time series via Euler integration of:

        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

    Parameters
    ----------
    n_points : int
        Number of output samples.
    tau : int
        Delay parameter (default 17).
    warmup : int
        Transient samples to discard.

    Returns
    -------
    series : ndarray, shape (n_points,)  -- normalised to [0, 1]
    """
    total = n_points + warmup
    history_len = int(tau / dt) + 1

    x = np.ones(history_len) * 1.2
    result = np.zeros(total)

    for i in range(total):
        x_tau = x[0]
        x_now = x[-1]
        dx = beta * x_tau / (1.0 + x_tau ** n_exp) - gamma * x_now
        x_new = x_now + dt * dx
        x = np.roll(x, -1)
        x[-1] = x_new
        result[i] = x_new

    series = result[warmup:]
    s_min, s_max = series.min(), series.max()
    if s_max > s_min:
        series = (series - s_min) / (s_max - s_min)
    return series


def lorenz(n_points, sigma=config.LZ_SIGMA, rho=config.LZ_RHO,
           beta=config.LZ_BETA_LZ, dt=config.LZ_DT,
           subsample=config.LZ_SUBSAMPLE, warmup=1000):
    """
    Generate a Lorenz attractor time series (x-component).

        dx/dt = sigma*(y - x)
        dy/dt = x*(rho - z) - y
        dz/dt = x*y - beta*z

    Parameters
    ----------
    n_points : int
        Number of output samples.
    subsample : int
        Keep every Nth integration step.
    warmup : int
        Transient steps to discard (before subsampling).

    Returns
    -------
    series : ndarray, shape (n_points,)  -- normalised x-component in [0, 1]
    """
    total_steps = n_points * subsample + warmup

    def rhs(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t_span = (0, total_steps * dt)
    t_eval = np.linspace(*t_span, total_steps)

    sol = solve_ivp(rhs, t_span, [1.0, 1.0, 1.0], t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-10)

    x = sol.y[0][warmup:]
    x = x[::subsample][:n_points]

    s_min, s_max = x.min(), x.max()
    if s_max > s_min:
        x = (x - s_min) / (s_max - s_min)
    return x


def rossler(n_points, a=config.RS_A, b=config.RS_B, c=config.RS_C,
            dt=config.RS_DT, subsample=config.RS_SUBSAMPLE, warmup=2000):
    """
    Generate a Rossler attractor time series (x-component).

        dx/dt = -y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x - c)

    Parameters
    ----------
    n_points : int
        Number of output samples.

    Returns
    -------
    series : ndarray, shape (n_points,)  -- normalised x-component in [0, 1]
    """
    total_steps = n_points * subsample + warmup

    def rhs(t, state):
        x, y, z = state
        return [-(y + z), x + a * y, b + z * (x - c)]

    t_span = (0, total_steps * dt)
    t_eval = np.linspace(*t_span, total_steps)

    sol = solve_ivp(rhs, t_span, [1.0, 1.0, 0.0], t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-10)

    x = sol.y[0][warmup:]
    x = x[::subsample][:n_points]

    s_min, s_max = x.min(), x.max()
    if s_max > s_min:
        x = (x - s_min) / (s_max - s_min)
    return x


def get_time_series(task_name, n_points=config.N_SYMBOLS):
    """
    Generate a time series by name.

    Parameters
    ----------
    task_name : str
        One of 'mackey_glass', 'lorenz', 'rossler'.
    n_points : int

    Returns
    -------
    series : ndarray, shape (n_points,)  -- normalised to [0, 1]
    """
    generators = {
        "mackey_glass": mackey_glass,
        "lorenz": lorenz,
        "rossler": rossler,
    }
    if task_name not in generators:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(generators)}")
    return generators[task_name](n_points)
