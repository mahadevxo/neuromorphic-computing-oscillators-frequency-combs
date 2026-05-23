"""
Phononic comb indicators aligned with Qi et al. (2020) APL 117, 183503.

1) Analytical (Qi): Arnold tongue (Eq. (7) discriminant >= 0), positive |w2P|^2
   on the P-branch, full Eq. (10) inequality, and the detuning boundary
   2*Delta1*Delta2 <= -(1 + Delta1^2 + 2*gamma21).

2) Numerical pulsing: IVP std(|w2|^2)/mean(|w2|^2) with a floor on mean |w2|^2
   so noise at vanishing amplitude does not trigger false positives.

Uses the same normalised Eqs. (5)-(6) as PhononicSolver.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.integrate import solve_ivp

DEFAULT_MODULATION_THRESHOLD = 0.01
DEFAULT_MIN_MEAN_POWER2 = 1e-4


def comb_eq10_detuning_ok(delta1: float, delta2: float, gamma21: float) -> bool:
    """
    Detuning boundary from Qi et al. (2020) (after Eq. (10)):
    2*Delta1*Delta2 <= -(1 + Delta1^2 + 2*gamma21).
    """
    return 2.0 * float(delta1) * float(delta2) <= -(
        1.0 + float(delta1) ** 2 + 2.0 * float(gamma21)
    )


def arnold_tongue_minimum_f(delta1: float, delta2: float, gamma21: float) -> float:
    """
    Minimum drive amplitude for real P-branch solutions (Eq. (7) discriminant >= 0):
    f >= (1/2) * gamma21 * |Delta1 + Delta2|  (Qi 2020, text below Eq. (7)).
    """
    g = float(gamma21)
    return 0.5 * g * abs(float(delta1) + float(delta2))


def _eq7_coefficients(
    f: float, delta1: float, delta2: float, gamma21: float
) -> Tuple[float, float, float]:
    """u^2 + B*u + C = 0 for u = |w2P|^2 (Eq. (7)). Returns (B, C, discriminant)."""
    g = float(gamma21)
    d1, d2 = float(delta1), float(delta2)
    fv = float(f)
    B = g - d1 * d2
    C = 0.25 * (1.0 + d1**2) * (g**2 + d2**2) - fv**2
    disc = B * B - 4.0 * C
    return B, C, disc


def steady_state_w2p_squared(
    f: float, delta1: float, delta2: float, gamma21: float
) -> Tuple[Optional[float], float, np.ndarray]:
    """
    Largest positive root u = |w2P|^2 of Eq. (7), if any.

    Returns
    -------
    u_max : float or None
        Physical P-branch |w2P|^2; None if no positive real root.
    disc : float
    roots : ndarray
    """
    B, C, disc = _eq7_coefficients(f, delta1, delta2, gamma21)
    if disc < 0.0:
        return None, disc, np.array([], dtype=float)
    s = np.sqrt(disc)
    r0 = 0.5 * (-B - s)
    r1 = 0.5 * (-B + s)
    roots = np.array([r0, r1], dtype=float)
    positive = roots[roots > 0.0]
    if positive.size == 0:
        return None, disc, roots
    return float(np.max(positive)), disc, roots


def eq10_minimum_w2p_sq(delta1: float, delta2: float, gamma21: float) -> Optional[float]:
    """
    RHS of Qi et al. Eq. (10) as a lower bound on |w2P|^2:

    |w2P|^2 >= -gamma21*(1+Delta1^2)*(1+Delta1^2+4*gamma21*(1+gamma21))
               / (4*(1+gamma21)^2*(1+Delta1^2+2*gamma21+2*Delta1*Delta2))

    Returns None if the denominator is zero.
    """
    g = float(gamma21)
    d1, d2 = float(delta1), float(delta2)
    den = 1.0 + d1**2 + 2.0 * g + 2.0 * d1 * d2
    if abs(den) < 1e-14:
        return None
    num = g * (1.0 + d1**2) * (1.0 + d1**2 + 4.0 * g * (1.0 + g))
    return -num / (4.0 * (1.0 + g) ** 2 * den)


def comb_qi_analytical(
    f: float,
    delta1: float,
    delta2: float,
    gamma21: float,
    *,
    f_tol: float = 1e-9,
) -> Tuple[bool, dict]:
    """
    Qi 2020 comb existence (analytical): inside Arnold tongue, positive P-branch
    |w2P|^2, detuning boundary, and Eq. (10) satisfied.

    Returns (ok, diagnostics dict).
    """
    g = float(gamma21)
    d1, d2 = float(delta1), float(delta2)
    fv = float(f)

    f_min_arnold = arnold_tongue_minimum_f(d1, d2, g)
    arnold_ok = fv + f_tol >= f_min_arnold

    u_p, disc, roots = steady_state_w2p_squared(fv, d1, d2, g)
    has_p_branch = u_p is not None and u_p > f_tol

    detuning_ok = comb_eq10_detuning_ok(d1, d2, g)

    rhs10 = eq10_minimum_w2p_sq(d1, d2, g)
    if rhs10 is None:
        eq10_ok = False
    elif has_p_branch:
        eq10_ok = u_p >= rhs10 - 1e-9  # type: ignore
    else:
        eq10_ok = False

    ok = bool(arnold_ok and has_p_branch and detuning_ok and eq10_ok)

    return ok, {
        "arnold_ok": arnold_ok,
        "f_min_arnold": f_min_arnold,
        "disc_eq7": disc,
        "roots_eq7": roots,
        "u_w2p_sq": u_p,
        "detuning_ok": detuning_ok,
        "eq10_rhs": rhs10,
        "eq10_ok": eq10_ok,
    }


def power_modulation_index(power: np.ndarray) -> float:
    """std/mean on nonnegative series; 0 if mean vanishes."""
    p = np.asarray(power, dtype=float)
    m = float(np.mean(p))
    if m < 1e-30:
        return 0.0
    return float(np.std(p) / m)


def _rhs(tau, y, f_val, delta1, delta2, gamma21):
    lin1 = -(1.0 + 1j * delta1)
    lin2 = -(gamma21 + 1j * delta2)
    psi1 = y[0] + 1j * y[1]
    psi2 = y[2] + 1j * y[3]
    dpsi1 = -1j * f_val + lin1 * psi1 + 1j * psi2**2
    dpsi2 = lin2 * psi2 + 2j * psi1 * np.conj(psi2)
    return np.array([dpsi1.real, dpsi1.imag, dpsi2.real, dpsi2.imag], dtype=float)


def _p_branch_initial_guess(
    f_val: float, delta1: float, delta2: float, gamma21: float
) -> np.ndarray:
    """
    Real 4-vector [Re w1, Im w1, Re w2, Im w2] at the P-branch steady state.

    From Eq. (6) steady state: w1 = (gamma21 + i*Delta2) * exp(2i*alpha) / (2i)
    Phase alpha satisfies: exp(2i*alpha) = iF / (P + iQ)
      where P = -(Delta2 + gamma21*Delta1)/2
            Q = u_p + (gamma21 - Delta1*Delta2)/2
    """
    u_p, _, _ = steady_state_w2p_squared(f_val, delta1, delta2, gamma21)
    if u_p is None or u_p <= 0.0:
        return np.array([0.01, 0.01, 0.01, 0.0], dtype=float)
    g = float(gamma21)
    d1, d2, f = float(delta1), float(delta2), float(f_val)
    w2_mag = np.sqrt(u_p)

    P = -(d2 + g * d1) / 2.0
    Q = u_p + (g - d1 * d2) / 2.0
    alpha = 0.5 * np.arctan2(P, Q)

    w2 = w2_mag * np.exp(1j * alpha)
    w1 = (g + 1j * d2) * np.exp(2j * alpha) / (2j)
    return np.array([w1.real, w1.imag, w2.real, w2.imag], dtype=float)


@dataclass
class CombEvaluation:
    """Comb flags and metrics for one (f, Delta1, Delta2, gamma21) point."""

    comb_qi: bool
    comb_pulsing: bool
    mod_ratio: float
    mean_power2: float
    diag: dict


def evaluate_comb_at_point(
    f_drive: float,
    delta1: float,
    delta2: float,
    gamma21: float,
    *,
    t_max: float = 50.0,
    n_eval: int = 5000,
    transient_frac: float = 0.25,
    threshold: float = DEFAULT_MODULATION_THRESHOLD,
    min_mean_power2: float = DEFAULT_MIN_MEAN_POWER2,
    rng: Optional[np.random.Generator] = None,
    use_p_branch_seed: bool = True,
) -> CombEvaluation:
    """
    Analytical Qi comb (comb_qi) and IVP-based pulsing (comb_pulsing).

    comb_pulsing is True when std/mean of |w2|^2 exceeds threshold and
    mean |w2|^2 >= min_mean_power2. It is independent of comb_qi so plots
    can compare theory vs observation.
    """
    comb_qi, diag = comb_qi_analytical(f_drive, delta1, delta2, gamma21)

    if rng is None:
        rng = np.random.default_rng(0)
    if use_p_branch_seed:
        y0 = _p_branch_initial_guess(f_drive, delta1, delta2, gamma21)
        y0 = y0 + rng.normal(0, 0.002, size=4)
    else:
        y0 = rng.normal(0, 0.01, size=4)

    t_eval = np.linspace(0.0, t_max, n_eval)
    sol = solve_ivp(
        lambda tau, y: _rhs(tau, y, f_drive, delta1, delta2, gamma21),
        (0.0, t_max),
        y0,
        method="RK45",
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
    )

    mean_p2 = 0.0
    ratio = 0.0
    if sol.success and sol.y.shape[1] >= 2:
        psi2 = sol.y[2] + 1j * sol.y[3]
        power2 = np.abs(psi2) ** 2
        n0 = int(transient_frac * len(power2))
        steady = power2[n0:] if n0 < len(power2) - 2 else power2[len(power2) // 4:]
        mean_p2 = float(np.mean(steady))
        ratio = power_modulation_index(steady)

    comb_pulsing = bool(ratio > threshold and mean_p2 >= min_mean_power2)

    diag = {
        **diag,
        "mean_power2_steady": mean_p2,
        "mod_ratio_raw": ratio,
    }
    return CombEvaluation(
        comb_qi=comb_qi,
        comb_pulsing=comb_pulsing,
        mod_ratio=ratio,
        mean_power2=mean_p2,
        diag=diag,
    )


def comb_numerical_at_point(
    f_avg: float,
    delta1: float,
    delta2: float,
    gamma21: float,
    *,
    t_max: float = 50.0,
    n_eval: int = 5000,
    transient_frac: float = 0.25,
    threshold: float = DEFAULT_MODULATION_THRESHOLD,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bool, float]:
    """Backward-compatible wrapper. Returns (comb_pulsing, mod_ratio)."""
    ev = evaluate_comb_at_point(
        f_avg, delta1, delta2, gamma21,
        t_max=t_max, n_eval=n_eval, transient_frac=transient_frac,
        threshold=threshold, rng=rng,
    )
    return ev.comb_pulsing, ev.mod_ratio
