"""
Phononic coupled-mode solver implementing Eqs. (5) and (6) from Qi et al. (2020).

    dw1/ds = -iF - (1 + i*Delta1)*w1 + i*w2^2           (Eq. 5)
    dw2/ds = -(gamma21 + i*Delta2)*w2 + 2i*w1*conj(w2)   (Eq. 6)

w1, w2 are normalised complex mode amplitudes; s = gamma1*t is normalised time.
The frequency comb forms when a Hopf bifurcation produces periodic envelope oscillations.
"""

import numpy as np
from scipy.integrate import solve_ivp
import config


class PhononicSolver:
    """
    Solver for the phononic coupled-mode equations.

    Parameters
    ----------
    gamma21 : float
        Damping ratio gamma2/gamma1.
    delta1 : float
        Drive detuning (normalised).
    delta2 : float
        2:1 resonance detuning (normalised).
    n_virtual : int
        Number of time samples (virtual nodes) per symbol.
    n_fft : int
        Number of FFT spectral bins to keep per mode.
    method : str
        Integration method for solve_ivp.
    rtol, atol : float
        Solver tolerances.
    """

    def __init__(
        self,
        gamma21=config.GAMMA21,
        delta1=config.DELTA1,
        delta2=config.DELTA2,
        n_virtual=config.N_VIRTUAL,
        n_fft=config.N_FFT_FEATURES,
        method=config.SOLVER_METHOD,
        rtol=config.SOLVER_RTOL,
        atol=config.SOLVER_ATOL,
    ):
        self.gamma21 = gamma21
        self.delta1 = delta1
        self.delta2 = delta2
        self.n_virtual = n_virtual
        self.n_fft = n_fft
        self.method = method
        self.rtol = rtol
        self.atol = atol

        self._lin1 = -(1.0 + 1j * delta1)
        self._lin2 = -(gamma21 + 1j * delta2)

    def _rhs(self, tau, y, f_val):
        """
        RHS of the coupled-mode ODEs.
        y = [Re(psi1), Im(psi1), Re(psi2), Im(psi2)]
        """
        psi1 = y[0] + 1j * y[1]
        psi2 = y[2] + 1j * y[3]

        # Eq. 5: dw1/ds = -iF - (1 + i*Delta1)*w1 + i*w2^2
        dpsi1 = -1j * f_val + self._lin1 * psi1 + 1j * psi2**2

        # Eq. 6: dw2/ds = -(gamma21 + i*Delta2)*w2 + 2i*w1*conj(w2)
        dpsi2 = self._lin2 * psi2 + 2j * psi1 * np.conj(psi2)

        return np.array([dpsi1.real, dpsi1.imag, dpsi2.real, dpsi2.imag])

    def evolve_one_symbol(self, psi_init, f_val, t_symbol):
        """
        Evolve the coupled modes for one symbol period.

        Parameters
        ----------
        psi_init : ndarray, shape (4,)
            Initial state [Re(psi1), Im(psi1), Re(psi2), Im(psi2)].
        f_val : float
            Drive amplitude for this symbol.
        t_symbol : float
            Symbol duration in normalised time.

        Returns
        -------
        t_samples : ndarray, shape (n_virtual,)
        psi_samples : ndarray, shape (n_virtual, 4)
        psi_final : ndarray, shape (4,)
        """
        t_eval = np.linspace(0, t_symbol, self.n_virtual + 1)[1:]

        sol = solve_ivp(
            fun=lambda tau, y: self._rhs(tau, y, f_val),
            t_span=(0, t_symbol),
            y0=psi_init,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        psi_samples = sol.y.T  # (n_virtual, 4)
        psi_final = psi_samples[-1]

        return t_eval, psi_samples, psi_final

    def extract_features(self, psi_samples):
        """
        Extract feature vector from time-sampled mode amplitudes.

        Combines:
        - Virtual-node powers: |psi1(tk)|^2, |psi2(tk)|^2 for k=1..n_virtual
        - FFT magnitudes of psi1(t) and psi2(t) (comb lines), first n_fft bins each

        Parameters
        ----------
        psi_samples : ndarray, shape (n_virtual, 4)

        Returns
        -------
        features : ndarray, shape (n_features,)
        """
        psi1 = psi_samples[:, 0] + 1j * psi_samples[:, 1]
        psi2 = psi_samples[:, 2] + 1j * psi_samples[:, 3]

        power1 = np.abs(psi1) ** 2
        power2 = np.abs(psi2) ** 2

        # Use full fft (not rfft) since psi is complex-valued
        fft1 = np.abs(np.fft.fft(psi1))[: self.n_fft]
        fft2 = np.abs(np.fft.fft(psi2))[: self.n_fft]

        if len(fft1) < self.n_fft:
            fft1 = np.pad(fft1, (0, self.n_fft - len(fft1)))
        if len(fft2) < self.n_fft:
            fft2 = np.pad(fft2, (0, self.n_fft - len(fft2)))

        return np.concatenate([power1, power2, fft1, fft2])

    def simulate(self, f_sequence, t_symbol=config.SYMBOL_DURATION_TAU,
                 psi_init=None, progress=False):
        """
        Run the phononic solver for a full drive sequence.

        Parameters
        ----------
        f_sequence : ndarray, shape (n_symbols,)
            Drive amplitude per symbol.
        t_symbol : float
            Duration of each symbol in normalised time.
        psi_init : ndarray, shape (4,) or None
            Initial state. If None, uses a small random seed.
        progress : bool
            Print progress every 500 symbols.

        Returns
        -------
        feature_matrix : ndarray, shape (n_symbols, n_features)
        psi_final : ndarray, shape (4,)
        """
        n_symbols = len(f_sequence)

        if psi_init is None:
            rng = np.random.default_rng()
            psi_init = rng.normal(0, 0.01, size=4)

        dummy_samples = np.zeros((self.n_virtual, 4))
        n_features = len(self.extract_features(dummy_samples))

        feature_matrix = np.zeros((n_symbols, n_features))
        psi = psi_init.copy()

        for i in range(n_symbols):
            _, psi_samples, psi = self.evolve_one_symbol(psi, f_sequence[i], t_symbol)
            feature_matrix[i] = self.extract_features(psi_samples)

            if progress and (i + 1) % 500 == 0:
                print(f"  Symbol {i+1}/{n_symbols}")

        return feature_matrix, psi
