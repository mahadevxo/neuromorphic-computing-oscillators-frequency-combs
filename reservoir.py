"""
Reservoir Computing pipeline using the phononic frequency comb.

Orchestrates:
  1. Input encoding:  s(n) -> f(n) drive amplitude
  2. Reservoir run:   phononic solver -> time-sampled features
  3. Feature processing: log-scale, standardisation
  4. Ridge regression:  train / predict
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso

import config
from phononic_solver import PhononicSolver
from metrics import nmse


class ReservoirComputer:
    """
    Full reservoir computing pipeline using the phononic coupled-mode system.

    Parameters
    ----------
    f_avg : float
        Average drive amplitude.
    delta_f : float
        Drive modulation depth.
    n_virtual : int
        Virtual nodes (time samples per symbol).
    n_fft : int
        FFT features per mode.
    reg_lambda : float
        Ridge regression regularisation parameter.
    warmup : int
        Number of initial symbols to discard (transient).
    precondition : int
        Number of constant-drive symbols to run before data.
    t_symbol : float
        Symbol duration in normalised time.
    solver_kwargs : dict
        Extra arguments for PhononicSolver.
    """

    def __init__(
        self,
        f_avg=config.F_AVG,
        delta_f=config.DELTA_F,
        n_virtual=config.N_VIRTUAL,
        n_fft=config.N_FFT_FEATURES,
        reg_lambda=config.REG_LAMBDA,
        warmup=config.WARMUP_SYMBOLS,
        precondition=config.PRECONDITION_SYMBOLS,
        t_symbol=config.SYMBOL_DURATION_TAU,
        **solver_kwargs,
    ):
        self.f_avg = f_avg
        self.delta_f = delta_f
        self.n_virtual = n_virtual
        self.n_fft = n_fft
        self.reg_lambda = reg_lambda
        self.warmup = warmup
        self.precondition = precondition
        self.t_symbol = t_symbol

        self.solver = PhononicSolver(
            n_virtual=n_virtual,
            n_fft=n_fft,
            **solver_kwargs,
        )
        _reg = config.REGRESSION.lower()
        if _reg == "lasso":
            self.regressor = Lasso(alpha=reg_lambda, fit_intercept=True, max_iter=10000)
        else:
            self.regressor = Ridge(alpha=reg_lambda, fit_intercept=True)
        self._trained = False

        self._feat_mean = None
        self._feat_std = None
        self._feat_mask = None

    def encode_input(self, series):
        """
        Map normalised time series s(n) in [0,1] to drive amplitudes f(n).

        If delta_f > 0: f(n) = f_avg + delta_f * (s(n) - 0.5)
        If delta_f == 0: f(n) = f_avg * (0.5 + s(n))
        """
        s = np.asarray(series, dtype=float)
        if self.delta_f == 0.0:
            return self.f_avg * (0.5 + s)
        return self.f_avg + self.delta_f * (s - 0.5)

    def _precondition_cavity(self, psi_init=None):
        """Run at constant drive to establish steady state / comb."""
        f_const = np.full(self.precondition, self.f_avg)
        _, psi_final = self.solver.simulate(
            f_const, self.t_symbol, psi_init=psi_init, progress=False
        )
        return psi_final

    def run_reservoir(self, f_sequence, psi_init=None, progress=True):
        """
        Drive the reservoir with a sequence of drive amplitudes.

        1. Preconditions the cavity
        2. Runs the data sequence
        3. Discards warmup
        4. Applies log10 compression and standardisation

        Parameters
        ----------
        f_sequence : ndarray, shape (n_symbols,)
        psi_init : ndarray or None
        progress : bool

        Returns
        -------
        features : ndarray, shape (n_symbols - warmup, n_features)
        """
        if progress:
            print(f"  Preconditioning cavity ({self.precondition} symbols)...")
        psi = self._precondition_cavity(psi_init)

        if progress:
            print(f"  Running reservoir ({len(f_sequence)} symbols)...")
        raw_features, _ = self.solver.simulate(
            f_sequence, self.t_symbol, psi_init=psi, progress=progress
        )

        features = raw_features[self.warmup:]

        # Log-compression: ODE powers/FFT magnitudes span many orders of magnitude
        features = np.log10(np.abs(features) + 1e-30)

        return features

    def _normalise_features(self, features, fit=False):
        """Standardise (zero mean, unit variance) and drop dead features."""
        if fit:
            self._feat_mean = features.mean(axis=0)
            self._feat_std = features.std(axis=0)
            self._feat_mask = self._feat_std > 1e-10
            if not np.any(self._feat_mask):
                self._feat_mask[0] = True

        features = features[:, self._feat_mask]
        mean = self._feat_mean[self._feat_mask]  # type: ignore
        std = self._feat_std[self._feat_mask]    # type: ignore

        return (features - mean) / (std + 1e-30)

    def train(self, features, targets):
        """Train the Ridge/Lasso output layer."""
        features_norm = self._normalise_features(features, fit=True)
        self.regressor.fit(features_norm, targets)
        self._trained = True

    def predict(self, features):
        """Predict using the trained output layer."""
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        features_norm = self._normalise_features(features, fit=False)
        return self.regressor.predict(features_norm)

    def run_pipeline(self, series, train_frac=config.TRAIN_FRACTION,
                     n_test=None, progress=True):
        """
        Complete single-step RC pipeline.

        Parameters
        ----------
        series : ndarray, shape (n_symbols,)
            Input time series normalised to [0, 1].
        train_frac : float
            Training fraction (used when n_test is None).
        n_test : int or None
            Explicit number of test symbols.
        progress : bool

        Returns
        -------
        results : dict with keys:
            train_nmse, test_nmse,
            y_train_true, y_train_pred,
            y_test_true, y_test_pred,
            features, n_features_active
        """
        n = len(series)
        if progress:
            print(f"Pipeline: {n} symbols, {self.warmup} warmup, "
                  f"{train_frac:.0%} train")

        f_seq = self.encode_input(series)
        features = self.run_reservoir(f_seq, progress=progress)

        # Predict s(n+1) from features at step n
        effective_series = series[self.warmup:]
        targets = effective_series[1:]
        features = features[:-1]

        n_eff = len(targets)
        if n_test is not None:
            n_train = n_eff - n_test
        else:
            n_train = int(n_eff * train_frac)

        X_train = features[:n_train]
        Y_train = targets[:n_train]
        X_test = features[n_train:]
        Y_test = targets[n_train:]

        if progress:
            print(f"  Training on {n_train} samples, testing on {n_eff - n_train}...")
        self.train(X_train, Y_train)

        Y_train_pred = self.predict(X_train)
        Y_test_pred = self.predict(X_test)

        train_nmse = nmse(Y_train, Y_train_pred)
        test_nmse = nmse(Y_test, Y_test_pred)

        if progress:
            print(f"  Train NMSE: {train_nmse:.6e}")
            print(f"  Test  NMSE: {test_nmse:.6e}")
            print(f"  log10(Test NMSE): {np.log10(test_nmse + 1e-30):.3f}")

        return {
            "train_nmse": train_nmse,
            "test_nmse": test_nmse,
            "y_train_true": Y_train,
            "y_train_pred": Y_train_pred,
            "y_test_true": Y_test,
            "y_test_pred": Y_test_pred,
            "features": features,
            "n_features_active": int(self._feat_mask.sum()),  # type: ignore
        }
