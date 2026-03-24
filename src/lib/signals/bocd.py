# -*- coding: utf-8 -*-
"""Bayesian online changepoint detection (BOCD) utilities.

Implements a discrete run-length posterior :math:`P(r_t \mid x_{1:t})` with a
conjugate Student-t predictive for Gaussian-like observations and a constant
prior hazard of a new changepoint.

"""

from __future__ import annotations

import functools
from typing import Callable, Optional
import numpy as np
import pandas as pd
from scipy import stats


def constant_hazard(lam: float, r: np.ndarray) -> np.ndarray:
    """Constant hazard: probability of a changepoint per run length is ``1/lam``.

    Parameters
    ----------
    lam : float
        Expected run length between changepoints (must be > 0).
    r : ndarray
        Run-length indices (shape broadcast with internal use); only ``r.shape``
        is required for the output shape.

    Returns
    -------
    ndarray
        Array of shape ``r.shape`` filled with ``1/lam``.
    """
    return np.full(r.shape, 1.0 / lam, dtype=np.float64)


class ConstantHazard:
    """Constant hazard callable; enables optimized BOCD updates (no ``H`` vector alloc).

    Parameters
    ----------
    lam : float
        Same as ``lam`` in :func:`constant_hazard`.
    """

    __slots__ = ("lam", "inv_lam")

    def __init__(self, lam: float) -> None:
        self.lam = float(lam)
        self.inv_lam = 1.0 / self.lam

    def __call__(self, r: np.ndarray) -> np.ndarray:
        return np.full(r.shape, self.inv_lam, dtype=np.float64)


def _constant_hazard_inv_lam(hazard_function: Callable) -> Optional[float]:
    """Return ``1/lam`` if ``hazard_function`` is effectively constant hazard."""
    if isinstance(hazard_function, ConstantHazard):
        return hazard_function.inv_lam
    if isinstance(hazard_function, functools.partial):
        if hazard_function.func is constant_hazard and len(hazard_function.args) == 1:
            if not hazard_function.keywords:
                return 1.0 / float(hazard_function.args[0])
    return None


class StudentT:
    """Conjugate Normal-Inverse-Gamma / Student-t predictive for BOCD observations.

    Maintains arrays of parameters indexed by hypothetical run length after each
    sequential update. Used as ``observation_likelihood`` in :class:`BOCD`.

    Parameters follow the Normal-Inverse-Gamma parameterization where the marginal
    predictive for a new point is Student-t with degrees of freedom ``2*alpha``.

    Internally uses two sets of preallocated buffers swapped each update to avoid
    ``O(t^2)`` work from repeated ``np.concatenate`` over time ``t``.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        kappa: float,
        mu: float,
        capacity: int = 8192,
    ) -> None:
        """Initialize prior hyperparameters and allocation.

        Parameters
        ----------
        alpha, beta, kappa, mu : float
            NIG / Student-t prior hyperparameters (see ``pdf`` for predictive form).
        capacity : int
            Initial maximum number of run-length hypotheses; grows automatically if
            :meth:`BOCD.expand_matrix` requires more rows than ``capacity``.
        """
        self._capacity = max(int(capacity), 4)
        self._mu0 = float(mu)
        self._kappa0 = float(kappa)
        self._alpha0 = float(alpha)
        self._beta0 = float(beta)

        self._mu_a = np.zeros(self._capacity, dtype=np.float64)
        self._mu_b = np.zeros(self._capacity, dtype=np.float64)
        self._kappa_a = np.zeros(self._capacity, dtype=np.float64)
        self._kappa_b = np.zeros(self._capacity, dtype=np.float64)
        self._alpha_a = np.zeros(self._capacity, dtype=np.float64)
        self._alpha_b = np.zeros(self._capacity, dtype=np.float64)
        self._beta_a = np.zeros(self._capacity, dtype=np.float64)
        self._beta_b = np.zeros(self._capacity, dtype=np.float64)

        self._mu_a[0] = self._mu0
        self._kappa_a[0] = self._kappa0
        self._alpha_a[0] = self._alpha0
        self._beta_a[0] = self._beta0

        self._use_a = True
        self.n = 1

        self._scale_scratch = np.zeros(self._capacity, dtype=np.float64)

        # Back-compat: original code exposed these as length-1 arrays
        self.alpha0 = np.array([alpha], dtype=np.float64)
        self.beta0 = np.array([beta], dtype=np.float64)
        self.kappa0 = np.array([kappa], dtype=np.float64)
        self.mu0 = np.array([mu], dtype=np.float64)

    def _active(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._use_a:
            return self._mu_a, self._kappa_a, self._alpha_a, self._beta_a
        return self._mu_b, self._kappa_b, self._alpha_b, self._beta_b

    def _inactive(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._use_a:
            return self._mu_b, self._kappa_b, self._alpha_b, self._beta_b
        return self._mu_a, self._kappa_a, self._alpha_a, self._beta_a

    def _swap(self) -> None:
        self._use_a = not self._use_a

    @property
    def mu(self) -> np.ndarray:
        mu, _, _, _ = self._active()
        return mu[: self.n]

    @property
    def kappa(self) -> np.ndarray:
        _, kappa, _, _ = self._active()
        return kappa[: self.n]

    @property
    def alpha(self) -> np.ndarray:
        _, _, alpha, _ = self._active()
        return alpha[: self.n]

    @property
    def beta(self) -> np.ndarray:
        _, _, _, beta = self._active()
        return beta[: self.n]

    def _grow_capacity(self, min_cap: int) -> None:
        if min_cap <= self._capacity:
            return
        new_cap = max(min_cap, self._capacity * 2)
        for name in ("_mu_a", "_mu_b", "_kappa_a", "_kappa_b", "_alpha_a", "_alpha_b", "_beta_a", "_beta_b"):
            old = getattr(self, name)
            new = np.zeros(new_cap, dtype=np.float64)
            new[: self._capacity] = old
            setattr(self, name, new)
        self._scale_scratch = np.zeros(new_cap, dtype=np.float64)
        self._capacity = new_cap

    def pdf(self, data: float) -> np.ndarray:
        """Predictive Student-t density of ``data`` for each stored run-length state.

        Returns a **new** length-``n`` vector (SciPy allocates); hot path still
        benefits from contiguous slices and no ``concatenate``.
        """
        mu, kappa, alpha, beta = self._active()
        n = self.n
        m = mu[:n]
        k = kappa[:n]
        a = alpha[:n]
        b = beta[:n]
        scale = self._scale_scratch[:n]
        # scale^2 = beta * (kappa+1) / (alpha * kappa); guard degenerate kappa/alpha
        ak = a * k
        np.divide(b * (k + 1.0), np.maximum(ak, 1e-300), out=scale)
        np.sqrt(scale, out=scale)
        return stats.t.pdf(
            x=data,
            df=2.0 * a,
            loc=m,
            scale=scale,
        )

    def update_theta(self, data: float) -> None:
        """Bayesian update: extend run-length parameters after observing ``data``."""
        mu_in, kappa_in, alpha_in, beta_in = self._active()
        mu_out, kappa_out, alpha_out, beta_out = self._inactive()

        n = self.n
        # Tail updates (vectorized over current hypotheses)
        mu_out[0] = self._mu0
        kappa_out[0] = self._kappa0
        alpha_out[0] = self._alpha0
        beta_out[0] = self._beta0

        np.divide(
            kappa_in[:n] * mu_in[:n] + data,
            kappa_in[:n] + 1.0,
            out=mu_out[1 : n + 1],
        )
        np.add(kappa_in[:n], 1.0, out=kappa_out[1 : n + 1])
        np.add(alpha_in[:n], 0.5, out=alpha_out[1 : n + 1])
        diff = mu_in[:n] - data
        np.multiply(kappa_in[:n], diff * diff, out=beta_out[1 : n + 1])
        np.divide(beta_out[1 : n + 1], 2.0 * (kappa_in[:n] + 1.0), out=beta_out[1 : n + 1])
        np.add(beta_in[:n], beta_out[1 : n + 1], out=beta_out[1 : n + 1])

        self.n = n + 1
        self._swap()


class BOCD:
    """Online Bayesian changepoint detection on a scalar stream.

    Maintains a run-length posterior matrix ``R`` where ``R[r, t]`` is (proportional
    to) the probability that the current segment has length ``r`` after seeing
    ``t`` observations. A changepoint is heuristically flagged when the posterior
    mass at run length 1 exceeds a fixed threshold (0.35).

    Parameters
    ----------
    hazard_function : callable
        ``hazard_function(r_array) -> H`` with values in ``[0, 1]``; prior
        probability of reset as a function of run length (e.g. :class:`ConstantHazard`
        or ``functools.partial(constant_hazard, lam)``).
    observation_likelihood : object
        Must provide ``pdf(x)`` and ``update_theta(x)`` like :class:`StudentT`.
    length : int
        Initial maximum time index; matrix is expanded by :meth:`expand_matrix`
        when needed.
    verbose : bool
        If True, print when a changepoint is flagged.
    """

    __slots__ = (
        "t",
        "R",
        "H",
        "_h_inv_lam",
        "_hazard_buf",
        "_r_indices",
        "observation_likelihood",
        "changepoints",
        "cp_probs",
        "length",
        "cp_detected",
        "verbose",
    )

    def __init__(
        self,
        hazard_function: Callable,
        observation_likelihood: object,
        length: int,
        *,
        verbose: bool = False,
    ) -> None:
        """Allocate run-length matrix and wire hazard + likelihood."""
        self.t = 0
        self.R = np.zeros((length, length), dtype=np.float64)
        self.H = hazard_function
        self._h_inv_lam = _constant_hazard_inv_lam(hazard_function)
        self._hazard_buf: Optional[np.ndarray] = None
        self._r_indices: Optional[np.ndarray] = None
        self.observation_likelihood = observation_likelihood
        self.R[0, 0] = 1.0
        self.changepoints: list[int] = []
        self.cp_probs: list[float] = []
        self.length = length
        self.cp_detected = False
        self.verbose = verbose

    @staticmethod
    def _resolve_price_column(df: object, price_col: str) -> str:
        cols = getattr(df, "columns", None)
        if cols is None:
            raise TypeError("df must have a .columns attribute")
        if price_col in cols:
            return price_col
        normalized = {str(c).strip(): c for c in cols}
        key = price_col.strip()
        if key in normalized:
            return str(normalized[key])
        raise KeyError(
            f"column {price_col!r} not found; available (stripped): {list(normalized.keys())}"
        )

    def run(
        self,
        df: pd.DataFrame,
        *,
        price_col: str = "Close",
    ) -> pd.Series:
        """Feed each bar's price through :meth:`update` in row order.

        Same side effects as calling :meth:`update` in a loop (mutates ``R``,
        ``cp_probs``, etc.). For a full history from scratch, construct a new
        ``BOCD``.

        Typical input is OHLCV with a ``DatetimeIndex``; only ``price_col`` is
        used (default ``Close``). Leading/trailing spaces in column names are
        matched (e.g. column named ``Open`` vs `` Open``).

        Parameters
        ----------
        df : pandas.DataFrame
            Row order is iteration order (use a sorted index if time-ordered).
        price_col : str, default ``Close``
            Column passed to :meth:`update` as ``float``. Coerced with
            ``pd.to_numeric(..., errors="coerce")``; non-finite values still go
            through :meth:`update`.

        Returns
        -------
        pandas.Series
            ``cp_prob`` for rows processed in this call only (index ``df.index``),
            even if :meth:`update` ran earlier on the same instance.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"expected pandas.DataFrame, got {type(df).__name__}")
        col = self._resolve_price_column(df, price_col)
        prices = pd.to_numeric(df[col], errors="coerce")
        start = len(self.cp_probs)
        for x in prices:
            self.update(float(x))
        return pd.Series(
            self.cp_probs[start:],
            index=df.index,
            name="cp_prob",
            dtype=np.float64,
        )

    def expand_matrix(self) -> None:
        """Double the allocation of ``R`` by padding with zeros (square matrix).

        Called when ``t`` reaches ``length - 1`` so the filter can run past the
        initial preallocated horizon.
        """
        L = self.R.shape[0]
        self.R = np.pad(self.R, ((0, L), (0, L)))
        self.length = self.R.shape[0]
        ol = self.observation_likelihood
        if hasattr(ol, "_grow_capacity"):
            ol._grow_capacity(self.length)

    def update(self, x: float) -> None:
        """Ingest scalar observation ``x``, advance run-length posterior, detect CPs.

        Parameters
        ----------
        x : float
            Next observation in the series.

        Side effects
        ------------
        Increments ``self.t``, may print if ``verbose``, mutates ``self.R`` and
        ``observation_likelihood`` internal arrays.
        """
        self.cp_detected = False

        if self.t == self.length - 1:
            self.expand_matrix()

        t = self.t

        predprobs = self.observation_likelihood.pdf(x)
        np.clip(predprobs, 1e-300, None, out=predprobs)

        row_prev = self.R[: t + 1, t]
        col_next = self.R[:, t + 1]

        if self._h_inv_lam is not None:
            h = self._h_inv_lam
            omh = 1.0 - h
            # Growth: R[1:t+2, t+1] = R[0:t+1, t] * pred * (1-h)
            np.multiply(row_prev, predprobs, out=col_next[1 : t + 2])
            col_next[1 : t + 2] *= omh
            # Reset mass at r=0
            col_next[0] = h * float(np.dot(row_prev, predprobs))
        else:
            if self._hazard_buf is None or self._hazard_buf.size < t + 1:
                self._hazard_buf = np.empty(max(t + 1, 64), dtype=np.float64)
            rslice = self._hazard_buf[: t + 1]
            # Cached 0..T-1 for hazard(r); np.arange has no `out=` kwarg.
            need = t + 1
            if self._r_indices is None or self._r_indices.size < need:
                self._r_indices = np.arange(
                    max(need, 64 if self._r_indices is None else self._r_indices.size * 2),
                    dtype=np.float64,
                )
            np.copyto(rslice, self._r_indices[:need])
            # hazard may return new array — try in-place if same object
            H = self.H(rslice)
            if H is not rslice:
                if H.size != t + 1:
                    H = np.asarray(H, dtype=np.float64).reshape(-1)[: t + 1]
                np.copyto(rslice, H)
                H = rslice
            omh = 1.0 - H
            np.multiply(row_prev, predprobs, out=col_next[1 : t + 2])
            np.multiply(col_next[1 : t + 2], omh, out=col_next[1 : t + 2])
            col_next[0] = np.sum(row_prev * predprobs * H)

        s = float(col_next.sum())
        if s > 0.0:
            col_next /= s

        self.observation_likelihood.update_theta(x)

        cp = float(self.R[1, t])
        self.cp_probs.append(cp)
        if cp > 0.35 and t != 1:
            self.changepoints.append(t)
            self.cp_detected = True
            if self.verbose:
                print(f"cp detected at index {self.t}, value {x}")

        self.t += 1
