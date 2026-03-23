import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class CUSUMState:
    """Mutable state for :class:`OnlineCUSUMFilter`.

    Attributes
    ----------
    s_pos : float
        Upside CUSUM accumulator (positive standardized deviations).
    s_neg : float
        Downside CUSUM accumulator (negative standardized deviations).
    ewm_mean : float or None
        Exponentially weighted mean of the input series; ``None`` until first observation.
    ewm_var : float
        Exponentially weighted variance used to standardize innovations.
    t : int
        Number of ``update`` calls processed.
    """
    s_pos:    float = 0.0    # Upside accumulator
    s_neg:    float = 0.0    # Downside accumulator
    ewm_mean: float = None   # EWM mean for standardisation
    ewm_var:  float = 1.0    # EWM variance
    t:        int   = 0      # Observation counter


class OnlineCUSUMFilter:
    """Dual-sided CUSUM breakout detector with adaptive threshold.

    Parameters
    ----------
    k              : float  Allowance / slack.       Default 0.5
    h_base         : float  Base decision threshold. Default 4.0
    span           : int    EWM span for z-scoring.  Default 20
    vol_adapt_span : int    Slow vol EWM span.        Default 80
    """
    def __init__(self, k=0.5, h_base=4.0, span=20, vol_adapt_span=80):
        """Initialize filter hyperparameters and empty state.

        Parameters
        ----------
        k : float, default 0.5
            CUSUM allowance (slack). Only innovations beyond ``k`` in absolute
            z-score contribute to the accumulators.
        h_base : float, default 4.0
            Base decision threshold; alarm when an accumulator exceeds the
            adaptive threshold derived from ``h_base``.
        span : int, default 20
            Span for the fast EWM mean/variance used to compute per-step z-scores.
        vol_adapt_span : int, default 80
            Span for the slow EWM of variance; scales ``h`` relative to recent vol.
        """
        self.k          = k
        self.h_base     = h_base
        self.alpha      = 2.0 / (span + 1)
        self.alpha_slow = 2.0 / (vol_adapt_span + 1)
        self.state      = CUSUMState()
        self._var_slow: Optional[float] = None

    def _update_ewm(self, x: float) -> float:
        """Update EWM mean/variance with ``x`` and return a z-like score.

        On the first call, initializes ``ewm_mean`` and ``ewm_var`` and returns 0.
        Otherwise updates recursive EWM statistics and returns
        ``(x - ewm_mean) / sqrt(ewm_var)`` (with epsilon in the denominator).

        Parameters
        ----------
        x : float
            Latest raw observation (e.g. log price).

        Returns
        -------
        float
            Standardized innovation for the CUSUM recursions; 0.0 on first step.
        """
        st, a = self.state, self.alpha
        if st.ewm_mean is None:
            st.ewm_mean, st.ewm_var = x, 1.0
            return 0.0
        delta       = x - st.ewm_mean
        st.ewm_mean = a * x + (1 - a) * st.ewm_mean
        st.ewm_var  = (1 - a) * (st.ewm_var + a * delta ** 2)
        return delta / (np.sqrt(st.ewm_var) + 1e-10)

    def _adaptive_h(self) -> float:
        """Decision threshold for CUSUM alarms, scaled by recent vs slow volatility.

        Computes ``h_base * clip(sqrt(ewm_var / var_slow), 0.5, 2.0)``.
        If slow variance or EWM variance is unavailable, returns ``h_base``.

        Returns
        -------
        float
            Threshold ``h`` such that ``s_pos > h`` or ``s_neg > h`` triggers an alarm.
        """
        if self._var_slow is None or self.state.ewm_var is None:
            return self.h_base
        ratio = np.sqrt(self.state.ewm_var / (self._var_slow + 1e-10))
        ratio = float(np.clip(ratio, 0.5, 2.0))
        return self.h_base * ratio

    def update(self, x: float) -> int:
        """Ingest one observation and return a breakout signal.

        Increments the time counter, updates EWM-based z-scores, refreshes the
        slow variance track, then applies dual one-sided CUSUM recursions:
        ``s_pos = max(0, s_pos + z - k)``, ``s_neg = max(0, s_neg - z - k)``.
        If ``s_pos`` (resp. ``s_neg``) exceeds the adaptive threshold ``h``,
        that accumulator is reset and ``+1`` (resp. ``-1``) is returned.

        Parameters
        ----------
        x : float
            Latest observation in the same units as prior calls (e.g. log price).

        Returns
        -------
        int
            ``+1`` upside alarm, ``-1`` downside alarm, or ``0`` if no alarm.
        """
        st = self.state
        st.t += 1
        z = self._update_ewm(x)
        if self._var_slow is None:
            self._var_slow = st.ewm_var
        else:
            self._var_slow = ((1 - self.alpha_slow) * self._var_slow
                              + self.alpha_slow * st.ewm_var)
        h = self._adaptive_h()
        st.s_pos = max(0.0, st.s_pos + z - self.k)
        st.s_neg = max(0.0, st.s_neg - z - self.k)
        if st.s_pos > h: st.s_pos = 0.0; return +1
        if st.s_neg > h: st.s_neg = 0.0; return -1
        return 0

    def run(self, series: pd.Series) -> pd.Series:
        """Apply :meth:`update` sequentially to each value in ``series``.

        State carries across the loop; order matches ``series`` index order.

        Parameters
        ----------
        series : pd.Series
            Time-ordered observations (e.g. log prices or returns).

        Returns
        -------
        pd.Series
            Integer series aligned to ``series.index``: ``+1``, ``-1``, or ``0``
            per step, same semantics as :meth:`update`.
        """
        signals = pd.Series(0, index=series.index, dtype=int)
        for i, (_, val) in enumerate(series.items()):
            signals.iloc[i] = self.update(float(val))
        return signals
