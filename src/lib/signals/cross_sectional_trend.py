# -*- coding: utf-8 -*-
"""Cross-sectional momentum signals and a simple dollar-neutral backtest.

Builds multi-horizon log-return / rolling-volatility scores per asset, averages
them across horizons, then maps ranks to long-top-K / short-bottom-K weights.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Sequence

import numpy as np
import pandas as pd


class CrossSectionalTrendEngine:
    """Multi-horizon cross-sectional momentum from a wide price matrix.

    Parameters
    ----------
    prices : pandas.DataFrame
        **Rows** = time (any ``DatetimeIndex`` or ordered index), **columns** =
        assets (e.g. tickers). Values must be usable as price levels (positive
        floats for log-returns). Build this from stacked OHLCV via e.g.
        ``panel["Close"].unstack(level=0)`` if your panel is long-form with
        symbol on level 0.
    lookback_windows : sequence of int, default (12, 48, 96)
        Rolling lengths in **rows** (e.g. 5-minute bars: 48 ≈ 4 hours).
    fee_bps : float, default 5
        Linear transaction cost applied to turnover each period, in basis
        points (1 bp = 0.01%).
    copy : bool, default False
        If True, store ``prices.copy()`` so later mutations of the input do not
        affect the engine.

    Class attributes
    ----------------
    top_k : int, default 5
        Default number of names to long and short per bar in :meth:`generate_weights`.
        Override on the class (``CrossSectionalTrendEngine.top_k = 3``) or pass
        ``top_k=`` to that method.
    """

    top_k: ClassVar[int] = 5

    __slots__ = ("lookbacks", "fee", "prices", "signal", "weights")

    def __init__(
        self,
        prices: pd.DataFrame,
        lookback_windows: Sequence[int] = (12, 48, 96),
        fee_bps: float = 5.0,
        *,
        copy: bool = False,
    ) -> None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError(f"prices must be a pandas.DataFrame, got {type(prices).__name__}")
        self.lookbacks = tuple(int(x) for x in lookback_windows)
        if not self.lookbacks:
            raise ValueError("lookback_windows must be non-empty")
        self.fee = fee_bps / 10_000.0
        self.prices: pd.DataFrame = prices.copy() if copy else prices
        self.signal: Optional[pd.DataFrame] = None
        self.weights: Optional[pd.DataFrame] = None

    def compute_signal(self) -> pd.DataFrame:
        """Average momentum score across all lookback horizons.

        For each horizon ``n`` and asset, score is

        .. math::

            \\frac{\\ln(P_t / P_{t-n})}{\\sigma_n(t)}

        where :math:`\\sigma_n` is the rolling standard deviation of simple
        one-period returns over ``n`` rows. Horizons are averaged per cell
        using only finite values (all-NaN cells stay NaN).

        Returns
        -------
        pandas.DataFrame
            Same index and columns as ``self.prices``.
        """
        px = self.prices
        log_p = np.log(px.replace(0.0, np.nan))
        rets_1 = px.pct_change()

        layers: list[np.ndarray] = []
        cols = px.columns
        idx = px.index
        for n in self.lookbacks:
            mom = log_p.diff(n)
            vol = rets_1.rolling(window=n, min_periods=n).std()
            vol_safe = vol.replace(0.0, np.nan)
            z = (mom / vol_safe).to_numpy(dtype=np.float64, copy=False)
            layers.append(z)

        stacked = np.stack(layers, axis=0)
        counts = np.sum(np.isfinite(stacked), axis=0)
        sums = np.nansum(stacked, axis=0)
        composite = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        self.signal = pd.DataFrame(composite, index=idx, columns=cols)
        return self.signal

    def generate_weights(
        self,
        signal: Optional[pd.DataFrame] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Dollar-neutral long top-K / short bottom-K weights from cross-sectional ranks.

        At each time row, assets with rank :math:`\\le K` (best momentum) get
        +1, rank :math:`> R_{\\max} - K` get -1, others 0. Rows are then scaled
        so absolute weights sum to 1 (dollar neutral when gross exposure is
        interpreted in absolute terms).

        Parameters
        ----------
        signal : pandas.DataFrame, optional
            Same alignment as ``self.prices``. If omitted, uses the last
            :meth:`compute_signal` result (call :meth:`compute_signal` first).
        top_k : int, optional
            If omitted, uses :attr:`top_k` on the class.

        Returns
        -------
        pandas.DataFrame
            Target weights indexed like ``signal``; also stored on ``self.weights``.
        """
        if signal is None:
            if self.signal is None:
                raise TypeError("pass signal=... or call compute_signal() first")
            signal = self.signal
        k = int(type(self).top_k if top_k is None else top_k)
        ranks = signal.rank(axis=1, ascending=False, method="first")
        longs = (ranks <= k).astype(np.float64)
        cut = ranks.max(axis=1).sub(float(k))
        shorts = ranks.gt(cut, axis=0).astype(np.float64)
        raw = longs - shorts
        denom = raw.abs().sum(axis=1).replace(0.0, np.nan)
        self.weights = raw.div(denom, axis=0).fillna(0.0)
        return self.weights

    def run_backtest(self, weights: Optional[pd.DataFrame] = None) -> pd.Series:
        """Cumulative simple returns with one-bar signal lag and turnover fees.

        Positions at ``t`` are ``weights`` known at ``t-1`` (``shift(1)``),
        multiplied by simple returns ``P_t / P_{t-1} - 1``. Costs subtract
        ``fee * turnover`` where turnover is the sum of absolute weight changes
        from the prior row.

        Parameters
        ----------
        weights : pandas.DataFrame, optional
            Aligned with ``self.prices``. If omitted, uses the last
            :meth:`generate_weights` result.

        Returns
        -------
        pandas.Series
            Cumulative net simple return through time (not compounded stepwise
            in this helper — sum of per-period net returns).
        """
        if weights is None:
            if self.weights is None:
                raise TypeError("pass weights=... or call generate_weights() first")
            weights = self.weights
        asset_rets = self.prices.pct_change()
        lag_w = weights.shift(1)
        strat_rets = (lag_w * asset_rets).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        net = strat_rets - turnover * self.fee
        return net.cumsum()
