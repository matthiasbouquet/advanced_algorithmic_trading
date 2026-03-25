import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DAYS_PER_YEAR = 252


class PCAStatArbEngine:
    def __init__(self, df_ohlcv, n_components=5, lookup_window=252):
        """
        df_ohlcv: MultiIndex DataFrame (Ticker, Timestamp) with 'Close'
        n_components: Number of Eigen-factors to hedge against
        lookup_window: Number of OHLCV bars to fit the PCA
        """
        self.prices = df_ohlcv.copy()
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()
        self.n_components = n_components
        self.lookup_window = lookup_window
    
    def _get_pca(self, window_data, n_components=None):
        """Perform PCA decomposition on a returns matrix."""
        # Standardize the returns matrix
        scaler = StandardScaler()
        scaled_rets = scaler.fit_transform(window_data)
        # Perform PCA decomposition

        if n_components is None:
            n_components = self.n_components
        
        # Create PCA object and fit the model
        pca = PCA(n_components=n_components)
        pca.fit(scaled_rets)
        
        return pca, scaled_rets

    def _get_residuals(self, window_data):
        """Extract idiosyncratic noise from PCA decomposition."""
        
        pca, scaled_rets = self._get_pca(window_data)
        
        # Project and reconstruct to find the 'explained' portion
        projected_rets = pca.transform(scaled_rets)
        explained_rets = pca.inverse_transform(projected_rets)
        residuals = scaled_rets - explained_rets
        
        # We return the last row (the most recent residual)
        return residuals[-1]

    def generate_signals(self):
        """Computes the S-Score: (Residual - Mean) / Std."""
        # Rolling residual calculation (Computationally intensive)
        resids = []
        for i in range(self.lookup_window, len(self.returns)):
            window = self.returns.iloc[i-self.lookup_window:i]
            resids.append(self._get_residuals(window))
        
        resid_df = pd.DataFrame(resids, index=self.returns.index[self.lookup_window:], 
                                columns=self.returns.columns)
        
        # S-Score is the standardized residual
        s_scores = (resid_df - resid_df.rolling(self.lookup_window).mean()) / \
                    resid_df.rolling(self.lookup_window).std()
        return s_scores

    def compute_weights(self, s_scores, entry_z=2.0):
        """
        Mean Reversion Logic: 
        Short if S > 2.0 (Overbought relative to factors)
        Long if S < -2.0 (Oversold relative to factors)
        """
        weights = pd.DataFrame(0, index=s_scores.index, columns=s_scores.columns)
        
        weights[s_scores > entry_z] = -1
        weights[s_scores < -entry_z] = 1
        
        # Neutralize: Ensure net exposure is zero across the universe
        row_sums = weights.abs().sum(axis=1)
        return weights.div(row_sums, axis=0).fillna(0)

    def calculate_performance(self, weights, bars_per_day=78):
        """Calculates returns, Sharpe, and Drawdown."""
        # Align returns with weights (weights are based on T, trade occurs at T+1)
        strat_rets = (weights.shift(1) * self.returns.loc[weights.index]).sum(axis=1)
        
        cum_rets = (1 + strat_rets).cumprod()
        annual_sharpe = (strat_rets.mean() / strat_rets.std()) * np.sqrt(DAYS_PER_YEAR * bars_per_day)
        
        # Drawdown
        rolling_max = cum_rets.cummax()
        drawdown = (cum_rets - rolling_max) / rolling_max
        
        stats = {
            "Total Return": cum_rets.iloc[-1] - 1,
            "Annualized Sharpe": annual_sharpe,
            "Max Drawdown": drawdown.min(),
            "Win Rate": (strat_rets > 0).mean()
        }
        return pd.Series(stats), cum_rets
