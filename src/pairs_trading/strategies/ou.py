from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.api import OLS, add_constant

from pairs_trading.features.spreads import build_spread
from pairs_trading.selection.statistics import estimate_spread_beta
from .base import BasePairStrategy, extract_pair_series


def fit_ou_parameters(spread: pd.Series) -> dict[str, float]:
    """
    Fit a discrete-time OU model by estimating AR(1) on the spread.

    Model:
        x_t = a + b x_{t-1} + e_t

    Then:
        kappa = -ln(b)
        mu = a / (1 - b)
        sigma is estimated from residuals
    """
    s = pd.to_numeric(spread, errors="coerce").dropna().astype(float)
    if len(s) < 20:
        raise ValueError("spread must contain at least 20 observations")

    lagged = s.shift(1).dropna()
    current = s.loc[lagged.index]

    X = add_constant(lagged.values)
    model = OLS(current.values, X).fit()
    a = float(model.params[0])
    b = float(model.params[1])

    if b <= 0:
        b = 1e-6
    if b >= 1:
        b = 0.999999

    residuals = current.values - model.predict(X)
    sigma_eps = float(np.std(residuals, ddof=1))

    kappa = float(-np.log(b))
    mu = float(a / (1.0 - b))
    sigma = float(sigma_eps)

    return {
        "a": a,
        "b": b,
        "kappa": kappa,
        "mu": mu,
        "sigma": sigma,
        "resid_std": sigma_eps,
    }


def simulate_forecasts(params: dict, horizon: int, n_paths: int = 20000) -> pd.DataFrame:
    """
    Simulate OU paths for a given forecast horizon.

    Returns a DataFrame:
        rows   = simulation paths
        cols   = forecast steps 1..horizon
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")

    mu = float(params["mu"])
    b = float(params["b"])
    sigma = float(params["sigma"])

    # Start each path at the mean-reversion level.
    x = np.full(n_paths, mu, dtype=float)
    paths = np.zeros((n_paths, horizon), dtype=float)

    for t in range(horizon):
        eps = np.random.normal(0.0, sigma, size=n_paths)
        x = mu + b * (x - mu) + eps
        paths[:, t] = x

    cols = [f"t+{i+1}" for i in range(horizon)]
    return pd.DataFrame(paths, columns=cols)


def forecast_band(paths: pd.DataFrame, confidence: float = 0.98) -> tuple[pd.Series, pd.Series]:
    """
    Compute lower and upper forecast bands from simulated paths.
    """
    if paths is None or paths.empty:
        raise ValueError("paths is empty")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")

    alpha = (1.0 - confidence) / 2.0
    lower = paths.quantile(alpha, axis=0)
    upper = paths.quantile(1.0 - alpha, axis=0)
    return lower, upper


class OUForecastStrategy(BasePairStrategy):
    def __init__(
        self,
        window: int = 20,
        confidence: float = 0.98,
        n_paths: int = 20000,
        entry_z: float = 1.0,
        exit_z: float = 0.2,
    ) -> None:
        super().__init__()
        self.state.window = window
        self.confidence = confidence
        self.n_paths = n_paths
        self.entry_z = entry_z
        self.exit_z = exit_z

        self.spread_: pd.Series | None = None
        self.params_: dict[str, float] | None = None
        self.rolling_mean_: pd.Series | None = None
        self.rolling_std_: pd.Series | None = None

    def fit(self, pair_data: pd.DataFrame) -> None:
        left, right = extract_pair_series(pair_data)
        beta = estimate_spread_beta(left, right)
        spread = build_spread(left, right, beta=beta)

        self.state.hedge_ratio = beta
        self.state.pair_left = left.name
        self.state.pair_right = right.name
        self.spread_ = spread
        self.params_ = fit_ou_parameters(spread)
        self.rolling_mean_ = spread.rolling(self.state.window).mean()
        self.rolling_std_ = spread.rolling(self.state.window).std(ddof=0)
        self.is_fitted = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            self.fit(data)

        left, right = extract_pair_series(data)
        spread = build_spread(left, right, beta=self.state.hedge_ratio)

        signals = pd.Series(0.0, index=spread.index, name="signal")
        position = 0.0

        for i in range(len(spread)):
            if i < self.state.window:
                signals.iloc[i] = 0.0
                continue

            window_spread = spread.iloc[i - self.state.window : i].dropna()
            if len(window_spread) < self.state.window // 2:
                signals.iloc[i] = position
                continue

            params = fit_ou_parameters(window_spread)
            paths = simulate_forecasts(params, horizon=1, n_paths=self.n_paths)
            lower, upper = forecast_band(paths, confidence=self.confidence)

            current = float(spread.iloc[i])
            mean_forecast = float(paths.iloc[:, 0].mean())
            std_forecast = float(paths.iloc[:, 0].std(ddof=0))
            z = 0.0 if std_forecast == 0 else (current - mean_forecast) / std_forecast

            if position == 0.0:
                if current > float(upper.iloc[0]) or z >= self.entry_z:
                    position = -1.0
                elif current < float(lower.iloc[0]) or z <= -self.entry_z:
                    position = 1.0
            else:
                if abs(z) <= self.exit_z:
                    position = 0.0

            signals.iloc[i] = position

        return signals

    def backtest(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().backtest(data)