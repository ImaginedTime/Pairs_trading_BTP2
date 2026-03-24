from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .constants import (
    DEFAULT_BOLLINGER_EXIT_STD,
    DEFAULT_BOLLINGER_NUM_STD,
    DEFAULT_BOLLINGER_WINDOW,
    DEFAULT_COINTEGRATION_ALPHA,
    DEFAULT_COINTEGRATION_ENTRY_Z,
    DEFAULT_COINTEGRATION_EXIT_Z,
    DEFAULT_COPULA_ENTRY_QUANTILE,
    DEFAULT_COPULA_EXIT_QUANTILE,
    DEFAULT_END_DATE,
    DEFAULT_FORECAST_REFRESH_DAYS,
    DEFAULT_HURST_THRESHOLD,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_INTERVAL_DAILY,
    DEFAULT_INTERVAL_HOURLY,
    DEFAULT_MAX_ALLOWED_PREDICTION_ERROR,
    DEFAULT_MAX_HALF_LIFE_DAYS,
    DEFAULT_MAX_PAIR_LOSS_PCT,
    DEFAULT_MAX_PORTFOLIO_DRAWDOWN_PCT,
    # DEFAULT_MEAN_CROSSES_PER_YEAR if False else None,  # keeps lint happy if edited later
    DEFAULT_MIN_HALF_LIFE_DAYS,
    DEFAULT_NOTIONAL_PER_TRADE,
    DEFAULT_OU_CONFIDENCE,
    DEFAULT_OU_ENTRY_Z,
    DEFAULT_OU_EXIT_Z,
    DEFAULT_OU_N_PATHS,
    DEFAULT_OU_WINDOW,
    DEFAULT_OPTICS_MIN_SAMPLES,
    DEFAULT_PAIR_REFRESH_DAYS,
    DEFAULT_PCA_VARIANCE_THRESHOLD,
    DEFAULT_PERIODS_PER_YEAR,
    DEFAULT_RISK_FRACTION,
    DEFAULT_SLIPPAGE_BPS,
    DEFAULT_START_DATE,
    DEFAULT_TRANSACTION_COST_BPS,
)

from .constants import DEFAULT_MIN_MEAN_CROSSES_PER_YEAR


@dataclass
class DataConfig:
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    daily_interval: str = DEFAULT_INTERVAL_DAILY
    hourly_interval: str = DEFAULT_INTERVAL_HOURLY
    universe_as_of: str = DEFAULT_END_DATE
    universe_csv: Path | None = None
    raw_dir: Path | None = None
    interim_dir: Path | None = None
    processed_dir: Path | None = None


@dataclass
class SelectionConfig:
    pca_variance_threshold: float = DEFAULT_PCA_VARIANCE_THRESHOLD
    optics_min_samples: int = DEFAULT_OPTICS_MIN_SAMPLES
    cointegration_alpha: float = DEFAULT_COINTEGRATION_ALPHA
    hurst_threshold: float = DEFAULT_HURST_THRESHOLD
    min_half_life_days: float = DEFAULT_MIN_HALF_LIFE_DAYS
    max_half_life_days: float = DEFAULT_MAX_HALF_LIFE_DAYS
    min_mean_crosses_per_year: float = DEFAULT_MIN_MEAN_CROSSES_PER_YEAR


@dataclass
class StrategyConfig:
    bollinger_window: int = DEFAULT_BOLLINGER_WINDOW
    bollinger_num_std: float = DEFAULT_BOLLINGER_NUM_STD
    bollinger_exit_std: float = DEFAULT_BOLLINGER_EXIT_STD

    ou_window: int = DEFAULT_OU_WINDOW
    ou_confidence: float = DEFAULT_OU_CONFIDENCE
    ou_n_paths: int = DEFAULT_OU_N_PATHS
    ou_entry_z: float = DEFAULT_OU_ENTRY_Z
    ou_exit_z: float = DEFAULT_OU_EXIT_Z

    copula_entry_quantile: float = DEFAULT_COPULA_ENTRY_QUANTILE
    copula_exit_quantile: float = DEFAULT_COPULA_EXIT_QUANTILE

    cointegration_entry_z: float = DEFAULT_COINTEGRATION_ENTRY_Z
    cointegration_exit_z: float = DEFAULT_COINTEGRATION_EXIT_Z


@dataclass
class BacktestConfig:
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    notional_per_trade: float = DEFAULT_NOTIONAL_PER_TRADE
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR


@dataclass
class RiskConfig:
    max_pair_loss_pct: float = DEFAULT_MAX_PAIR_LOSS_PCT
    max_portfolio_drawdown_pct: float = DEFAULT_MAX_PORTFOLIO_DRAWDOWN_PCT
    pair_refresh_days: int = DEFAULT_PAIR_REFRESH_DAYS
    forecast_refresh_days: int = DEFAULT_FORECAST_REFRESH_DAYS
    max_allowed_prediction_error: float = DEFAULT_MAX_ALLOWED_PREDICTION_ERROR
    risk_fraction: float = DEFAULT_RISK_FRACTION


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def to_dict(self) -> dict:
        return {
            "data": self.data.__dict__,
            "selection": self.selection.__dict__,
            "strategy": self.strategy.__dict__,
            "backtest": self.backtest.__dict__,
            "risk": self.risk.__dict__,
        }