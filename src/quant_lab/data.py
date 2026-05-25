from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketConfig:
    """Parameters for synthetic OHLCV data generation."""

    n_days: int = 2500
    seed: int = 42
    start_price: float = 100.0
    annual_drift: float = 0.06
    annual_volatility: float = 0.22
    start_date: str = "2014-01-01"


def generate_synthetic_ohlcv(config: MarketConfig = MarketConfig()) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data with mild regime shifts.

    This keeps the repository runnable without proprietary market data.
    The data is not intended to represent any actual asset.
    """
    rng = np.random.default_rng(config.seed)
    dates = pd.bdate_range(config.start_date, periods=config.n_days)

    daily_drift = config.annual_drift / 252
    base_vol = config.annual_volatility / np.sqrt(252)

    regimes = np.ones(config.n_days)
    regimes[config.n_days // 3 : 2 * config.n_days // 3] = 1.6
    regimes[2 * config.n_days // 3 :] = 0.8

    shocks = rng.normal(daily_drift, base_vol * regimes)
    close = config.start_price * np.exp(np.cumsum(shocks))

    overnight = rng.normal(0, base_vol * 0.25, config.n_days)
    open_ = close * np.exp(overnight)

    intraday_range = np.abs(rng.normal(0, base_vol * 0.8, config.n_days))
    high = np.maximum(open_, close) * (1 + intraday_range)
    low = np.minimum(open_, close) * (1 - intraday_range)

    volume = rng.lognormal(mean=13.0, sigma=0.45, size=config.n_days).astype(int)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from CSV.

    Required columns: timestamp, open, high, low, close, volume.
    """
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)
