from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def performance_summary(
    returns: pd.Series,
    equity: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    returns = returns.dropna()
    if equity is None:
        equity = (1 + returns).cumprod()

    n = len(returns)
    if n == 0:
        raise ValueError("returns is empty")

    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if equity.iloc[0] != 0 else np.nan
    cagr = equity.iloc[-1] ** (periods_per_year / n) - 1
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = np.nan if vol == 0 else returns.mean() / returns.std() * np.sqrt(periods_per_year)

    downside = returns[returns < 0]
    sortino = (
        np.nan
        if downside.std() == 0 or np.isnan(downside.std())
        else returns.mean() / downside.std() * np.sqrt(periods_per_year)
    )

    mdd = max_drawdown(equity)
    calmar = np.nan if mdd == 0 else cagr / abs(mdd)
    hit_rate = float((returns > 0).mean())

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(mdd),
        "calmar": float(calmar),
        "hit_rate": hit_rate,
    }
