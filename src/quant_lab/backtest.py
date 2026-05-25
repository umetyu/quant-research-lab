from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    prices: pd.Series,
    signal: pd.Series,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Run a simple close-to-close vectorized backtest.

    The signal at time t is shifted by one period before applying returns,
    so today's prediction is traded from the next bar.
    """
    aligned = pd.concat(
        [
            prices.rename("close"),
            signal.rename("signal"),
        ],
        axis=1,
    ).dropna()

    position = aligned["signal"].clip(-1, 1).shift(1).fillna(0)
    asset_return = aligned["close"].pct_change().fillna(0)

    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (transaction_cost_bps / 10_000)

    strategy_return = position * asset_return - cost
    equity = (1 + strategy_return).cumprod()

    return pd.DataFrame(
        {
            "close": aligned["close"],
            "signal": aligned["signal"],
            "position": position,
            "asset_return": asset_return,
            "strategy_return": strategy_return,
            "turnover": turnover,
            "equity": equity,
        },
        index=aligned.index,
    )
