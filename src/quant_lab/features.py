from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged price/volume features for financial ML.

    All rolling statistics are shifted by one bar to reduce look-ahead leakage.
    """
    out = df.copy()
    close = out["close"]

    for window in [1, 2, 5, 10, 20, 60]:
        out[f"ret_{window}"] = close.pct_change(window).shift(1)

    for window in [5, 10, 20, 60]:
        returns = close.pct_change()
        out[f"vol_{window}"] = returns.rolling(window).std().shift(1)
        ma = close.rolling(window).mean()
        std = close.rolling(window).std()
        out[f"ma_dist_{window}"] = (close / ma - 1).shift(1)
        out[f"zscore_{window}"] = ((close - ma) / std).shift(1)
        out[f"breakout_high_{window}"] = (close / close.rolling(window).max() - 1).shift(1)
        out[f"breakout_low_{window}"] = (close / close.rolling(window).min() - 1).shift(1)

    out["volume_change_5"] = out["volume"].pct_change(5).shift(1)
    out["hl_range"] = ((out["high"] - out["low"]) / out["close"]).shift(1)

    return out.replace([np.inf, -np.inf], np.nan)


def make_direction_label(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.002,
) -> pd.Series:
    """Create a three-class forward-return label.

    Returns:
        1 for positive future return above threshold,
       -1 for negative future return below threshold,
        0 otherwise.
    """
    fwd_return = df["close"].pct_change(horizon).shift(-horizon)
    label = pd.Series(0, index=df.index, name="label")
    label[fwd_return > threshold] = 1
    label[fwd_return < -threshold] = -1
    return label


def make_model_frame(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.002,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build aligned feature matrix and label vector."""
    featured = add_price_features(df)
    label = make_direction_label(featured, horizon=horizon, threshold=threshold)

    feature_cols = [
        col
        for col in featured.columns
        if col
        not in {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
    ]

    model_df = featured[feature_cols].join(label).dropna()
    X = model_df[feature_cols]
    y = model_df["label"].astype(int)
    return X, y
