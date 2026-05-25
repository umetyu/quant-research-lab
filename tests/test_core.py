import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import numpy as np

from quant_lab.backtest import run_backtest
from quant_lab.data import MarketConfig, generate_synthetic_ohlcv
from quant_lab.features import make_model_frame
from quant_lab.metrics import performance_summary
from quant_lab.validation import WalkForwardSplit


def test_feature_frame_has_labels():
    df = generate_synthetic_ohlcv(MarketConfig(n_days=300, seed=1))
    X, y = make_model_frame(df)
    assert len(X) == len(y)
    assert set(y.unique()).issubset({-1, 0, 1})


def test_walk_forward_split_order():
    splitter = WalkForwardSplit(train_size=100, test_size=20)
    train_idx, test_idx = next(splitter.split(200))
    assert train_idx.max() < test_idx.min()


def test_backtest_outputs_equity():
    df = generate_synthetic_ohlcv(MarketConfig(n_days=300, seed=2))
    X, y = make_model_frame(df)
    signal = y.copy()
    bt = run_backtest(df["close"], signal)
    assert "equity" in bt.columns
    assert np.isfinite(bt["equity"]).all()


def test_performance_summary_keys():
    df = generate_synthetic_ohlcv(MarketConfig(n_days=300, seed=3))
    returns = df["close"].pct_change().dropna()
    summary = performance_summary(returns)
    assert "sharpe" in summary
    assert "max_drawdown" in summary
