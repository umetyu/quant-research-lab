from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd

from quant_lab.backtest import run_backtest
from quant_lab.data import MarketConfig, generate_synthetic_ohlcv
from quant_lab.features import make_model_frame
from quant_lab.metrics import performance_summary
from quant_lab.models import make_random_forest_model, walk_forward_predict
from quant_lab.validation import WalkForwardSplit


def main() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    df = generate_synthetic_ohlcv(MarketConfig(n_days=2500, seed=42))
    X, y = make_model_frame(df, horizon=5, threshold=0.002)

    splitter = WalkForwardSplit(train_size=750, test_size=125)
    model = make_random_forest_model(random_state=42)
    preds = walk_forward_predict(model, X, y, splitter)

    results = run_backtest(
        prices=df["close"],
        signal=preds,
        transaction_cost_bps=5.0,
    )

    summary = performance_summary(results["strategy_return"], results["equity"])
    summary["average_turnover"] = float(results["turnover"].mean())

    summary_df = pd.DataFrame([summary], index=["ML Direction Signal"]).T
    print(summary_df.round(4))

    results.to_csv(reports_dir / "equity_curve.csv")
    pd.DataFrame([summary], index=["ML Direction Signal"]).to_csv(
        reports_dir / "experiment_results.csv"
    )


if __name__ == "__main__":
    main()
