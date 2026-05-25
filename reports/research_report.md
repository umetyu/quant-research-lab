# Research Report: ML Direction Signal Backtest

## 1. Objective

This project evaluates whether simple lagged price and volume features can create a useful direction signal for a systematic trading strategy.

The emphasis is not on finding a production-ready alpha.  
The emphasis is on demonstrating a disciplined quant research workflow.

## 2. Hypothesis

Short-term price patterns such as recent momentum, volatility expansion, and distance from moving averages may contain weak information about near-future direction.

## 3. Data

The default experiment uses synthetic OHLCV data. This makes the repository public and reproducible without exposing private or paid datasets.

For a stronger version of this project, replace the synthetic generator with real liquid market data and document the data source.

## 4. Features

The model uses only lagged features:

- return features over 1, 2, 5, 10, 20, and 60 bars
- rolling volatility
- moving-average distance
- z-score relative to rolling mean and volatility
- high/low breakout distance
- volume change

All rolling features are shifted by one period to reduce look-ahead leakage.

## 5. Labeling

The target is a three-class future return label:

- `1`: forward return is above a positive threshold
- `-1`: forward return is below a negative threshold
- `0`: otherwise

This is a simplified public implementation inspired by financial machine learning labeling concepts.

## 6. Validation

The model is evaluated with walk-forward validation.  
This is more appropriate than random train/test splitting for time-series data because financial observations are ordered and non-stationary.

## 7. Backtest Assumptions

- signal is shifted by one bar before trading
- long/short/flat positions are allowed
- transaction cost is charged based on turnover
- no leverage constraint beyond position clipping to [-1, 1]
- no market impact or liquidity modeling

## 8. Metrics

The project reports:

- total return
- CAGR
- annualized volatility
- Sharpe ratio
- Sortino ratio
- maximum drawdown
- Calmar ratio
- hit rate
- average turnover

## 9. Limitations

- Synthetic data is not representative of real markets.
- No survivorship-bias-free equity universe is used.
- No exchange-specific fee model is implemented.
- No slippage or market impact model is implemented.
- Model selection is intentionally simple.
- Hyperparameter tuning could overfit without nested validation.
- A profitable backtest would not imply future profitability.

## 10. Next Steps

- Use real futures, ETF, or equity data.
- Add purged cross-validation with embargo.
- Add triple-barrier labeling.
- Add portfolio construction and risk targeting.
- Compare against buy-and-hold and simple momentum baselines.
- Add stress tests by volatility regime.
