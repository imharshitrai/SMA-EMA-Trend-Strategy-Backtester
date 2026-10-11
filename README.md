# SMA/EMA Trend Strategy Backtester

Compares 20/50-day SMA and EMA crossover strategies on Nifty 50 daily data (2007 onward) against buy-and-hold, with an honest in-sample / out-of-sample split.

## Methodology
- **Rule:** long/flat. Position = 1 when fast MA > slow MA, else 0.
- **No look-ahead:** the signal at close of day *t* is executed on day *t+1* (`signal.shift(1)`).
- **Costs:** 5 bps charged on every position change (entry and exit).
- **Fixed parameters:** 20/50 chosen before testing; nothing is tuned.
- **Validation:** chronological split. First 70% = in-sample, final 30% = out-of-sample. Metrics reported separately for each, plus full period.
- **Metrics:** annualized return (geometric), annualized volatility, Sharpe (rf = 0, 252 days), max drawdown, trade count.
- **Warm-up:** all strategies and the benchmark start on the same day (first day the 50-day MA is valid), so comparisons are like-for-like.

## Run
```bash
pip install -r requirements.txt
python backtest.py                     # downloads ^NSEI via yfinance, caches to data/nifty50.csv
python backtest.py --csv my_data.csv   # or supply your own CSV with Date, Close columns
```

## Output
- `results/metrics.csv` – all metrics by strategy and segment
- `results/equity_curves.png` – full-period (log scale), in-sample, and out-of-sample (rebased) equity curves

## Notes / limitations
- Uses the index level (price return, no dividends) and ignores slippage beyond the 5 bps cost.
- Nifty 50 is not directly tradable; in practice you'd use an ETF or futures.
- Sharpe uses a zero risk-free rate; swap in an Indian T-bill rate if desired.
- Report the results as they come out, including if the strategies lag buy-and-hold out-of-sample.
