"""
SMA/EMA Trend Strategy Backtester
---------------------------------
Compares 20/50-day SMA and EMA crossover strategies on Nifty 50 daily data
against buy-and-hold.

Design rules (to avoid fooling ourselves):
  * Long/flat only: position is 1 when fast MA > slow MA, else 0.
  * Signals are computed at close of day t and executed on day t+1 (shift(1)),
    so no look-ahead bias.
  * 5 bps cost charged on every position change (entry or exit).
  * Parameters (20/50) are fixed BEFORE any testing and never tuned.
  * Data is split chronologically: first 70% in-sample, final 30% out-of-sample.
    Metrics are reported separately for each segment.

Usage:
    python backtest.py                    # downloads ^NSEI via yfinance
    python backtest.py --csv my_data.csv  # CSV with Date and Close columns
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Fixed parameters (set before testing, never tuned) ---------------------
FAST, SLOW = 20, 50
COST_BPS = 5
TRADING_DAYS = 252
START_DATE = "2007-01-01"
OOS_FRACTION = 0.30
TICKER = "^NSEI"

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "nifty50.csv"
OUT_DIR = ROOT / "results"


# ---- Data -------------------------------------------------------------------
def load_data(csv: str | None = None) -> pd.Series:
    """Return a daily Close price series indexed by date."""
    if csv:
        df = pd.read_csv(csv)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        import yfinance as yf
        raw = yf.download(TICKER, start=START_DATE, auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("Download failed. Provide a CSV with --csv.")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close"]].reset_index()
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df.columns = [c.strip().title() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    close = (df.set_index("Date")["Close"].astype(float)
               .sort_index().dropna())
    close = close[~close.index.duplicated()]
    return close.loc[START_DATE:]


# ---- Strategy ---------------------------------------------------------------
def signal_sma(close: pd.Series) -> pd.Series:
    fast = close.rolling(FAST).mean()
    slow = close.rolling(SLOW).mean()
    return ((fast > slow) & slow.notna()).astype(int)


def signal_ema(close: pd.Series) -> pd.Series:
    fast = close.ewm(span=FAST, adjust=False).mean()
    slow = close.ewm(span=SLOW, adjust=False).mean()
    # Ignore the EMA warm-up so both strategies start trading on the same day.
    valid = close.rolling(SLOW).mean().notna()
    return ((fast > slow) & valid).astype(int)


def run_strategy(close: pd.Series, signal: pd.Series) -> pd.DataFrame:
    """Daily net returns for a long/flat strategy with next-day execution."""
    ret = close.pct_change()
    position = signal.shift(1).fillna(0)              # trade the day AFTER the signal
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * COST_BPS / 10_000
    net = position * ret - cost
    return pd.DataFrame({"ret": net, "position": position, "turnover": turnover})


# ---- Metrics ----------------------------------------------------------------
def metrics(ret: pd.Series, turnover: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    equity = (1 + ret).cumprod()
    years = len(ret) / TRADING_DAYS
    ann_ret = equity.iloc[-1] ** (1 / years) - 1
    vol = ret.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = ret.mean() * TRADING_DAYS / vol if vol > 0 else np.nan   # rf = 0
    max_dd = (equity / equity.cummax() - 1).min()
    out = {
        "Ann. Return": ann_ret,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
    }
    out["Trades"] = int(turnover.loc[ret.index].sum() // 1) if turnover is not None else np.nan
    return out


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["Ann. Return", "Volatility", "Max Drawdown"]:
        out[c] = (out[c] * 100).map("{:.2f}%".format)
    out["Sharpe"] = out["Sharpe"].map("{:.2f}".format)
    out["Trades"] = out["Trades"].map(lambda x: "-" if pd.isna(x) else f"{int(x)}")
    return out


# ---- Plots ------------------------------------------------------------------
def plot_results(returns: pd.DataFrame, split_date: pd.Timestamp) -> None:
    styles = {"SMA 20/50": "tab:blue", "EMA 20/50": "tab:orange", "Buy & Hold": "gray"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    # 1) Full-period equity (log scale) with split marker
    ax = axes[0]
    for name, color in styles.items():
        ax.plot((1 + returns[name]).cumprod(), label=name, color=color, lw=1.4)
    ax.axvline(split_date, color="red", ls="--", lw=1)
    ax.text(split_date, ax.get_ylim()[1], "  out-of-sample →", color="red", va="top")
    ax.set_yscale("log")
    ax.set_title("Equity curve, full period (growth of 1, log scale)")
    ax.legend(); ax.grid(alpha=.3)

    # 2 & 3) Each segment rebased to 1 so risk/return is comparable
    for ax, (label, seg) in zip(axes[1:], [
        ("In-sample", returns.loc[:split_date - pd.Timedelta(days=1)]),
        ("Out-of-sample", returns.loc[split_date:]),
    ]):
        for name, color in styles.items():
            ax.plot((1 + seg[name]).cumprod(), label=name, color=color, lw=1.4)
        ax.set_title(f"{label} (rebased to 1)")
        ax.legend(); ax.grid(alpha=.3)

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)


# ---- Main -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="CSV with Date and Close columns")
    args = parser.parse_args()

    close = load_data(args.csv)

    # Compute signals on the full history (they only use past data), then
    # drop the warm-up so every strategy and the benchmark share one window.
    sma = run_strategy(close, signal_sma(close))
    ema = run_strategy(close, signal_ema(close))
    bh_ret = close.pct_change()
    bh = pd.DataFrame({"ret": bh_ret, "turnover": np.nan})

    start = close.index[SLOW]                       # first day all MAs are valid
    returns = pd.DataFrame({
        "SMA 20/50": sma["ret"], "EMA 20/50": ema["ret"], "Buy & Hold": bh["ret"],
    }).loc[start:]
    turnover = {"SMA 20/50": sma["turnover"], "EMA 20/50": ema["turnover"], "Buy & Hold": None}

    split_idx = int(len(returns) * (1 - OOS_FRACTION))
    split_date = returns.index[split_idx]
    segments = {
        "Full period": returns,
        "In-sample": returns.iloc[:split_idx],
        "Out-of-sample": returns.iloc[split_idx:],
    }

    print(f"Data: {returns.index[0].date()} → {returns.index[-1].date()} "
          f"({len(returns)} trading days)")
    print(f"In-sample ends {returns.index[split_idx - 1].date()}, "
          f"out-of-sample starts {split_date.date()}\n")

    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    for seg_name, seg in segments.items():
        table = pd.DataFrame({n: metrics(seg[n], turnover[n]) for n in seg.columns}).T
        print(f"=== {seg_name} ===")
        print(fmt(table).to_string(), "\n")
        table.insert(0, "Segment", seg_name)
        rows.append(table)
    pd.concat(rows).to_csv(OUT_DIR / "metrics.csv", index_label="Strategy")

    plot_results(returns, split_date)
    print(f"Saved results/metrics.csv and results/equity_curves.png")


if __name__ == "__main__":
    main()
