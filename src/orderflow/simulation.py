"""Cost-aware trading simulation for the ES order-flow signal (issue #9).

Strategy
--------
At each second t on the 1-second grid, the signal is a predicted forward
mid-price change in ticks (e.g. the Ridge predictions from #8), built only
from market events at or before t. If signal > threshold we buy one contract
at the ask; if signal < -threshold we sell one at the bid. The position is
held for h seconds and closed by crossing the spread again (sell at the bid
or buy back at the ask). By default only one position is open at a time.

Costs
-----
PnL before costs is mid to mid: side * (mid(exit) - mid(entry)).
PnL after costs subtracts the half-spread paid at entry and at exit, plus a
fee per side per contract. The base fee is the $1.00 placeholder proposed in
#4, with a sensitivity sweep from $0 to $2.50.

Assumptions
-----------
- One contract per trade, filled in full at the top of book.
- Zero latency by default: the order fills at the last quote at or before t,
  the same quote the signal saw. ``delay`` shifts entry and exit later by
  whole seconds to test how much that matters.
- Trades never span sessions. Build quotes per session with
  ``quotes_on_grid`` and concat; trades whose exit time is missing from the
  grid (the last h seconds of a session) are skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.25                 # ES tick size in index points
TICK_VALUE = 12.50          # dollars per tick per contract
BASE_FEE_PER_SIDE = 1.00    # dollars per contract per side (placeholder from #4)
FEE_SWEEP = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
CT = "America/Chicago"


def quotes_on_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Best bid, ask, and mid on the 1-second grid used by build_features.

    The row at t holds the last quote at or before t (bins closed and
    labelled on the right), and quiet seconds carry the previous quote.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Expected a timezone-aware DatetimeIndex")
    if len(pd.unique(df.index.tz_convert(CT).date)) > 1:
        raise ValueError(
            "quotes_on_grid expects one session; call it per day and concat"
        )

    book = df.sort_index()[["bid_px_00", "ask_px_00"]].astype("float64")
    grid = (
        book.resample("1s", closed="right", label="right")
        .last()
        .ffill()
        .rename(columns={"bid_px_00": "bid", "ask_px_00": "ask"})
    )
    grid["mid"] = (grid["bid"] + grid["ask"]) / 2
    return grid


def simulate_trades(
    signal: pd.Series,
    quotes: pd.DataFrame,
    horizon: int,
    threshold: float,
    delay: int = 0,
    allow_overlap: bool = False,
) -> pd.DataFrame:
    """Turn a signal into one-contract trades held for ``horizon`` seconds.

    signal     predicted forward mid change in ticks, indexed by grid time
    quotes     output of quotes_on_grid (bid, ask, mid), possibly several days
    horizon    holding period in whole seconds
    threshold  trade only when |signal| > threshold (ticks)
    delay      seconds between the signal and the fill (0 = same second)

    Returns one row per trade with entry/exit prices and PnL in ticks:
    gross_ticks (mid to mid), spread_cost_ticks, and exec_ticks (after
    crossing the spread, before fees).
    """
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive whole number of seconds")
    if int(delay) != delay or delay < 0:
        raise ValueError("delay must be a non-negative whole number of seconds")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    signal = signal.dropna().sort_index()
    candidates = signal[signal.abs() > threshold]

    entry_t = candidates.index + pd.Timedelta(seconds=int(delay))
    exit_t = entry_t + pd.Timedelta(seconds=int(horizon))
    entry_q = quotes.reindex(entry_t)
    exit_q = quotes.reindex(exit_t)

    valid = entry_q.notna().all(axis=1).to_numpy() & exit_q.notna().all(axis=1).to_numpy()

    trades = pd.DataFrame({
        "signal_time": candidates.index[valid],
        "entry_time": entry_t[valid],
        "exit_time": exit_t[valid],
        "side": np.sign(candidates.to_numpy()[valid]).astype(int),
        "signal": candidates.to_numpy()[valid],
        "entry_bid": entry_q["bid"].to_numpy()[valid],
        "entry_ask": entry_q["ask"].to_numpy()[valid],
        "exit_bid": exit_q["bid"].to_numpy()[valid],
        "exit_ask": exit_q["ask"].to_numpy()[valid],
    })

    if not allow_overlap and len(trades):
        keep, free_at = [], None
        for i, (enter, leave) in enumerate(zip(trades["entry_time"], trades["exit_time"])):
            if free_at is None or enter >= free_at:
                keep.append(i)
                free_at = leave
        trades = trades.iloc[keep].reset_index(drop=True)

    long = trades["side"] > 0
    entry_mid = (trades["entry_bid"] + trades["entry_ask"]) / 2
    exit_mid = (trades["exit_bid"] + trades["exit_ask"]) / 2
    trades["entry_px"] = np.where(long, trades["entry_ask"], trades["entry_bid"])
    trades["exit_px"] = np.where(long, trades["exit_bid"], trades["exit_ask"])

    trades["gross_ticks"] = trades["side"] * (exit_mid - entry_mid) / TICK
    trades["exec_ticks"] = trades["side"] * (trades["exit_px"] - trades["entry_px"]) / TICK
    trades["spread_cost_ticks"] = trades["gross_ticks"] - trades["exec_ticks"]
    return trades


def apply_costs(trades: pd.DataFrame, fee_per_side: float) -> pd.DataFrame:
    """Add dollar PnL columns: gross, spread cost, fees, and net."""
    if fee_per_side < 0:
        raise ValueError("fee_per_side must be non-negative")
    out = trades.copy()
    out["gross_usd"] = out["gross_ticks"] * TICK_VALUE
    out["spread_usd"] = out["spread_cost_ticks"] * TICK_VALUE
    out["fees_usd"] = 2.0 * fee_per_side
    out["net_usd"] = out["gross_usd"] - out["spread_usd"] - out["fees_usd"]
    return out


def summarize_trades(trades: pd.DataFrame, fee_per_side: float = BASE_FEE_PER_SIDE) -> dict:
    """Trade count and average PnL per trade before and after costs.

    breakeven_fee_per_side is the fee at which net PnL per trade is zero;
    a negative value means the strategy loses money even with free trading.
    gross_t_stat treats trades as independent, which is only roughly true
    when trades do not overlap.
    """
    n = len(trades)
    if n == 0:
        keys = ["gross_win_rate", "gross_per_trade_usd",
                "gross_t_stat", "spread_per_trade_usd", "fees_per_trade_usd",
                "net_per_trade_usd", "net_total_usd", "breakeven_fee_per_side"]
        return {"n_trades": 0, "n_long": 0, "n_short": 0,
                **{k: np.nan for k in keys}}

    t = apply_costs(trades, fee_per_side)
    gross = t["gross_usd"]
    sd = gross.std(ddof=1) if n > 1 else np.nan
    t_stat = gross.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan

    return {
        "n_trades": n,
        "n_long": int((t["side"] > 0).sum()),
        "n_short": int((t["side"] < 0).sum()),
        "gross_win_rate": float((t["gross_ticks"] > 0).mean()),
        "gross_per_trade_usd": float(gross.mean()),
        "gross_t_stat": float(t_stat),
        "spread_per_trade_usd": float(t["spread_usd"].mean()),
        "fees_per_trade_usd": 2.0 * fee_per_side,
        "net_per_trade_usd": float(t["net_usd"].mean()),
        "net_total_usd": float(t["net_usd"].sum()),
        "breakeven_fee_per_side": float((gross - t["spread_usd"]).mean() / 2),
    }


def cost_sensitivity(trades: pd.DataFrame, fees=FEE_SWEEP) -> pd.DataFrame:
    """Net PnL per trade and in total across a range of fees per side."""
    rows = []
    for fee in fees:
        s = summarize_trades(trades, fee)
        rows.append({
            "fee_per_side": float(fee),
            "n_trades": s["n_trades"],
            "net_per_trade_usd": s["net_per_trade_usd"],
            "net_total_usd": s["net_total_usd"],
        })
    return pd.DataFrame(rows)
