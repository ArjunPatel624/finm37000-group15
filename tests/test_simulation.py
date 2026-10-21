"""Tests for the cost-aware trading simulation (issue #9)."""
import numpy as np
import pandas as pd
import pytest

from orderflow.features import build_features
from orderflow.simulation import (
    TICK_VALUE,
    apply_costs,
    cost_sensitivity,
    quotes_on_grid,
    simulate_trades,
    summarize_trades,
)

CT = "America/Chicago"


def make_session(day="2026-09-24", bids=None, spread_ticks=None) -> pd.DataFrame:
    """One event per second at hh:mm:ss.5, so the quote at grid t is event t-1."""
    if bids is None:
        bids = [5000.00, 5000.00, 5000.25, 5000.50, 5000.50, 5000.25,
                5000.25, 5000.00, 5000.00, 5000.25, 5000.25, 5000.50]
    if spread_ticks is None:
        spread_ticks = [1] * len(bids)
    times = pd.Timestamp(f"{day} 08:30:00.500", tz=CT) + pd.to_timedelta(range(len(bids)), "s")
    return pd.DataFrame({
        "bid_px_00": bids,
        "ask_px_00": [b + 0.25 * s for b, s in zip(bids, spread_ticks)],
        "bid_sz_00": np.full(len(bids), 10, dtype="uint32"),
        "ask_sz_00": np.full(len(bids), 12, dtype="uint32"),
        "action": ["T", "A"] * (len(bids) // 2) + ["T"] * (len(bids) % 2),
        "side": ["B", "N"] * (len(bids) // 2) + ["B"] * (len(bids) % 2),
        "size": [1, 0] * (len(bids) // 2) + [1] * (len(bids) % 2),
    }, index=times)


def grid_time(day, sec):
    return pd.Timestamp(f"{day} 08:30:{sec:02d}", tz=CT)


def test_quotes_grid_matches_features():
    df = make_session()
    q = quotes_on_grid(df)
    assert q.index.equals(build_features(df).index)
    assert (q["ask"] > q["bid"]).all()
    assert np.allclose(q["mid"], (q["bid"] + q["ask"]) / 2)
    # grid 08:30:03 holds the event at 08:30:02.5 (bid 5000.25)
    assert q.loc[grid_time("2026-09-24", 3), "bid"] == 5000.25


def test_quotes_grid_rejects_multiple_days():
    two_days = pd.concat([make_session("2026-09-23"), make_session("2026-09-24")])
    with pytest.raises(ValueError, match="one session"):
        quotes_on_grid(two_days)


def test_long_trade_prices_and_pnl():
    q = quotes_on_grid(make_session())
    t0 = grid_time("2026-09-24", 2)              # bid 5000.00 / ask 5000.25
    signal = pd.Series([2.0], index=[t0])
    trades = simulate_trades(signal, q, horizon=2, threshold=0.5)

    assert len(trades) == 1
    tr = trades.iloc[0]
    assert tr["side"] == 1
    assert tr["entry_px"] == 5000.25             # buy at the ask
    assert tr["exit_time"] == grid_time("2026-09-24", 4)
    assert tr["exit_px"] == 5000.50              # sell at the bid (5000.50 / 5000.75)
    assert tr["gross_ticks"] == 2.0              # mid 5000.125 -> 5000.625
    assert tr["spread_cost_ticks"] == 1.0        # half a tick in, half a tick out
    assert tr["exec_ticks"] == 1.0


def test_short_trade_prices_and_pnl():
    q = quotes_on_grid(make_session())
    t0 = grid_time("2026-09-24", 5)              # bid 5000.50 / ask 5000.75
    trades = simulate_trades(pd.Series([-3.0], index=[t0]), q, horizon=3, threshold=0.5)

    tr = trades.iloc[0]
    assert tr["side"] == -1
    assert tr["entry_px"] == 5000.50             # sell at the bid
    assert tr["exit_px"] == 5000.25              # buy back at the ask (5000.00 / 5000.25)
    assert tr["gross_ticks"] == 2.0              # mid 5000.625 -> 5000.125, short gains
    assert tr["exec_ticks"] == 1.0


def test_wider_spread_costs_more():
    df = make_session(spread_ticks=[1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    q = quotes_on_grid(df)
    trades = simulate_trades(pd.Series([2.0], index=[grid_time("2026-09-24", 3)]),
                             q, horizon=1, threshold=0.5)
    # entry spread 3 ticks, exit spread 1 tick -> (3 + 1) / 2 = 2 ticks paid
    assert trades.iloc[0]["spread_cost_ticks"] == 2.0


def test_threshold_is_strict_and_symmetric():
    q = quotes_on_grid(make_session())
    times = [grid_time("2026-09-24", s) for s in (1, 2, 3, 4)]
    signal = pd.Series([0.5, -0.5, 0.51, -0.51], index=times)
    trades = simulate_trades(signal, q, horizon=1, threshold=0.5, allow_overlap=True)
    assert trades["signal_time"].tolist() == times[2:]
    assert trades["side"].tolist() == [1, -1]


def test_one_position_at_a_time():
    q = quotes_on_grid(make_session())
    signal = pd.Series(1.0, index=q.index)
    no_overlap = simulate_trades(signal, q, horizon=3, threshold=0.0)
    overlap = simulate_trades(signal, q, horizon=3, threshold=0.0, allow_overlap=True)

    gaps = no_overlap["entry_time"].diff().dropna()
    assert (gaps >= pd.Timedelta(seconds=3)).all()
    assert (no_overlap["entry_time"].iloc[1:].to_numpy()
            >= no_overlap["exit_time"].iloc[:-1].to_numpy()).all()
    assert len(overlap) == len(q) - 3            # every second that has an exit
    assert len(no_overlap) < len(overlap)


def test_no_trade_when_exit_falls_after_session_end():
    day1, day2 = make_session("2026-09-23"), make_session("2026-09-24")
    q = pd.concat([quotes_on_grid(day1), quotes_on_grid(day2)])
    last_day1 = q.loc["2026-09-23"].index[-1]
    signal = pd.Series([5.0, 5.0], index=[last_day1 - pd.Timedelta(seconds=2), last_day1])
    trades = simulate_trades(signal, q, horizon=5, threshold=0.0, allow_overlap=True)
    assert trades.empty


def test_delay_shifts_entry_and_exit():
    q = quotes_on_grid(make_session())
    t0 = grid_time("2026-09-24", 2)
    trades = simulate_trades(pd.Series([2.0], index=[t0]), q, horizon=2, threshold=0.5, delay=1)
    tr = trades.iloc[0]
    assert tr["signal_time"] == t0
    assert tr["entry_time"] == grid_time("2026-09-24", 3)
    assert tr["exit_time"] == grid_time("2026-09-24", 5)


def test_costs_and_breakeven_fee():
    q = quotes_on_grid(make_session())
    trades = simulate_trades(pd.Series([2.0], index=[grid_time("2026-09-24", 2)]),
                             q, horizon=2, threshold=0.5)
    costed = apply_costs(trades, fee_per_side=1.0)
    assert costed["gross_usd"].iloc[0] == 2 * TICK_VALUE
    assert costed["spread_usd"].iloc[0] == TICK_VALUE
    assert costed["net_usd"].iloc[0] == 2 * TICK_VALUE - TICK_VALUE - 2.0

    s = summarize_trades(trades, fee_per_side=1.0)
    assert s["n_trades"] == 1 and s["n_long"] == 1
    assert s["breakeven_fee_per_side"] == TICK_VALUE / 2
    assert s["net_per_trade_usd"] == s["gross_per_trade_usd"] - s["spread_per_trade_usd"] - 2.0


def test_fee_sweep_is_linear():
    q = quotes_on_grid(make_session())
    trades = simulate_trades(pd.Series(1.0, index=q.index), q, horizon=2, threshold=0.0)
    sens = cost_sensitivity(trades, fees=(0.0, 1.0, 2.5))
    base = sens["net_per_trade_usd"].iloc[0]
    assert np.allclose(sens["net_per_trade_usd"], base - 2 * sens["fee_per_side"])
    assert (sens["n_trades"] == len(trades)).all()


def test_no_trades_summary():
    q = quotes_on_grid(make_session())
    trades = simulate_trades(pd.Series(0.1, index=q.index), q, horizon=1, threshold=1.0)
    assert trades.empty
    s = summarize_trades(trades)
    assert s["n_trades"] == 0 and np.isnan(s["net_per_trade_usd"])
    assert cost_sensitivity(trades)["n_trades"].eq(0).all()


def test_rejects_bad_arguments():
    q = quotes_on_grid(make_session())
    signal = pd.Series(1.0, index=q.index)
    with pytest.raises(ValueError):
        simulate_trades(signal, q, horizon=0, threshold=0.0)
    with pytest.raises(ValueError):
        simulate_trades(signal, q, horizon=1, threshold=-1.0)
    with pytest.raises(ValueError):
        simulate_trades(signal, q, horizon=1, threshold=0.0, delay=-1)
