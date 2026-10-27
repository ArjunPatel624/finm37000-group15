"""Cost-aware trading simulation for issue #9.

Signal: Ridge predictions of the forward mid change in ticks, from #8. The
model is trained on the earlier days and applied to the last day, exactly as
in run_issue8.py. Ridge is used rather than the logistic model because it is
fit on every second, including flat moves, which a real-time strategy can't
filter out (see the review discussion on #18).

Thresholds are set without looking at the test day: quantiles of |prediction|
on the training days, plus one threshold where the predicted move covers the
round-trip cost (a 1-tick spread plus two fees).

If the predictions are small, Ridge's intercept (the average move on the
training days) can decide the sign of every trade, so the strategy is just
long or short all day. Check n_long / n_short, and use --center to subtract
the training-mean prediction so trades come from the order-flow features only.

Usage, from the repository root:
    python3 scripts/run_issue9.py                      # 08:30-08:40 CT, as in #8
    python3 scripts/run_issue9.py --window-minutes 0   # full 08:30-15:00 session
    python3 scripts/run_issue9.py --fee 1.50 --delay 1 --center
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orderflow.evaluation import fit_ridge_regression, split_train_test_by_day  # noqa: E402
from orderflow.features import build_features  # noqa: E402
from orderflow.labels import build_labels  # noqa: E402
from orderflow.loader import DEFAULT_SYMBOL, load_es, session_bounds  # noqa: E402
from orderflow.simulation import (  # noqa: E402
    BASE_FEE_PER_SIDE,
    FEE_SWEEP,
    TICK_VALUE,
    cost_sensitivity,
    quotes_on_grid,
    simulate_trades,
    summarize_trades,
)

DAYS = ("2026-09-22", "2026-09-23", "2026-09-24")
HORIZONS = (1, 5, 10)
QUANTILES = (0.50, 0.75, 0.90, 0.95)
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
COVERAGE_TOLERANCE = pd.Timedelta(minutes=1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", nargs="+", default=list(DAYS),
                   help="trading days; the last one is the test day")
    p.add_argument("--window-minutes", type=int, default=10,
                   help="minutes from 08:30 CT to use each day (0 = full session)")
    p.add_argument("--fee", type=float, default=BASE_FEE_PER_SIDE,
                   help="base fee per side per contract in dollars")
    p.add_argument("--delay", type=int, default=0,
                   help="seconds between signal and fill")
    p.add_argument("--center", action="store_true",
                   help="subtract the training-mean prediction from the signal")
    p.add_argument("--allow-overlap", action="store_true",
                   help="allow a new trade while one is still open")
    p.add_argument("--allow-download", action="store_true",
                   help="let load_es download days missing from data/cache")
    return p.parse_args()


def check_coverage(df, start, end, cache_file):
    """Raise if the cached data does not span the requested window.

    load_es always caches the full session, so this only fails for a cache
    file made some other way, e.g. a hand-made 10-minute file.
    """
    if (df.empty or df.index.min() > start + COVERAGE_TOLERANCE
            or df.index.max() < end - COVERAGE_TOLERANCE):
        if df.empty:
            covered = "no data"
        else:
            first, last = df.index.min().tz_convert(start.tz), df.index.max().tz_convert(start.tz)
            covered = f"only {first:%H:%M:%S}-{last:%H:%M:%S} CT"
        raise ValueError(
            f"{cache_file.name} has {covered}, but this run needs "
            f"{start:%H:%M}-{end:%H:%M} CT. Delete the file and rerun with "
            "--allow-download to cache the full session."
        )


def build_dataset(days, window_minutes, allow_download):
    """Features, labels, and quotes, built separately for each session."""
    features, labels, quotes = [], [], []
    for day in days:
        start, end = session_bounds(day)
        cache_file = CACHE / f"{DEFAULT_SYMBOL}_{start.date()}_mbp-1.parquet"
        if not cache_file.exists() and not allow_download:
            raise FileNotFoundError(
                f"Missing {cache_file}. Pass --allow-download to fetch it "
                "(load_es checks the billable size first)."
            )

        df = load_es(day, cache_dir=CACHE)
        if window_minutes:
            end = min(end, start + pd.Timedelta(minutes=window_minutes))
        check_coverage(df, start, end, cache_file)
        df = df.loc[(df.index >= start) & (df.index < end)]
        if df.empty:
            raise ValueError(f"No observations for {day} in the chosen window")

        features.append(build_features(df))
        labels.append(build_labels(df, horizons=HORIZONS))
        quotes.append(quotes_on_grid(df))
        print(f"{day}: {len(df):,} events, {start:%H:%M}-{end:%H:%M} CT")

    return (pd.concat(features).sort_index(),
            pd.concat(labels).sort_index(),
            pd.concat(quotes).sort_index())


def ridge_signal(features, labels, h, center=False):
    """Train Ridge on the earlier days.

    Returns (train predictions, test predictions, test dmid, train mean dmid).
    """
    target = f"dmid_{h}s"
    data = features.join(labels[[target]], how="inner").dropna()
    train, test = split_train_test_by_day(data)
    cols = list(features.columns)

    # One fit, predicted on both splits; scaling uses the training rows only.
    pred = fit_ridge_regression(train[cols], train[target],
                                pd.concat([train[cols], test[cols]]))
    if center:
        pred = pred - pred.loc[train.index].mean()
    return pred.loc[train.index], pred.loc[test.index], test[target], train[target].mean()


def thresholds_for(train_pred, fee):
    """Thresholds in ticks, chosen from the training days only."""
    abs_pred = train_pred.abs()
    out = {f"q{int(q * 100)}": float(abs_pred.quantile(q)) for q in QUANTILES}
    out["covers_cost"] = 1.0 + 2.0 * fee / TICK_VALUE
    return out


def main():
    args = parse_args()
    features, labels, quotes = build_dataset(args.days, args.window_minutes, args.allow_download)

    signal_rows, summary_rows, sensitivity, all_trades = [], [], [], []
    for h in HORIZONS:
        train_pred, test_pred, test_dmid, train_drift = ridge_signal(
            features, labels, h, center=args.center)
        cost_ticks = 1.0 + 2.0 * args.fee / TICK_VALUE
        signal_rows.append({
            "horizon": f"{h}s",
            "test_seconds": len(test_pred),
            "train_mean_dmid_ticks": float(train_drift),
            "test_corr_with_dmid": float(test_pred.corr(test_dmid)),
            "pred_std_ticks": float(test_pred.std()),
            "pred_max_abs_ticks": float(test_pred.abs().max()),
            "share_above_cost": float((test_pred.abs() > cost_ticks).mean()),
        })

        for name, thr in thresholds_for(train_pred, args.fee).items():
            trades = simulate_trades(test_pred, quotes, h, thr,
                                     delay=args.delay, allow_overlap=args.allow_overlap)
            tag = {"horizon": f"{h}s", "threshold": name, "threshold_ticks": thr}
            summary_rows.append({**tag, **summarize_trades(trades, args.fee)})
            sensitivity.append(cost_sensitivity(trades, FEE_SWEEP).assign(**tag))
            all_trades.append(trades.assign(**tag))

    signal_df = pd.DataFrame(signal_rows)
    summary = pd.DataFrame(summary_rows)
    sens = pd.concat(sensitivity, ignore_index=True)
    trades = pd.concat(all_trades, ignore_index=True)

    RESULTS.mkdir(exist_ok=True)
    signal_df.to_csv(RESULTS / "issue9_signal.csv", index=False)
    summary.to_csv(RESULTS / "issue9_summary.csv", index=False)
    sens.to_csv(RESULTS / "issue9_cost_sensitivity.csv", index=False)
    trades.to_csv(RESULTS / "issue9_trades.csv", index=False)

    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print(f"\nSetup: test day {args.days[-1]}, fee ${args.fee:.2f}/side, "
          f"delay {args.delay}s, overlap {'on' if args.allow_overlap else 'off'}, "
          f"centered {'yes' if args.center else 'no'}, "
          f"tick ${TICK_VALUE:.2f}")

    print("\nSIGNAL (Ridge, test day)")
    print(signal_df.to_string(index=False, float_format=fmt))

    print(f"\nPNL PER TRADE, BEFORE AND AFTER COSTS (fee ${args.fee:.2f}/side)")
    cols = ["horizon", "threshold", "threshold_ticks", "n_trades", "n_long",
            "n_short", "gross_win_rate",
            "gross_per_trade_usd", "gross_t_stat", "spread_per_trade_usd",
            "fees_per_trade_usd", "net_per_trade_usd", "net_total_usd",
            "breakeven_fee_per_side"]
    print(summary[cols].to_string(index=False, float_format=fmt))

    print("\nNET PNL PER TRADE ($) BY FEE PER SIDE")
    pivot = sens.pivot_table(index=["horizon", "threshold"], columns="fee_per_side",
                             values="net_per_trade_usd", sort=False, dropna=False)
    pivot.columns = [f"${c:.2f}" for c in pivot.columns]
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    n_none = int((summary["n_trades"] == 0).sum())
    if n_none:
        print(f"\n{n_none} horizon/threshold combinations produced no trades.")
    print(f"\nTables saved to {RESULTS.relative_to(ROOT)}/issue9_*.csv")


if __name__ == "__main__":
    main()
