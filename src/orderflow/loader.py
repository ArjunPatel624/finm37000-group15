"""Load and clean ES mbp-1 data from Databento, cached as parquet."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import pandas as pd

CT = "America/Chicago"
OPEN, CLOSE = "08:30", "15:00"
DEFAULT_SYMBOL = "ESZ6"
DEFAULT_CACHE = Path("data/cache")
BOOK_COLS = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
MAX_DOWNLOAD_BYTES = 2_000_000_000  # refuse a single pull above ~2 GB


def session_bounds(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Regular session [08:30, 15:00) Chicago time for a calendar day."""
    d = pd.Timestamp(day).date()
    return (pd.Timestamp(f"{d} {OPEN}", tz=CT), pd.Timestamp(f"{d} {CLOSE}", tz=CT))


def clean_mbp1(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, return_summary: bool = False):
    """Sort, convert the index to CT, keep [start, end), drop bad quotes.

    Drops rows with missing prices, a zero-size bid or ask, or a crossed or
    locked book (bid >= ask). With return_summary=True, also returns a dict
    of how many rows each step removed (and how many were trade rows).
    """
    out = df.copy()
    out.index = out.index.tz_convert(CT)
    out = out.sort_index()
    n_raw = len(out)

    out = out[(out.index >= start) & (out.index < end)]
    n_session = len(out)
    has_action = "action" in out.columns
    trades_in = int((out["action"] == "T").sum()) if has_action else 0

    out = out.dropna(subset=BOOK_COLS)
    n_quotes = len(out)
    out = out[(out["bid_sz_00"] > 0) & (out["ask_sz_00"] > 0)]
    n_sizes = len(out)
    out = out[out["bid_px_00"] < out["ask_px_00"]]

    if not return_summary:
        return out
    summary = {
        "raw_rows": n_raw,
        "outside_session": n_raw - n_session,
        "missing_quotes": n_session - n_quotes,
        "zero_size": n_quotes - n_sizes,
        "crossed_or_locked": n_sizes - len(out),
        "kept": len(out),
    }
    if has_action:
        summary["trade_rows_dropped"] = trades_in - int((out["action"] == "T").sum())
    return out, summary


def format_summary(s: dict) -> str:
    parts = [
        f"raw={s['raw_rows']:,}",
        f"outside_session={s['outside_session']:,}",
        f"missing_quotes={s['missing_quotes']:,}",
        f"zero_size={s['zero_size']:,}",
        f"crossed_or_locked={s['crossed_or_locked']:,}",
        f"kept={s['kept']:,}",
    ]
    if "trade_rows_dropped" in s:
        parts.append(f"trade_rows_dropped={s['trade_rows_dropped']:,}")
    return "[load_es] cleaning: " + ", ".join(parts)


def _fetch_databento(symbol, start, end, max_bytes=MAX_DOWNLOAD_BYTES, client=None, verbose=True):
    """Download mbp-1, first checking the billable size against max_bytes."""
    if client is None:
        import databento as db
        from finm37000 import get_databento_api_key  # reads ~/.databento_api_key

        client = db.Historical(get_databento_api_key())
    params = dict(
        dataset="GLBX.MDP3", schema="mbp-1", symbols=symbol,
        stype_in="raw_symbol", start=start, end=end,
    )
    size = client.metadata.get_billable_size(**params)
    if verbose:
        print(f"[load_es] billable size for {symbol} {start.date()}: {size / 1e6:,.1f} MB")
    if size > max_bytes:
        raise ValueError(
            f"billable size {size:,} bytes exceeds max_bytes={max_bytes:,}; "
            "narrow the window or raise max_bytes"
        )
    return client.timeseries.get_range(**params).to_df()


def load_es(day, symbol: str = DEFAULT_SYMBOL, cache_dir=DEFAULT_CACHE, fetch=None,
            max_bytes: int = MAX_DOWNLOAD_BYTES, verbose: bool = True) -> pd.DataFrame:
    """Return cleaned mbp-1 for one regular session.

    The first call checks the billable size, downloads, prints a cleaning
    summary, and writes parquet; later calls read the cache.
    `fetch(symbol, start, end) -> DataFrame` is injectable so tests run offline.
    """
    start, end = session_bounds(day)
    path = Path(cache_dir) / f"{symbol}_{start.date()}_mbp-1.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if fetch is None:
        fetch = partial(_fetch_databento, max_bytes=max_bytes, verbose=verbose)
    raw = fetch(symbol, start, end)
    df, summary = clean_mbp1(raw, start, end, return_summary=True)
    if verbose:
        print(format_summary(summary))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
