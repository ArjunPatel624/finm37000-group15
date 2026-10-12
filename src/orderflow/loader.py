"""Load and clean ES mbp-1 data from Databento, cached as parquet."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CT = "America/Chicago"
OPEN, CLOSE = "08:30", "15:00"
DEFAULT_SYMBOL = "ESZ6"
DEFAULT_CACHE = Path("data/cache")
BOOK_COLS = ["bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]


def session_bounds(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Regular session [08:30, 15:00) Chicago time for a calendar day."""
    d = pd.Timestamp(day).date()
    return (pd.Timestamp(f"{d} {OPEN}", tz=CT), pd.Timestamp(f"{d} {CLOSE}", tz=CT))


def clean_mbp1(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Sort, convert the index to CT, keep [start, end), drop bad quotes.

    Drops rows with missing prices, a zero-size bid or ask, or a crossed or
    locked book (bid >= ask).
    """
    out = df.copy()
    out.index = out.index.tz_convert(CT)
    out = out.sort_index()
    out = out[(out.index >= start) & (out.index < end)]
    out = out.dropna(subset=BOOK_COLS)
    ok = (
        (out["bid_sz_00"] > 0)
        & (out["ask_sz_00"] > 0)
        & (out["bid_px_00"] < out["ask_px_00"])
    )
    return out[ok]


def _fetch_databento(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import databento as db
    from finm37000 import get_databento_api_key  # reads ~/.databento_api_key

    client = db.Historical(get_databento_api_key())
    return client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="mbp-1",
        symbols=symbol,
        stype_in="raw_symbol",
        start=start,
        end=end,
    ).to_df()


def load_es(day, symbol: str = DEFAULT_SYMBOL, cache_dir=DEFAULT_CACHE, fetch=None) -> pd.DataFrame:
    """Return cleaned mbp-1 for one regular session.

    The first call downloads and writes parquet; later calls read the cache.
    `fetch(symbol, start, end) -> DataFrame` is injectable so tests run offline.
    """
    start, end = session_bounds(day)
    path = Path(cache_dir) / f"{symbol}_{start.date()}_mbp-1.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = (fetch or _fetch_databento)(symbol, start, end)
    df = clean_mbp1(raw, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
