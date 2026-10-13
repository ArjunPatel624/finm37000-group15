
"""Order-flow features for E-mini S&P 500 futures."""

import numpy as np
import pandas as pd

def compute_book_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Calculate top-of-book size imbalance.

    Converts unsigned Databento sizes to float64 to avoid
    integer underflow when ask size exceeds bid size.
    """
    bid_size = df["bid_sz_00"].astype("float64")
    ask_size = df["ask_sz_00"].astype("float64")

    total_size = bid_size + ask_size

    imbalance = (bid_size - ask_size) / total_size.where(
        total_size > 0, np.nan
    )

    return imbalance.rename("book_imbalance")


def compute_signed_volume(df: pd.DataFrame) -> pd.Series:
    """
    Calculate signed aggressive volume for each market-data event.

    B = buyer-initiated trade (+size)
    A = seller-initiated trade (-size)

    Non-trade events and unknown aggressor sides contribute zero.
    """
    direction = df["side"].map({"B": 1, "A": -1}).fillna(0)

    signed_volume = df["size"] * direction

    signed_volume = signed_volume.where(
        df["action"] == "T", 0
    )

    return signed_volume.rename("signed_volume")


def compute_trade_indicator(df: pd.DataFrame) -> pd.Series:
    """
    Return 1 for each trade event and 0 for non-trade events.

    These indicators will later be summed over trailing time
    windows to calculate trade intensity.
    """
    return (df["action"] == "T").astype(int).rename("trade_count")


def build_features(
    df: pd.DataFrame, window_seconds: int = 5
) -> pd.DataFrame:
    """
    Build order-flow features on a one-second time grid.

    Each timestamp t uses only observations at or before t.
    Input should contain one trading session with a
    timezone-aware DatetimeIndex.
    """
    if window_seconds < 1:
        raise ValueError("window_seconds must be positive")

    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Expected a timezone-aware DatetimeIndex")

    df = df.sort_index()

    events = pd.DataFrame({
        "book_imbalance": compute_book_imbalance(df),
        "signed_volume": compute_signed_volume(df),
        "trade_count": compute_trade_indicator(df),
    })

    # Group events into seconds ending at each timestamp
    per_second = events.resample(
        "1s", closed="right", label="right"
    ).agg({
        "book_imbalance": "last",
        "signed_volume": "sum",
        "trade_count": "sum",
    })

    # Carry forward the last known book imbalance
    per_second["book_imbalance"] = (
        per_second["book_imbalance"].ffill()
    )

    # Aggregate trades over trailing seconds
    volume_col = f"signed_volume_{window_seconds}s"
    count_col = f"trade_count_{window_seconds}s"

    per_second[volume_col] = (
        per_second["signed_volume"]
        .rolling(window_seconds, min_periods=1)
        .sum()
    )

    per_second[count_col] = (
        per_second["trade_count"]
        .rolling(window_seconds, min_periods=1)
        .sum()
    )

    return per_second[
        ["book_imbalance", volume_col, count_col]
    ]
