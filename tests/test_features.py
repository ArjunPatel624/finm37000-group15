
"""Tests for the ES order-flow features."""

import pandas as pd
from orderflow.features import build_features, compute_book_imbalance


def test_no_lookahead():
    """Future market events must not change past features."""

    times = pd.to_datetime([
        "2026-09-24 08:30:00.200",
        "2026-09-24 08:30:01.200",
        "2026-09-24 08:30:02.200",
    ]).tz_localize("America/Chicago")

    df = pd.DataFrame({
        "bid_sz_00": [120, 80, 150],
        "ask_sz_00": [80, 120, 50],
        "action": ["T", "T", "T"],
        "side": ["B", "A", "B"],
        "size": [10, 6, 8],
    }, index=times)

    # Calculate features using only the original events
    original = build_features(df)

    # Add a trade that happens later in the future
    future_time = pd.Timestamp(
    "2026-09-24 08:30:03.200",
    tz="America/Chicago"
    )

    future_event = pd.DataFrame({
        "bid_sz_00": [500],
        "ask_sz_00": [10],
        "action": ["T"],
        "side": ["B"],
        "size": [1000],
    }, index=pd.DatetimeIndex([future_time]))

    extended = pd.concat([df, future_event])

    # Recalculate features with the future trade included
    updated = build_features(extended)

    # The original feature values must remain unchanged
    pd.testing.assert_frame_equal(
        original,
        updated.loc[original.index]
    )


def test_book_imbalance_uint32():
    """
    Ensure unsigned Databento sizes do not cause
    integer overflow when ask size exceeds bid size.
    """
    df = pd.DataFrame({
        "bid_sz_00": pd.Series([10, 120], dtype="uint32"),
        "ask_sz_00": pd.Series([23, 80], dtype="uint32"),
    })

    result = compute_book_imbalance(df)

    assert abs(result.iloc[0] - (-13 / 33)) < 1e-10
    assert abs(result.iloc[1] - 0.2) < 1e-10
    assert result.between(-1, 1).all()


def test_feature_values():
    """Verify book imbalance, signed volume, and trade intensity."""

    times = pd.to_datetime([
        "2026-09-24 08:30:00.200",
        "2026-09-24 08:30:01.200",
        "2026-09-24 08:30:02.200",
    ]).tz_localize("America/Chicago")

    df = pd.DataFrame({
        "bid_sz_00": pd.Series(
            [120, 80, 150], index=times, dtype="uint32"
        ),
        "ask_sz_00": pd.Series(
            [80, 120, 50], index=times, dtype="uint32"
        ),
        "action": ["T", "T", "T"],
        "side": ["B", "A", "B"],
        "size": [10, 6, 8],
    }, index=times)

    features = build_features(df, window_seconds=5)

    assert len(features) == 3

    # Latest bid size = 150, ask size = 50
    assert abs(features["book_imbalance"].iloc[-1] - 0.5) < 1e-10

    # Signed volume = 10 - 6 + 8 = 12
    assert features["signed_volume_5s"].iloc[-1] == 12

    # Three trade events
    assert features["trade_count_5s"].iloc[-1] == 3

    # All imbalance values must be within [-1, 1]
    assert features["book_imbalance"].between(-1, 1).all()
