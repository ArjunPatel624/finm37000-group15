"""Tests for forward mid-price labels (issue #7)."""
import numpy as np
import pandas as pd
import pytest

from orderflow.features import build_features
from orderflow.labels import align, build_labels, class_balance

CT = "America/Chicago"


def make_session() -> pd.DataFrame:
    """Twelve events, one per second, with a quiet gap at 08:30:06-07.

    Bid/ask chosen so mids are: 5000.125 x3, 5000.375 x2, (gap), 5000.125 ...
    """
    secs = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
    bids = [5000.00, 5000.00, 5000.00, 5000.25, 5000.25, 5000.00,
            5000.00, 5000.00, 4999.75, 4999.75, 5000.00, 5000.00]
    times = pd.Timestamp("2026-09-24 08:30:00.500", tz=CT) + pd.to_timedelta(secs, "s")
    return pd.DataFrame({
        "bid_px_00": bids,
        "ask_px_00": [b + 0.25 for b in bids],
        "bid_sz_00": pd.Series([10] * 12, dtype="uint32").to_numpy(),
        "ask_sz_00": pd.Series([12] * 12, dtype="uint32").to_numpy(),
        "action": ["T", "A"] * 6,
        "side": ["B", "N"] * 6,
        "size": [1, 0] * 6,
    }, index=times)


def test_grid_matches_features():
    df = make_session()
    assert build_labels(df).index.equals(build_features(df).index)


def test_label_values():
    labels = build_labels(make_session(), horizons=(1,))
    # mid at 08:30:03 is 5000.125, at 08:30:04 it is 5000.375: up one tick
    assert labels.loc["2026-09-24 08:30:03-05:00", "dmid_1s"] == 1.0
    assert labels.loc["2026-09-24 08:30:03-05:00", "y_1s"] == 1
    # flat second stays as class 0, not dropped
    assert labels.loc["2026-09-24 08:30:01-05:00", "y_1s"] == 0
    # down one tick
    assert labels.loc["2026-09-24 08:30:05-05:00", "y_1s"] == -1


def test_quiet_seconds_carry_last_mid():
    labels = build_labels(make_session(), horizons=(1,))
    # no events in (08:30:06, 08:30:08]; mid carries forward from 08:30:06
    assert labels.loc["2026-09-24 08:30:07-05:00", "mid"] == labels.loc["2026-09-24 08:30:06-05:00", "mid"]


@pytest.mark.parametrize("h", [1, 5, 10])
def test_last_h_seconds_dropped(h):
    df = make_session()
    labels = build_labels(df, horizons=(h,))
    assert labels[f"y_{h}s"].isna().sum() == h
    assert labels[f"y_{h}s"].iloc[-h:].isna().all()
    X, y = align(build_features(df), labels, h)
    assert len(X) == len(y) == len(labels) - h
    assert X.index.equals(y.index)


def test_no_lookahead_into_label_origin():
    """Changing quotes after t must not change mid(t), only labels before t."""
    df = make_session()
    base = build_labels(df, horizons=(1,))
    later = df.copy()
    later.iloc[-1, later.columns.get_loc("bid_px_00")] = 4990.0
    later.iloc[-1, later.columns.get_loc("ask_px_00")] = 4990.25
    changed = build_labels(later, horizons=(1,))
    pd.testing.assert_series_equal(base["mid"].iloc[:-1], changed["mid"].iloc[:-1])


def test_drop_zero():
    df = make_session()
    _, y_all = align(build_features(df), build_labels(df, horizons=(1,)), 1)
    _, y_nz = align(build_features(df), build_labels(df, horizons=(1,)), 1, drop_zero=True)
    assert (y_all == 0).any()
    assert not (y_nz == 0).any()
    assert len(y_nz) == (y_all != 0).sum()


def test_rejects_multiple_days():
    a = make_session()
    b = make_session()
    b.index = b.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="one session"):
        build_labels(pd.concat([a, b]))


def test_class_balance(capsys):
    table = class_balance(build_labels(make_session()))
    assert list(table.index) == ["1s", "5s", "10s"]
    assert np.allclose(table[["down", "flat", "up"]].sum(axis=1), 1.0)
    assert "flat" in capsys.readouterr().out
