"""Forward mid-price labels for ES futures (issue #7).

Labels live on the same 1-second grid as ``build_features``: a timestamp t
holds the last quote at or before t (bins closed and labelled on the right).
For each horizon h, the label at t is the sign of mid(t + h) - mid(t).

Zero moves are kept as a third class (0). At 1s most ES seconds have no mid
change, so dropping them would discard most of the sample and change the
question being asked. ``align`` can drop them if a two-class view is wanted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK = 0.25  # ES tick size in index points ($12.50 per contract)
DEFAULT_HORIZONS = (1, 5, 10)
CT = "America/Chicago"


def _check_session(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Expected a timezone-aware DatetimeIndex")
    days = pd.unique(df.index.tz_convert(CT).date)
    if len(days) > 1:
        raise ValueError(
            "build_labels expects one session; call it per day and concat, "
            "so forward windows never cross from one day into the next"
        )


def mid_on_grid(df: pd.DataFrame) -> pd.Series:
    """Mid-price on a 1-second grid, matching build_features' convention."""
    df = df.sort_index()
    mid = (df["bid_px_00"].astype("float64") + df["ask_px_00"].astype("float64")) / 2
    return (
        mid.resample("1s", closed="right", label="right")
        .last()
        .ffill()
        .rename("mid")
    )


def build_labels(df: pd.DataFrame, horizons=DEFAULT_HORIZONS) -> pd.DataFrame:
    """Return forward mid changes and their signs for one session.

    Columns per horizon h:
      dmid_{h}s  forward mid change in ticks (useful for #8 correlations, #9 PnL)
      y_{h}s     sign of that change: -1, 0, +1

    Rows in the last h seconds of the session have no forward mid and are
    left as NaN / <NA> for that horizon; ``align`` drops them.
    """
    _check_session(df)
    if any(int(h) != h or h < 1 for h in horizons):
        raise ValueError("horizons must be positive whole seconds")

    mid = mid_on_grid(df)
    out = pd.DataFrame({"mid": mid})
    for h in horizons:
        # Look up by timestamp rather than shifting rows, so this stays correct
        # even if the grid is ever built with gaps.
        fwd = mid.reindex(mid.index + pd.Timedelta(seconds=h)).to_numpy()
        dmid = (fwd - mid.to_numpy()) / TICK
        out[f"dmid_{h}s"] = dmid
        out[f"y_{h}s"] = pd.array(np.sign(dmid), dtype="Int8")
    return out


def align(features: pd.DataFrame, labels: pd.DataFrame, horizon: int,
          drop_zero: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Join features and one horizon's labels; drop rows missing either.

    Returns (X, y). With drop_zero=True, seconds where the mid didn't move
    are removed, leaving a two-class up/down problem.
    """
    y_col = f"y_{horizon}s"
    joined = features.join(labels[[y_col]], how="inner").dropna()
    if drop_zero:
        joined = joined[joined[y_col] != 0]
    return joined.drop(columns=y_col), joined[y_col].astype(int)


def class_balance(labels: pd.DataFrame, horizons=DEFAULT_HORIZONS,
                  show: bool = True) -> pd.DataFrame:
    """Share of down / flat / up labels per horizon (NaN rows excluded)."""
    rows = {}
    for h in horizons:
        y = labels[f"y_{h}s"].dropna().astype(int)
        shares = y.value_counts(normalize=True).reindex([-1, 0, 1], fill_value=0.0)
        rows[f"{h}s"] = {"n": len(y), "down": shares[-1], "flat": shares[0], "up": shares[1]}
    table = pd.DataFrame(rows).T
    table["n"] = table["n"].astype(int)
    if show:
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    return table
