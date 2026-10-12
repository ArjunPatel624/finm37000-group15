import numpy as np
import pandas as pd

from orderflow.loader import CT, clean_mbp1, load_es, session_bounds


def make_raw() -> pd.DataFrame:
    # Sept is CDT (UTC-5): 13:30 UTC = 08:30 CT, 20:00 UTC = 15:00 CT.
    rows = [
        ("2026-09-24 19:59:59", 5000.00, 5000.25, 5, 5),      # keep (listed first: tests sorting)
        ("2026-09-24 13:00:00", 5000.00, 5000.25, 5, 5),      # before open
        ("2026-09-24 13:30:00.5", 5000.00, 5000.25, 10, 12),  # keep
        ("2026-09-24 13:30:01", 5000.00, 5000.25, 10, 0),     # zero ask size
        ("2026-09-24 13:30:02", 5000.50, 5000.25, 10, 10),    # crossed
        ("2026-09-24 13:30:03", np.nan, 5000.25, 10, 10),     # missing bid
        ("2026-09-24 20:00:00", 5000.00, 5000.25, 5, 5),      # at close (end exclusive)
    ]
    idx = pd.to_datetime([r[0] for r in rows], utc=True, format="ISO8601")
    return pd.DataFrame(
        {
            "bid_px_00": [r[1] for r in rows],
            "ask_px_00": [r[2] for r in rows],
            "bid_sz_00": [r[3] for r in rows],
            "ask_sz_00": [r[4] for r in rows],
        },
        index=idx,
    )


def test_clean_filters_and_orders():
    start, end = session_bounds("2026-09-24")
    out = clean_mbp1(make_raw(), start, end)
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert str(out.index.tz) == CT
    assert out.index[0].hour == 8 and out.index[0].minute == 30


def test_load_es_uses_cache(tmp_path):
    calls = []

    def fake_fetch(symbol, start, end):
        calls.append(symbol)
        return make_raw()

    first = load_es("2026-09-24", cache_dir=tmp_path, fetch=fake_fetch)
    second = load_es("2026-09-24", cache_dir=tmp_path, fetch=fake_fetch)
    assert calls == ["ESZ6"]  # second call came from parquet
    assert len(first) == len(second) == 2
    assert (tmp_path / "ESZ6_2026-09-24_mbp-1.parquet").exists()
