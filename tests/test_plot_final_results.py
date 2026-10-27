"""Check plots against the committed research results without rerunning models."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_CSV = REPOSITORY_ROOT / "results" / "issue8_classification.csv"
SCRIPT = REPOSITORY_ROOT / "scripts" / "plot_final_results.py"

spec = importlib.util.spec_from_file_location("plot_final_results", SCRIPT)
plots = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plots
spec.loader.exec_module(plots)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def write_classification(tmp_path, frame):
    path = tmp_path / "classification.csv"
    frame.to_csv(path, index=False)
    return path


def assert_classification_bars_match_csv(figure, expected):
    """Match visible series to source rows, independent of CSV row ordering."""
    axes = figure.get_axes()
    for metric, scale in (("accuracy", 100), ("auc", 1)):
        matching_axes = [
            axis for axis in axes
            if metric in f"{axis.get_ylabel()} {axis.get_title()}".lower()
        ]
        assert len(matching_axes) == 1
        axis = matching_axes[0]
        horizons = [
            f"{label.get_text().split()[0].removesuffix('s')}s"
            for label in axis.get_xticklabels()
        ]
        assert horizons == ["1s", "5s", "10s"]
        containers = {container.get_label(): container for container in axis.containers}
        for model in expected["model"].unique():
            assert model in containers
            values = expected.loc[expected["model"] == model].set_index("horizon")
            heights = [bar.get_height() for bar in containers[model]]
            np.testing.assert_allclose(
                heights, values.loc[horizons, metric].to_numpy() * scale,
                rtol=1e-12, atol=1e-12,
            )
    visible_text = " ".join(text.get_text() for text in figure.findobj(matplotlib.text.Text))
    assert "flat" in visible_text.lower()
    assert "excluded" in visible_text.lower()


@pytest.mark.parametrize("shuffle", [False, True], ids=["source-order", "shuffled"])
def test_predictive_bars_reproduce_source_results(tmp_path, shuffle):
    source = pd.read_csv(CLASSIFICATION_CSV)
    path = CLASSIFICATION_CSV
    if shuffle:
        path = write_classification(tmp_path, source.sample(frac=1, random_state=42))
    frame = plots.load_classification(path)
    figure = plots.plot_classification(frame)
    assert_classification_bars_match_csv(figure, source)


@pytest.mark.parametrize(
    "corruption",
    ["missing-column", "duplicate-row", "missing-row", "bad-accuracy", "bad-auc", "nonnumeric"],
)
def test_classification_rejects_invalid_input(tmp_path, corruption):
    frame = pd.read_csv(CLASSIFICATION_CSV)
    if corruption == "missing-column":
        frame = frame.drop(columns="accuracy")
    elif corruption == "duplicate-row":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif corruption == "missing-row":
        frame = frame.iloc[1:]
    elif corruption == "bad-accuracy":
        frame.loc[0, "accuracy"] = 1.1
    elif corruption == "bad-auc":
        frame.loc[0, "auc"] = -0.1
    elif corruption == "nonnumeric":
        frame["accuracy"] = frame["accuracy"].astype(object)
        frame.loc[0, "accuracy"] = "invalid"
    with pytest.raises(ValueError):
        plots.load_classification(write_classification(tmp_path, frame))


def test_profitability_rejects_unrelated_schema():
    # No local Issue #9 result exists: do not manufacture a profitability fixture.
    with pytest.raises(ValueError):
        plots.load_profitability(CLASSIFICATION_CSV)


def test_profitability_rejects_missing_metric_columns(tmp_path):
    path = tmp_path / "incomplete_profitability.csv"
    pd.read_csv(CLASSIFICATION_CSV)[["horizon"]].to_csv(path, index=False)
    with pytest.raises(ValueError):
        plots.load_profitability(path)


def test_profitability_bars_reproduce_source_results_when_available():
    path = REPOSITORY_ROOT / "results" / "issue9_summary_full.csv"
    if not path.is_file():
        pytest.skip("The optional local Issue #9 source CSV is not available")
    source = pd.read_csv(path)
    figure = plots.plot_profitability(plots.load_profitability(path))
    series = (
        ("Gross P&L (mid to mid)", "gross_per_trade_usd", 1),
        ("Spread cost (deduction)", "spread_per_trade_usd", -1),
        ("Fees (round trip; deduction)", "fees_per_trade_usd", -1),
        ("Net P&L (after costs)", "net_per_trade_usd", 1),
    )
    for axis in figure.get_axes():
        horizon = f"{axis.get_title().split('-')[0]}s"
        rows = source.loc[source["horizon"] == horizon].set_index("threshold")
        by_display_label = {str(value).replace("_", " "): value for value in rows.index}
        thresholds = [
            by_display_label[label.get_text().split("\n")[0]]
            for label in axis.get_yticklabels()
        ]
        rows = rows.loc[thresholds]
        active = rows["n_trades"] > 0
        containers = {container.get_label(): container for container in axis.containers}
        for label, column, sign in series:
            widths = [bar.get_width() for bar in containers[label]]
            np.testing.assert_allclose(widths, sign * rows.loc[active, column], rtol=1e-12)
    visible_text = " ".join(text.get_text() for text in figure.findobj(matplotlib.text.Text))
    assert "USD" in visible_text
    assert "round trip" in visible_text.lower()


def test_main_generates_prediction_plot_when_issue9_is_missing(tmp_path, monkeypatch, capsys):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    copied_source = result_dir / CLASSIFICATION_CSV.name
    copied_source.write_bytes(CLASSIFICATION_CSV.read_bytes())
    original_hash = hashlib.sha256(copied_source.read_bytes()).hexdigest()
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    assert plots.main() == 0

    image_path = result_dir / "figures" / "issue8_predictive_performance.png"
    assert image_path.is_file()
    image = plt.imread(image_path)
    assert image.shape[0] > 0 and image.shape[1] > 0
    assert not (result_dir / "figures" / "issue9_trading_profitability.png").exists()
    output = capsys.readouterr()
    message = f"{output.out} {output.err}".lower()
    assert "issue9_summary_full.csv" in message
    assert any(word in message for word in ("missing", "not found", "unavailable", "skip"))
    assert hashlib.sha256(copied_source.read_bytes()).hexdigest() == original_hash


def test_main_reports_invalid_input(tmp_path, monkeypatch, capsys):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    invalid = pd.read_csv(CLASSIFICATION_CSV).drop(columns="accuracy")
    invalid.to_csv(result_dir / CLASSIFICATION_CSV.name, index=False)
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    assert plots.main() == 1
    output = capsys.readouterr()
    assert "accuracy" in f"{output.out} {output.err}".lower()
