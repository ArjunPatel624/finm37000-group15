"""Plot saved research results without running models or accessing market data.

Run from the repository root: ``python3 scripts/plot_final_results.py``.
The optional Issue #9 input is only read when it exists locally; this script
never fetches it from PR #19. All outputs are 300 dpi PNGs in results/figures/.

Verified Issue #8 columns: horizon, model, train_samples, test_samples,
accuracy, auc, correlation. The correlation column is not plotted.
Issue #9's schema was checked against PR #19, commit 2f471f06768222698b3e3bf03e829a7b4a81bb68:
https://github.com/ArjunPatel624/finm37000-group15/pull/19
Its *_per_trade_usd columns already include the contract size and tick value;
fees_per_trade_usd is the round-trip fee, not the fee per side.
"""

from pathlib import Path
import os
import sys
import tempfile

# Support headless runs and avoid writing matplotlib's cache into the user's home.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "finm-plots-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "finm-plots-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ("1s", "5s", "10s")
MODELS = ("Logistic regression", "Majority baseline")
PNL_COLUMNS = (
    "gross_per_trade_usd",
    "spread_per_trade_usd",
    "fees_per_trade_usd",
    "net_per_trade_usd",
)
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#7b838b",
    "axes.axisbelow": True,
    "grid.color": "#dce1e5",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.facecolor": "white",
}


def _read_csv(path, required):
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name}: no result rows")
    return frame


def _numeric(frame, columns):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")


def _validate_counts(frame, columns):
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError(f"{', '.join(columns)} must contain nonnegative whole counts")


def _validate_horizons(frame):
    if set(frame["horizon"]) != set(HORIZONS):
        raise ValueError("Expected exactly the 1s, 5s, and 10s horizons")


def load_classification(path):
    """Validate every plotted metric and enforce paired model comparisons."""
    frame = _read_csv(
        path, ("horizon", "model", "train_samples", "test_samples", "accuracy", "auc")
    )
    _validate_horizons(frame)
    if set(frame["model"]) != set(MODELS):
        raise ValueError(f"Expected models: {', '.join(MODELS)}")
    if frame.duplicated(["horizon", "model"]).any() or len(frame) != 6:
        raise ValueError("Expected one row per horizon and model (six rows)")
    _numeric(frame, ("train_samples", "test_samples", "accuracy", "auc"))
    _validate_counts(frame, ("train_samples", "test_samples"))
    if (frame[["train_samples", "test_samples"]] == 0).any().any():
        raise ValueError("Classification requires positive train/test sample counts")
    if (frame.groupby("horizon")[["train_samples", "test_samples"]].nunique() != 1).any().any():
        raise ValueError("Models must use matching train/test sample counts at each horizon")
    metrics = frame[["accuracy", "auc"]].to_numpy(dtype=float)
    if not np.isfinite(metrics).all() or ((metrics < 0) | (metrics > 1)).any():
        raise ValueError("Accuracy and AUC must be finite values between 0 and 1")
    return frame


def load_profitability(path):
    """Read dollar amounts directly and verify net = gross - spread - fees."""
    frame = _read_csv(path, ("horizon", "threshold", "n_trades", *PNL_COLUMNS))
    _validate_horizons(frame)
    if frame["threshold"].isna().any() or frame["threshold"].astype(str).str.strip().eq("").any():
        raise ValueError("Every profitability row must identify its signal threshold")
    if frame.duplicated(["horizon", "threshold"]).any():
        raise ValueError("Duplicate horizon/threshold profitability rows")
    _numeric(frame, ("n_trades", *PNL_COLUMNS))
    _validate_counts(frame, ("n_trades",))
    traded = frame.loc[frame["n_trades"] > 0]
    if not np.isfinite(traded[list(PNL_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Rows with trades require finite gross, spread, fees, and net values")
    if (traded[["spread_per_trade_usd", "fees_per_trade_usd"]] < 0).any().any():
        raise ValueError("Spread costs and fees must be nonnegative cost amounts")
    expected = traded[PNL_COLUMNS[0]] - traded[PNL_COLUMNS[1]] - traded[PNL_COLUMNS[2]]
    if not np.allclose(expected, traded[PNL_COLUMNS[3]], rtol=1e-7, atol=1e-8):
        raise ValueError("Inconsistent P&L: net must equal gross minus spread minus fees")
    return frame


def plot_classification(frame):
    """Return a figure of conditional test accuracy and ROC AUC."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.3))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.73, wspace=0.24)
        fig.suptitle("Predictive performance by forecast horizon", x=0.08, y=0.96, ha="left", fontsize=18)
        fig.text(0.08, 0.895, "Up vs. down · flat-price movements excluded", fontsize=12)
        counts = frame.groupby("horizon")["test_samples"].first()
        labels = [f"{h[:-1]} second{'s' if h != '1s' else ''}\nn = {int(counts[h]):,}" for h in HORIZONS]
        x = np.arange(len(HORIZONS))
        for ax, metric, scale, title, ylabel in zip(
            axes,
            ("accuracy", "auc"),
            (100, 1),
            ("(a) Directional accuracy", "(b) ROC AUC"),
            ("Test accuracy (%)", "ROC AUC (unitless)"),
        ):
            for i, (model, color, hatch) in enumerate(zip(MODELS, ("#0072B2", "#747f89"), (None, "///"))):
                values = frame.loc[frame["model"] == model].set_index("horizon").loc[list(HORIZONS), metric] * scale
                bars = ax.bar(x + (i - 0.5) * 0.35, values, 0.33, label=model,
                              color=color, hatch=hatch, edgecolor="white", linewidth=0.7)
                annotations = [f"{value:.2f}%" if metric == "accuracy" else f"{value:.3f}" for value in values]
                ax.bar_label(bars, labels=annotations, padding=5, fontsize=9)
            ax.set(title=title, ylabel=ylabel, ylim=(0, scale), xticks=x, xticklabels=labels)
            ax.grid(axis="y")
            ax.set_xlabel("Forecast horizon · test observations", labelpad=10)
            if metric == "accuracy":
                ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.07, 0.855), ncol=2)
        fig.text(0.08, 0.11, "Accuracy is conditional on a nonzero future mid-price move; it is not an estimate of trading profitability.", fontsize=9, color="#46515b")
        fig.text(0.08, 0.07, "Point estimates only; the aggregate CSV does not provide uncertainty intervals.", fontsize=9, color="#46515b")
        fig.text(0.08, 0.03, "Source: results/issue8_classification.csv", fontsize=9, color="#46515b")
        return fig


def _threshold_key(value):
    value = str(value)
    if value.startswith("q") and value[1:].isdigit():
        return (0, int(value[1:]), value)
    return (1, 0, value)


def plot_profitability(frame):
    """Plot every threshold; costs are negative contributions, not extra P&L."""
    with plt.rc_context(STYLE):
        max_rows = max(len(frame.loc[frame["horizon"] == horizon]) for horizon in HORIZONS)
        fig, axes = plt.subplots(1, 3, figsize=(16, max(8, 2.1 + 1.35 * max_rows)), sharex=True)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.22, top=0.77, wspace=0.36)
        fig.suptitle("Trading profitability before and after transaction costs", x=0.09, y=0.96, ha="left", fontsize=19)
        fig.text(0.09, 0.915, "Ridge signal · full-session summary · one-contract round trips", fontsize=12)
        colors = ("#0072B2", "#D55E00", "#CC79A7", "#333b44")
        names = ("Gross P&L (mid to mid)", "Spread cost (deduction)", "Fees (round trip; deduction)", "Net P&L (after costs)")
        signs = (1, -1, -1, 1)
        plotted_values = [0.0]
        for ax, horizon in zip(axes, HORIZONS):
            rows = frame.loc[frame["horizon"] == horizon].copy()
            rows = rows.set_index("threshold").loc[sorted(rows["threshold"], key=_threshold_key)]
            y = np.arange(len(rows))
            active = rows["n_trades"].to_numpy() > 0
            for i, (column, sign, color, name) in enumerate(zip(PNL_COLUMNS, signs, colors, names)):
                values = sign * rows.loc[active, column].to_numpy(dtype=float)
                plotted_values.extend(values)
                bars = ax.barh(y[active] + (i - 1.5) * 0.18, values, height=0.16,
                               color=color, label=name, hatch="///" if i == 2 else None)
                ax.bar_label(bars, labels=[f"{value:,.2f}" for value in values], padding=4, fontsize=8)
            for position in y[~active]:
                ax.text(0.5, position, "No trades; per-trade values undefined", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=8, color="#46515b")
            labels = [f"{str(threshold).replace('_', ' ')}\nn = {int(row['n_trades']):,}" for threshold, row in rows.iterrows()]
            ax.set(title=f"{horizon[:-1]}-second horizon", yticks=y, yticklabels=labels,
                   ylim=(len(rows) - 0.5, -0.5))
            ax.axvline(0, color="#46515b", linewidth=0.8)
            ax.grid(axis="x")
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        low, high = min(plotted_values), max(plotted_values)
        span = max(high - low, 1)
        axes[0].set_xlim(low - 0.25 * span, high + 0.25 * span)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.08, 0.88), ncol=2)
        fig.supxlabel("Average contribution (USD per one-contract round trip)", y=0.155)
        fig.text(0.09, 0.10, "Spread and fees are shown as negative deductions: net P&L = gross P&L − spread cost − fees. n = completed round trips.", fontsize=9, color="#46515b")
        fig.text(0.09, 0.065, "q50–q95: training-signal magnitude quantiles; covers cost: the strategy's cost-covering signal threshold. All reported thresholds are shown.", fontsize=9, color="#46515b")
        fig.text(0.09, 0.03, "Source: results/issue9_summary_full.csv. Per-trade results are undefined when n = 0.", fontsize=9, color="#46515b")
        return fig


def _save(figure, path):
    try:
        figure.savefig(path, dpi=300, metadata={"Software": "scripts/plot_final_results.py"})
    finally:
        plt.close(figure)
    print(f"Saved {path}")


def main():
    """Generate available figures; missing optional Issue #9 input is not an error."""
    results = ROOT / "results"
    output = results / "figures"
    try:
        classification = load_classification(results / "issue8_classification.csv")
        output.mkdir(parents=True, exist_ok=True)
        _save(plot_classification(classification), output / "issue8_predictive_performance.png")
        issue9 = results / "issue9_summary_full.csv"
        if not issue9.is_file():
            print(
                "Skipping Issue #9 trading profitability figure: missing local input "
                f"{issue9}. PR #19 may not have merged. "
                "Issue #8 figure was generated; rerun after the CSV is available locally."
            )
            return 0
        profitability = load_profitability(issue9)
        _save(plot_profitability(profitability), output / "issue9_trading_profitability.png")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"Cannot generate final results figures: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
