"""Reproduce Issue #8 evaluation from cached ESZ6 market data."""

from pathlib import Path
import importlib
import importlib.util
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
sys.path.insert(0, str(ROOT / "src"))

from orderflow.features import build_features
from orderflow.evaluation import (
    evaluate_horizons,
    split_train_test_by_day,
    fit_ridge_regression,
    evaluate_regression_predictions,
    prepare_binary_labels,
    majority_class_baseline,
    fit_logistic_regression,
    block_bootstrap_accuracy_difference,
    logistic_training_diagnostics,
)

DAYS = ("2026-09-22", "2026-09-23", "2026-09-24")
HORIZONS = (1, 5, 10)


def get_function(module_name, function_name, fallback):
    """Use the merged module, or a local copy of its pending PR."""
    try:
        module = importlib.import_module(f"orderflow.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"orderflow.{module_name}":
            raise
        path = CACHE / fallback
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Merge the corresponding dependency PR first."
            ) from exc
        spec = importlib.util.spec_from_file_location(
            f"issue8_{module_name}", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    return getattr(module, function_name)


def load_data():
    """Build features and labels separately for each trading day."""
    load_es = get_function("loader", "load_es", "loader_pr14.py")
    build_labels = get_function("labels", "build_labels", "labels_pr16.py")

    feature_frames = []
    label_frames = []

    for day in DAYS:
        cache_file = CACHE / f"ESZ6_{day}_mbp-1.parquet"

        # Prevent accidental Databento downloads.
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Missing {cache_file}. This runner requires cached data."
            )

        df = load_es(day, cache_dir=CACHE)

        # Analyze only 08:30-08:40 Chicago time.
        start = pd.Timestamp(f"{day} 08:30", tz="America/Chicago")
        end = start + pd.Timedelta(minutes=10)
        df = df.loc[(df.index >= start) & (df.index < end)]

        if df.empty:
            raise ValueError(f"No observations found for {day}")

        feature_frames.append(build_features(df))
        label_frames.append(build_labels(df, horizons=HORIZONS))

        print(f"{day}: {len(df):,} market events processed")

    return (
        pd.concat(feature_frames).sort_index(),
        pd.concat(label_frames).sort_index(),
    )


def evaluate_regression_horizons(features, labels):
    """Compare mean-return baseline and Ridge across horizons."""
    rows = []

    for h in HORIZONS:
        target = f"dmid_{h}s"

        data = features.join(
            labels[[target]], how="inner"
        ).dropna()

        train, test = split_train_test_by_day(data)

        X_train = train[features.columns]
        X_test = test[features.columns]

        y_train = train[target]
        y_test = test[target]

        baseline = pd.Series(y_train.mean(), index=y_test.index)
        ridge = fit_ridge_regression(X_train, y_train, X_test)

        for name, predictions in [
            ("Mean baseline", baseline),
            ("Ridge regression", ridge),
        ]:
            metrics = evaluate_regression_predictions(
                y_test, predictions
            )

            rows.append({
                "horizon": f"{h}s",
                "model": name,
                "train_samples": len(train),
                "test_samples": len(test),
                **metrics,
            })

    return pd.DataFrame(rows)


def main():
    features, labels = load_data()

    classification = evaluate_horizons(
        features, labels, horizons=HORIZONS
    )

    regression = evaluate_regression_horizons(features, labels)


    # Calculate uncertainty in the 1-second accuracy improvement.
    data_1s = features.join(
        labels[["y_1s", "dmid_1s"]],
        how="inner",
    ).dropna()

    binary_labels = prepare_binary_labels(data_1s["y_1s"])
    data_1s = data_1s.loc[binary_labels.index].copy()
    data_1s["binary_target"] = binary_labels

    train, test = split_train_test_by_day(data_1s)

    X_train = train[features.columns]
    X_test = test[features.columns]
    y_train = train["binary_target"]
    y_test = test["binary_target"]

    baseline_predictions = majority_class_baseline(
        y_train, y_test
    )

    model_predictions, _ = fit_logistic_regression(
        X_train, y_train, X_test
    )

    bootstrap = block_bootstrap_accuracy_difference(
        y_test,
        baseline_predictions,
        model_predictions,
        n_bootstrap=2000,
        seed=42,
    )

    print("\n1-SECOND ACCURACY IMPROVEMENT: BLOCK BOOTSTRAP")
    print(f"Observed improvement: {bootstrap['accuracy_difference']:.4f}")
    print(
        f"95% bootstrap CI: "
        f"[{bootstrap['ci_lower']:.4f}, "
        f"{bootstrap['ci_upper']:.4f}]"
    )
    print(f"One-minute blocks: {bootstrap['n_blocks']}")


    # Examine training-set relationships and fitted feature coefficients.
    diagnostic_rows = []

    for h in HORIZONS:
        y_col = f"y_{h}s"
        return_col = f"dmid_{h}s"

        data = features.join(
            labels[[y_col, return_col]],
            how="inner",
        ).dropna()

        binary_labels = prepare_binary_labels(data[y_col])
        data = data.loc[binary_labels.index].copy()
        data["binary_target"] = binary_labels

        train, _ = split_train_test_by_day(data)

        diagnostics = logistic_training_diagnostics(
            train[features.columns],
            train["binary_target"],
            train[return_col],
        )

        diagnostic_rows.append({
            "horizon": f"{h}s",
            "train_correlation": diagnostics["train_correlation"],
            **{
                f"coef_{name}": float(value)
                for name, value in diagnostics["coefficients"].items()
            },
        })

    diagnostics_df = pd.DataFrame(diagnostic_rows)

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    diagnostics_df.to_csv(
        output_dir / "issue8_diagnostics.csv",
        index=False,
    )

    print("\nLOGISTIC REGRESSION TRAINING DIAGNOSTICS")
    print(diagnostics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    ))

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    classification.to_csv(
        output_dir / "issue8_classification.csv",
        index=False,
    )

    regression.to_csv(
        output_dir / "issue8_regression.csv",
        index=False,
    )

    print("\nEvaluation tables saved to results/")

    print("\nCLASSIFICATION RESULTS")
    print(classification.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    print("\nREGRESSION RESULTS")
    print(regression.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))


    # Count flat-price observations excluded from binary classification.
    flat_rows = []

    for h in HORIZONS:
        y_col = f"y_{h}s"
        return_col = f"dmid_{h}s"

        data = features.join(
            labels[[y_col, return_col]],
            how="inner",
        ).dropna()

        train, test = split_train_test_by_day(data)

        for split_name, subset in [
            ("Train", train),
            ("Test", test),
        ]:
            total = len(subset)
            flat = int((subset[y_col] == 0).sum())

            flat_rows.append({
                "horizon": f"{h}s",
                "split": split_name,
                "total_valid": total,
                "flat_excluded": flat,
                "nonflat_retained": total - flat,
                "flat_pct": 100 * flat / total,
            })

    flat_df = pd.DataFrame(flat_rows)

    flat_df.to_csv(
        ROOT / "results" / "issue8_flat_observations.csv",
        index=False,
    )

    print("\nFLAT-PRICE OBSERVATIONS EXCLUDED")
    print(flat_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    ))

if __name__ == "__main__":
    main()
