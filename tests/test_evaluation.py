"""Tests for predictive-power evaluation."""

import pandas as pd
import numpy as np

from orderflow.evaluation import (
    split_train_test_by_day,
    majority_class_baseline,
    fit_logistic_regression,
    evaluate_predictions,
    prepare_binary_labels,
    evaluate_horizons,
    fit_ridge_regression,
    evaluate_regression_predictions
)



def test_split_train_test_by_day():
    """Earlier days train; the last day is reserved for testing."""

    times = pd.to_datetime([
        "2026-09-22 08:30:00",
        "2026-09-22 08:30:01",
        "2026-09-23 08:30:00",
        "2026-09-23 08:30:01",
        "2026-09-24 08:30:00",
        "2026-09-24 08:30:01",
    ]).tz_localize("America/Chicago")

    df = pd.DataFrame({
        "book_imbalance": [0.2, -0.1, 0.3, -0.2, 0.5, -0.4],
        "signed_volume_5s": [10, -5, 8, -3, 12, -7],
    }, index=times)

    train, test = split_train_test_by_day(df)

    assert len(train) == 4
    assert len(test) == 2

    assert train.index.max() < test.index.min()

    assert train.index.normalize().nunique() == 2
    assert test.index.normalize().nunique() == 1

def test_majority_class_baseline():
    """Baseline must predict the majority class from training only."""

    y_train = pd.Series([1, 1, 1, 0, 0])
    y_test = pd.Series([0, 0, 1, 0])

    predictions = majority_class_baseline(y_train, y_test)

    # Training majority is 1 (upward movement)
    assert predictions.tolist() == [1, 1, 1, 1]

    # Predictions must preserve the test index
    assert predictions.index.equals(y_test.index)

    # Changing test labels must not change predictions
    different_test = pd.Series([1, 1, 1, 1])
    updated = majority_class_baseline(y_train, different_test)

    assert updated.tolist() == predictions.tolist()

def test_logistic_regression():
    """Check predictions, probabilities, and test index alignment."""

    X_train = pd.DataFrame({
        "book_imbalance": [-0.8, -0.6, 0.7, 0.9, -0.5, 0.6],
        "signed_volume_5s": [-10, -8, 9, 12, -6, 8],
    })

    y_train = pd.Series([0, 0, 1, 1, 0, 1])

    X_test = pd.DataFrame({
        "book_imbalance": [-0.7, 0.8],
        "signed_volume_5s": [-9, 10],
    }, index=[100, 101])

    predictions, probabilities = fit_logistic_regression(
        X_train, y_train, X_test
    )

    # One prediction for each test observation
    assert len(predictions) == 2
    assert len(probabilities) == 2

    # Preserve the original test index
    assert predictions.index.equals(X_test.index)
    assert probabilities.index.equals(X_test.index)

    # Predictions must be binary
    assert predictions.isin([0, 1]).all()

    # Probabilities must be between 0 and 1
    assert probabilities.between(0, 1).all()

    # Stronger positive signals should receive higher probabilities
    assert probabilities.iloc[1] > probabilities.iloc[0]

def test_evaluate_predictions():
    """Verify accuracy, AUC, and forward-return correlation."""

    index = pd.RangeIndex(4)

    y_true = pd.Series([0, 1, 0, 1], index=index)
    predictions = pd.Series([0, 1, 1, 1], index=index)
    probabilities = pd.Series([0.1, 0.9, 0.6, 0.8], index=index)
    forward_returns = pd.Series(
        [-0.02, 0.03, -0.01, 0.02],
        index=index,
    )

    results = evaluate_predictions(
        y_true,
        predictions,
        probabilities,
        forward_returns,
    )

    # Three correct predictions out of four
    assert abs(results["accuracy"] - 0.75) < 1e-10

    # AUC = 1.0 because both positive examples
    # receive higher probabilities than both negatives
    assert abs(results["auc"] - 1.0) < 1e-10

    # Check correlation against pandas directly
    expected_correlation = probabilities.corr(forward_returns)

    assert abs(
        results["correlation"] - expected_correlation
    ) < 1e-10

def test_prepare_binary_labels():
    """Verify conversion of down, flat, and up labels."""

    labels = pd.Series(
        [-1, 0, 1, -1, 1, None],
        index=[10, 11, 12, 13, 14, 15],
    )

    result = prepare_binary_labels(labels)

    # Flat and missing labels should be removed
    assert result.index.tolist() == [10, 12, 13, 14]

    # Down becomes 0, up becomes 1
    assert result.tolist() == [0, 1, 0, 1]

    # Retain the original observation indices
    assert result.name == "direction"

def test_evaluate_horizons():
    """Evaluate three horizons using chronological synthetic data."""

    days = ["2026-09-22", "2026-09-23", "2026-09-24"]

    times = pd.DatetimeIndex([
        t
        for day in days
        for t in pd.date_range(
            f"{day} 08:30:00",
            periods=8,
            freq="s",
            tz="America/Chicago",
        )
    ])

    imbalance = [-0.8, 0.7, -0.6, 0.9, -0.5, 0.6, -0.7, 0.8] * 3

    features = pd.DataFrame({
        "book_imbalance": imbalance,
        "signed_volume_5s": [x * 10 for x in imbalance],
        "trade_count_5s": [2, 3, 1, 4, 2, 3, 1, 4] * 3,
    }, index=times)

    labels = pd.DataFrame(index=times)

    for h in (1, 5, 10):
        returns = pd.Series(
            [1.0 if x > 0 else -1.0 for x in imbalance],
            index=times,
        )

        labels[f"dmid_{h}s"] = returns
        labels[f"y_{h}s"] = returns.astype(int)

    results = evaluate_horizons(
        features,
        labels,
        horizons=(1, 5, 10),
    )

    assert len(results) == 6
    assert set(results["horizon"]) == {"1s", "5s", "10s"}
    assert set(results["model"]) == {
        "Majority baseline",
        "Logistic regression",
    }

    assert (results["train_samples"] == 16).all()
    assert (results["test_samples"] == 8).all()

    assert results["accuracy"].between(0, 1).all()
    assert results["auc"].between(0, 1).all()

def test_split_uses_chicago_dates():
    """UTC timestamps must be split by Chicago trading dates."""

    times = pd.to_datetime([
        "2026-09-23 23:30:00+00:00",
        "2026-09-24 00:30:00+00:00",
        "2026-09-24 23:30:00+00:00",
        "2026-09-25 00:30:00+00:00",
    ], utc=True)

    df = pd.DataFrame({
        "book_imbalance": [0.1, 0.2, 0.3, 0.4],
    }, index=times)

    train, test = split_train_test_by_day(df)

    # First two UTC timestamps are September 23 in Chicago.
    # Last two are September 24 in Chicago.
    assert len(train) == 2
    assert len(test) == 2

    assert train.index.equals(times[:2])
    assert test.index.equals(times[2:])

    assert train.index.max() < test.index.min()

def test_evaluate_horizons_with_flat_and_missing_labels():
    """Flat and unavailable forward returns must be excluded."""

    days = ["2026-09-22", "2026-09-23", "2026-09-24"]

    times = pd.DatetimeIndex([
        t
        for day in days
        for t in pd.date_range(
            f"{day} 08:30:00",
            periods=6,
            freq="s",
            tz="America/Chicago",
        )
    ])

    features = pd.DataFrame({
        "book_imbalance": [-0.8, 0.0, 0.7, -0.6, 0.9, 0.1] * 3,
        "signed_volume_5s": [-10, 0, 8, -7, 12, 0] * 3,
        "trade_count_5s": [2, 1, 3, 2, 4, 1] * 3,
    }, index=times)

    labels = pd.DataFrame({
        "y_1s": [-1, 0, 1, -1, 1, None] * 3,
        "dmid_1s": [-1.0, 0.0, 1.0, -1.0, 1.0, None] * 3,
    }, index=times)

    results = evaluate_horizons(
        features,
        labels,
        horizons=(1,),
    )

    # Two models for one prediction horizon
    assert len(results) == 2

    # Four usable observations per day
    assert (results["train_samples"] == 8).all()
    assert (results["test_samples"] == 4).all()

    assert results["accuracy"].between(0, 1).all()
    assert results["auc"].between(0, 1).all()

def test_fit_ridge_regression():
    idx = pd.date_range(
        "2026-09-22 13:30:00",
        periods=8,
        freq="s",
        tz="UTC",
    )

    X = pd.DataFrame(
        {
            "book_imbalance": [-0.8, -0.5, -0.2, 0.1, 0.4, 0.7, -0.3, 0.6],
            "signed_volume_5s": [-10, -5, -2, 3, 7, 12, -4, 9],
            "trade_count_5s": [4, 6, 5, 8, 10, 12, 7, 11],
        },
        index=idx,
    )

    y = pd.Series(
        [-2.0, -1.0, -0.5, 0.0, 1.0, 2.0],
        index=idx[:6],
    )

    predictions = fit_ridge_regression(
        X_train=X.iloc[:6],
        y_train=y,
        X_test=X.iloc[6:],
    )

    assert len(predictions) == 2
    assert predictions.index.equals(X.iloc[6:].index)
    assert predictions.name == "predicted_dmid"
    assert predictions.notna().all()

def test_evaluate_regression_predictions():
    idx = pd.date_range(
        "2026-09-24",
        periods=3,
        freq="s",
        tz="UTC",
    )

    y_true = pd.Series([1.0, -1.0, 0.0], index=idx)
    predictions = pd.Series([0.5, -0.5, 0.0], index=idx)

    metrics = evaluate_regression_predictions(
        y_true,
        predictions,
    )

    assert np.isclose(metrics["mae"], 1 / 3)
    assert np.isclose(metrics["rmse"], np.sqrt(1 / 6))
    assert np.isclose(metrics["correlation"], 1.0)