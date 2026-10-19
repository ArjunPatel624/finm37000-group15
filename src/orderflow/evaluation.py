"""
Evaluate predictive power of ES order-flow features.

Methodology
-----------
- Train on earlier Chicago trading days and test on the final day.
- Never shuffle observations or fit preprocessing on test data.
- Compare a majority-class baseline against logistic regression.
- Evaluate 1s, 5s, and 10s prediction horizons.
- Report accuracy, AUC, and correlation with forward returns.

Limitations
-----------
- Flat forward-price movements are excluded from the binary analysis.
  Results therefore describe direction prediction conditional on a
  nonzero future move, not unconditional trading performance.
- Overlapping forward-return windows introduce serial dependence.
  Naive significance estimates may overstate predictive evidence.
- Evaluation does not include transaction costs or execution effects.
- Forward labels must be calculated separately for each session
  to prevent prediction windows from crossing session boundaries.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def split_train_test_by_day(df: pd.DataFrame):
    """
    Split time-series data into training and testing sets.

    Train on all earlier trading days.
    Test on the final trading day.
    No random shuffling is performed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")

    if df.index.tz is None:
        raise ValueError("Expected timezone-aware timestamps")

    df = df.sort_index()

    # Identify trading dates in Chicago time
    chicago_dates = df.index.tz_convert("America/Chicago").normalize()
    days = chicago_dates.unique()

    if len(days) < 2:
        raise ValueError("At least two trading days are required")

    # Reserve the final day for testing
    test_day = days[-1]

    train = df.loc[chicago_dates < test_day].copy()
    test = df.loc[chicago_dates == test_day].copy()

    return train, test

def majority_class_baseline(
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.Series:
    """
    Predict the most common training label for every test observation.

    Labels:
        1 = upward price movement
        0 = downward or non-upward price movement

    The test labels are never used to choose the prediction.
    """
    if y_train.empty:
        raise ValueError("Training labels cannot be empty")

    if not y_train.isin([0, 1]).all():
        raise ValueError("Training labels must be binary (0 or 1)")

    # Identify the majority class using training data only
    majority_class = int(y_train.mode().iloc[0])

    # Predict that class for every test observation
    return pd.Series(
        majority_class,
        index=y_test.index,
        name="baseline_prediction"
    )

def fit_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """
    Train logistic regression on historical observations only.

    Returns predicted directions and upward-movement probabilities
    for the test observations.
    """
    if X_train.empty or X_test.empty:
        raise ValueError("Training and testing features cannot be empty")

    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Training and testing feature columns must match")

    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels must have matching indices")

    if y_train.isna().any() or not y_train.isin([0, 1]).all():
        raise ValueError("Training labels must be binary and non-missing")

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both classes")

    if not X_train.select_dtypes(include="number").shape[1] == X_train.shape[1]:
        raise ValueError("All features must be numeric")

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Features cannot contain missing values")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )

    # Both scaling and model fitting use training data only
    model.fit(X_train, y_train)

    predictions = pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="logistic_prediction",
    )

    probabilities = pd.Series(
        model.predict_proba(X_test)[:, 1],
        index=X_test.index,
        name="probability_up",
    )

    return predictions, probabilities

def evaluate_predictions(
    y_true: pd.Series,
    predictions: pd.Series,
    probabilities: pd.Series,
    forward_returns: pd.Series,
) -> dict:
    """Calculate classification and forward-return metrics."""

    if not (
        y_true.index.equals(predictions.index)
        and y_true.index.equals(probabilities.index)
        and y_true.index.equals(forward_returns.index)
    ):
        raise ValueError("Evaluation inputs must have matching indices")

    accuracy = accuracy_score(y_true, predictions)

    # AUC requires both classes to appear in the test labels
    auc = (
        roc_auc_score(y_true, probabilities)
        if y_true.nunique() == 2
        else np.nan
    )

    # Correlation between predicted probability and forward return
    # Correlation is undefined for constant predictions or returns
    if (
        probabilities.nunique() < 2
        or forward_returns.nunique() < 2
    ):
        correlation = np.nan
    else:
        correlation = probabilities.corr(forward_returns)

    return {
        "accuracy": float(accuracy),
        "auc": float(auc),
        "correlation": float(correlation),
    }

def prepare_binary_labels(
    labels: pd.Series,
) -> pd.Series:
    """
    Convert three-class price movements to binary labels.

    Original: -1 = down, 0 = flat, 1 = up
    Binary:    0 = down,           1 = up

    Flat observations are excluded.
    """
    if not labels.dropna().isin([-1, 0, 1]).all():
        raise ValueError("Expected labels -1, 0, or 1")

    nonzero = labels.dropna()
    nonzero = nonzero[nonzero != 0]

    return (nonzero == 1).astype(int).rename("direction")

def evaluate_horizons(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    horizons=(1, 5, 10),
) -> pd.DataFrame:
    """
    Evaluate order-flow prediction for each forecast horizon.

    Train on earlier days and test on the final day.
    Flat-price observations are excluded from the binary analysis.

    Returns accuracy, AUC, and forward-return correlation
    for both logistic regression and the majority baseline.
    """
    results = []

    for h in horizons:
        y_col = f"y_{h}s"
        return_col = f"dmid_{h}s"

        # Align feature observations with forward labels and returns
        data = features.join(
            labels[[y_col, return_col]],
            how="inner",
        ).dropna()

        # Keep up/down observations, excluding flat moves
        binary_labels = prepare_binary_labels(data[y_col])
        data = data.loc[binary_labels.index].copy()
        data["binary_target"] = binary_labels

        # Split chronologically by trading day
        train, test = split_train_test_by_day(data)

        feature_cols = list(features.columns)

        X_train = train[feature_cols]
        y_train = train["binary_target"]

        X_test = test[feature_cols]
        y_test = test["binary_target"]

        forward_returns = test[return_col]

        # Majority-class baseline
        baseline_pred = majority_class_baseline(
            y_train, y_test
        )

        baseline_results = evaluate_predictions(
            y_test,
            baseline_pred,
            baseline_pred.astype(float),
            forward_returns,
        )

        # Logistic regression
        model_pred, model_prob = fit_logistic_regression(
            X_train, y_train, X_test
        )

        model_results = evaluate_predictions(
            y_test,
            model_pred,
            model_prob,
            forward_returns,
        )

        # Store results for both approaches
        for name, metrics in [
            ("Majority baseline", baseline_results),
            ("Logistic regression", model_results),
        ]:
            results.append({
                "horizon": f"{h}s",
                "model": name,
                "train_samples": len(train),
                "test_samples": len(test),
                **metrics,
            })

    return pd.DataFrame(results)

def fit_ridge_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    """
    Predict numeric forward mid-price changes in ES ticks.

    Scaling and model fitting use training data only.
    """
    if X_train.empty or X_test.empty:
        raise ValueError("Feature datasets cannot be empty")

    if not X_train.columns.equals(X_test.columns):
        raise ValueError("Feature columns must match")

    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and targets must align")

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    model.fit(X_train, y_train)

    return pd.Series(
        model.predict(X_test),
        index=X_test.index,
        name="predicted_dmid",
    )

def evaluate_regression_predictions(
    y_true: pd.Series,
    predictions: pd.Series,
) -> dict:
    """Evaluate numeric forward mid-price predictions in ticks."""
    if y_true.empty:
        raise ValueError("Evaluation data cannot be empty")

    if not y_true.index.equals(predictions.index):
        raise ValueError("Target and prediction indices must match")

    if y_true.isna().any() or predictions.isna().any():
        raise ValueError("Evaluation data cannot contain missing values")

    errors = y_true - predictions

    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(np.mean(errors**2)))

    if y_true.nunique() < 2 or predictions.nunique() < 2:
        correlation = np.nan
    else:
        correlation = float(predictions.corr(y_true))

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
    }