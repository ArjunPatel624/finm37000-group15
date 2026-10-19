# Issue #8: Out-of-Sample Predictive Power

## Experimental Setup

- Instrument: ESZ6 (E-mini S&P 500 futures)
- Data: Databento GLBX.MDP3, mbp-1
- Sessions: September 22–24, 2026, 08:30–08:40 CT
- Training: September 22–23
- Testing: September 24
- Features: book imbalance, 5-second signed volume, and 5-second trade count
- Prediction horizons: 1, 5, and 10 seconds

Features and forward-price labels are generated independently for each session to prevent windows from crossing trading days. Models are trained chronologically without shuffling.

## Classification: Up vs. Down

Flat-price movements are excluded from this binary experiment.

| Horizon | Model | Train | Test | Accuracy | AUC | Return correlation |
|---|---|---:|---:|---:|---:|---:|
| 1s | Majority baseline | 731 | 424 | 0.5189 | 0.5000 | N/A |
| 1s | Logistic regression | 731 | 424 | 0.5307 | 0.5659 | 0.1247 |
| 5s | Majority baseline | 1001 | 514 | 0.5642 | 0.5000 | N/A |
| 5s | Logistic regression | 1001 | 514 | 0.5467 | 0.4973 | -0.0935 |
| 10s | Majority baseline | 1022 | 527 | 0.5275 | 0.5000 | N/A |
| 10s | Logistic regression | 1022 | 527 | 0.4801 | 0.4507 | -0.1817 |

Logistic regression slightly outperforms the baseline at 1 second but underperforms at 5 and 10 seconds.

## Regression: Numeric Price Change

Following Professor Patterson's suggestion, we also predict forward mid-price changes in ES ticks using Ridge regression.

Unlike binary classification, this experiment includes zero-price-change observations.

The baseline predicts the mean forward price change from the training period.

| Horizon | Model | Train | Test | MAE (ticks) | RMSE (ticks) | Correlation |
|---|---|---:|---:|---:|---:|---:|
| 1s | Mean baseline | 1198 | 599 | 1.1421 | 1.5840 | N/A |
| 1s | Ridge regression | 1198 | 599 | 1.1679 | 1.5753 | 0.1016 |
| 5s | Mean baseline | 1190 | 595 | 2.6285 | 3.5320 | N/A |
| 5s | Ridge regression | 1190 | 595 | 2.6519 | 3.5768 | -0.1004 |
| 10s | Mean baseline | 1180 | 590 | 3.7016 | 4.9768 | N/A |
| 10s | Ridge regression | 1180 | 590 | 3.7780 | 5.1070 | -0.1590 |

Ridge regression slightly improves RMSE at 1 second, but not MAE. At 5 and 10 seconds, the mean baseline performs better on both error metrics.

## Conclusions and Limitations

The experiments show limited predictive information at the 1-second horizon, but no consistent improvement over simple baselines.

Important limitations:

- Only three 10-minute sessions were evaluated, with one testing day.
- Overlapping forward-return labels create dependence between observations and can overstate statistical significance.
- Binary classification excludes flat-price movements, so its accuracy is conditional on a future move occurring.
- Constant baseline predictions have undefined correlation with forward returns.
- Predictive accuracy does not imply trading profitability.
- Transaction costs, bid-ask spreads, and execution slippage are not evaluated here.

These results are preliminary and do not establish a statistically reliable or economically profitable trading signal.
