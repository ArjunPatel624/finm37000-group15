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


### Data Window and Scope Limitations

The original project scope specifies regular trading sessions from 08:30 to 15:00 CT. However, this preliminary evaluation uses only the first 10 minutes (08:30–08:40 CT) of September 22–24, 2026.

This shorter window was selected to validate the feature engineering, labeling, and out-of-sample evaluation pipeline while keeping data processing and computational requirements manageable.

A complete regular session contains 390 minutes, compared with the 10 minutes evaluated here. Therefore, the current experiment covers approximately 2.56% of each full session.

Extending the analysis to full sessions would require substantially more market data and processing time. The exact billable data size and runtime have not yet been measured. Before retrieving full-session data, the Databento billable-size estimate should be checked to avoid unexpected costs.

The results should therefore be interpreted as preliminary and may not generalize to other intraday periods or full trading sessions.


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



### Excluded Flat-Price Observations

The binary classification experiment removes observations where the forward mid-price change is zero. The following table reports how many valid observations are excluded at each horizon.

| Horizon | Split | Total valid | Flat excluded | Non-flat retained | Flat excluded (%) |
|---|---|---:|---:|---:|---:|
| 1s | Train | 1198 | 467 | 731 | 38.98% |
| 1s | Test | 599 | 175 | 424 | 29.22% |
| 5s | Train | 1190 | 189 | 1001 | 15.88% |
| 5s | Test | 595 | 81 | 514 | 13.61% |
| 10s | Train | 1180 | 158 | 1022 | 13.39% |
| 10s | Test | 590 | 63 | 527 | 10.68% |

A substantial proportion of observations are excluded, particularly at the 1-second horizon. As a result, binary classification accuracy measures directional prediction conditional on a nonzero future price movement.

This conditional accuracy should not be interpreted as achievable real-time trading accuracy because whether the future price moves is unknown at prediction time.

For the cost-aware trading simulation in Issue #9, the Ridge regression predictions of numeric forward mid-price changes provide an alternative signal that includes flat-price observations. Trading decisions can be based on predicted changes exceeding a threshold, with bid-ask spreads, fees, and slippage considered separately. The profitability of such a strategy remains to be tested.


### Statistical Uncertainty: 1-Second Horizon

To assess whether the improvement in 1-second classification accuracy is meaningful, we use a paired block bootstrap with 2,000 resamples and a fixed random seed of 42.

The September 24 test observations are grouped into 10 one-minute blocks. Entire blocks are resampled to preserve some of the dependence between nearby observations.

- Observed logistic regression accuracy improvement: **1.18 percentage points**
- 95% bootstrap confidence interval: **[-6.58, +8.12] percentage points**
- Number of one-minute blocks: **10**

The confidence interval includes zero, indicating that the observed improvement is not statistically distinguishable from the majority-class baseline on this sample.

Because the experiment includes only one testing day and 10 blocks, this confidence interval is preliminary and does not account for uncertainty from retraining the model.


### Logistic Regression Training Diagnostics

To investigate whether the predictive relationships remain stable out of sample, we compare the correlation between logistic regression predicted probabilities and forward mid-price changes during training and testing.

| Horizon | Training correlation | Testing correlation |
|---|---:|---:|
| 1s | 0.1682 | 0.1247 |
| 5s | 0.1169 | -0.0935 |
| 10s | 0.1570 | -0.1817 |

The training correlations are positive across all three horizons. However, the testing correlations become negative at 5 and 10 seconds, suggesting that the relationships learned during training do not generalize consistently to the testing period.

The standardized logistic regression coefficients are:

| Horizon | Book imbalance | Signed volume (5s) | Trade count (5s) |
|---|---:|---:|---:|
| 1s | 0.3208 | -0.0502 | 0.0324 |
| 5s | 0.0840 | 0.0530 | 0.2066 |
| 10s | 0.0269 | 0.0748 | 0.2248 |

At the 1-second horizon, book imbalance has the largest positive coefficient. At longer horizons, trade count receives a larger positive coefficient, while the contribution of book imbalance decreases.

These coefficients describe associations within the fitted model rather than causal relationships. The training correlations are in-sample, and the sign reversal at longer horizons may reflect changing market conditions, overfitting, or sampling noise. With only one testing day, the evidence is insufficient to establish a regime change.


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

The experiments do not demonstrate a statistically distinguishable improvement over simple baselines. Although logistic regression slightly outperforms the majority-class baseline at the 1-second horizon, the block-bootstrap confidence interval includes zero. Performance also deteriorates at longer horizons, so the results do not establish a consistent predictive advantage.

Important limitations:

- Only three 10-minute sessions were evaluated, with one testing day.
- Overlapping forward-return labels create dependence between observations and can overstate statistical significance.
- Binary classification excludes flat-price movements, so its accuracy is conditional on a future move occurring.
- Constant baseline predictions have undefined correlation with forward returns.
- Predictive accuracy does not imply trading profitability.
- Transaction costs, bid-ask spreads, and execution slippage are not evaluated here.

These results are preliminary and do not establish a statistically reliable or economically profitable trading signal.


## Reproducing the Results

The evaluation tables are generated by `scripts/run_issue8.py`.

### Data and Setup

The experiment uses ESZ6 `mbp-1` data for September 22, 23, and 24, 2026, restricted to 08:30–08:40 CT each day.

The script expects the following locally cached parquet files:

- `data/cache/ESZ6_2026-09-22_mbp-1.parquet`
- `data/cache/ESZ6_2026-09-23_mbp-1.parquet`
- `data/cache/ESZ6_2026-09-24_mbp-1.parquet`

These files are not committed to GitHub. The runner checks that they exist and does not automatically download additional market data.

The script uses `build_features`, `build_labels`, and `load_es`. Until PRs #14 and #16 are merged, it can use local copies of their implementations saved as `data/cache/loader_pr14.py` and `data/cache/labels_pr16.py`. After those PRs are integrated, it uses the official modules.

The Python environment must include the project's dependencies, including `scikit-learn`.

### Running the Evaluation

From the repository root, execute:

```bash
python3 scripts/run_issue8.py
```

The script:

1. Loads the cached ESZ6 market data for each day.
2. Builds order-flow features and forward-price labels separately for each session.
3. Trains models using September 22–23 and evaluates them on September 24.
4. Compares logistic regression against the majority-class baseline.
5. Compares Ridge regression against the training-mean baseline.
6. Prints the results and saves the generated tables.

### Generated Outputs

- `results/issue8_classification.csv`
- `results/issue8_regression.csv`

The CSV files contain the numerical results summarized in this report.

**Note:** The current experiment uses three 10-minute windows rather than the complete 08:30–15:00 CT sessions specified in the broader project scope. These are preliminary results intended to validate the evaluation pipeline. A full-session analysis would require additional market data, with its billable size checked before downloading.
