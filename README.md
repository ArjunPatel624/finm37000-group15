# finm37000-group15

Group 15 project for FINM 37000: Futures and Related Derivatives.

## Goal

We want to know whether short-term order flow in E-mini S&P 500 futures (ES) predicts the direction of the mid-price over the next few seconds, and whether any edge survives realistic trading costs.

Using top-of-book data for the ESZ6 contract, we will:

1. build order-flow features: book size imbalance, signed aggressive volume, and trade intensity;
2. label the sign of the mid-price move 1s, 5s, and 10s ahead;
3. test whether the features beat a majority-class baseline out of sample, training on earlier days and testing on later ones;
4. simulate trading on the signal, crossing the spread and paying fees, to see what is left after costs.

"No edge after costs" is an acceptable outcome; we will report whatever we find.

The issues build on each other in this order: data access (#1–#3) → scope (#4) → data loader (#5) → features and labels (#6, #7) → evaluation (#8) → trading simulation (#9) → results and write-up (#10), with an optional regime breakdown (#11).

## Scope

The scope is proposed in #4 and is final once agreed there.

- **Contract:** ESZ6 (December 2026 E-mini S&P 500). Tick size is 0.25 points, or $12.50 per contract.
- **Data:** Databento `GLBX.MDP3`, `mbp-1` (top of book plus trades).
- **Window:** 2–3 consecutive regular sessions, 8:30–15:00 CT. Dates are still to be agreed.
- **Horizons:** 1s, 5s, 10s.

## Data

Data is pulled from Databento and cached as parquet in `data/`. It is never committed.

A sample pull of one hour of ESZ6 (Sept 24, 8:30–9:30 CT) from `GLBX.MDP3` gave these sizes:

| Schema | Size |
|---|---|
| `trades` | 2.9 MB |
| `mbp-1` (top of book) | 95.6 MB |

`mbp-1` is about 33× larger than `trades`, so each run should be scoped to a few days or specific hours. Check `client.metadata.get_billable_size()` before pulling a new window.

## How to run

These instructions describe how the finished project will run. They will be updated as the code is built.

1. **Get the code and install dependencies** in a fresh Python 3.12+ environment:
   ```
   git clone https://github.com/ArjunPatel624/finm37000-group15.git
   cd finm37000-group15
   pip install git+https://github.com/pattersonem/finm37000-autumn-2026
   pip install -r requirements.txt
   ```
2. **Save your Databento API key** on one line in a file named `.databento_api_key` in your home folder (`~/` on Mac/Linux, `C:\Users\<you>\` on Windows). Never commit it.
3. **Run the analysis.** Open `notebooks/es_orderflow.ipynb`, set the trading day(s) at the top, and run all cells. The notebook:
   - loads and cleans the data with `load_es(day)` (the first run downloads, later runs read from the cache);
   - builds the features and labels;
   - prints the results table for each horizon;
   - runs the cost-aware trading simulation and saves plots to `results/`.

## Repository layout

```
data/        local data cache (not committed)
notebooks/   analysis and results notebooks
src/         loader, features, labels, evaluation, simulation
```
