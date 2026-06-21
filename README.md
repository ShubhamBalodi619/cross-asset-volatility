# Cross-Asset Volatility Modeling \& VaR Validation

A from-scratch study of asymmetric volatility dynamics and Value-at-Risk
calibration across three asset classes — **S\&P 500** (equity), **WTI crude**
(commodity), and **Prologis** (REIT) — built around GJR-GARCH, a hand-rolled
Markov-switching GARCH, and formal out-of-sample backtesting.

\---

## What this project shows

The work is organized as a sequence of questions, each answered with a model
and a statistical test rather than an assertion.

### 1\. Do the three assets share the same volatility signature?

A GJR-GARCH(1,1,1) with Student's-t innovations is selected by BIC over four
candidate specifications (GARCH/GJR × Normal/t) for every asset, and validated
with Ljung-Box tests on standardized and squared-standardized residuals. The
assets turn out to have **distinct** signatures:

|Asset|leverage γ|tail ν|persistence|reading|
|-|-|-|-|-|
|S\&P 500|0.28|5.7|0.98|strongest leverage, fattest tails|
|WTI crude|0.07|6.4|0.98|most shock-reactive|
|Prologis|0.06|7.2|0.99|most persistent|

The leverage effect is statistically significant for all three (γ p < 0.001).

### 2\. Does the 95% VaR actually cover what it claims?

A rolling backtester re-estimates the model every few days, **re-filters the
conditional variance every day**, and scores 1-day VaR with the Kupiec POF and
Christoffersen coverage tests. Under symmetric Student's-t innovations, all
three assets **under-cover** the downside (≈6–7% violations vs the 5% target).
Tracing this to unmodeled **return skew**, switching to a skewed-t restores
correct coverage on equities and crude.

### 3\. Why does the REIT still fail?

Prologis passes the violation-*rate* test under skew-t but still fails the
Christoffersen **independence** test — its violations *cluster*. A breach
roughly doubles the probability of a breach the next day (5.4% → 10.7%). This
is a *dynamics* problem, not a distribution problem, and it is orthogonal to
the skew issue.

A **Markov-switching GJR-GARCH**, implemented from scratch
(Haas–Mittnik–Paolella parallel-regime formulation, Hamilton-filter likelihood,
numba-accelerated), restores the independence test (p ≈ 0.00 → 0.73) by letting
volatility switch between a calm and a crisis regime — where the crisis regime
activates in exactly the historical stress periods (2009, 2020, 2022).

**The headline finding is the decomposition:** the two VaR failures have two
orthogonal causes with two independent fixes — return skew drives the violation
*rate*, single-regime dynamics drive the *clustering*. Notably, the
Markov-switching model does **not** improve in-sample BIC for any asset; its
value is in tail-risk calibration, which BIC cannot see.

### 4\. Does the functional form of the asymmetry matter for forecasting?

An out-of-sample horse race scores GJR-GARCH against EGARCH on 1-day variance
forecasts using **QLIKE** loss (robust to noise in the squared-return proxy)
and the **Diebold–Mariano** test (Harvey–Leybourne–Newbold corrected).
GJR-GARCH produces more robust forecasts across all three assets, but the
margin is **driven by EGARCH's exponential over-reaction to extreme shocks**
(most severe on crude); on a typical day the two are close (GJR daily win-rates
54–58%).

\---

## Repository layout

|File|Role|
|-|-|
|`data\_pipeline.py`|Config-driven loading, cleaning, log returns, validation, caching|
|`gjr\_garch.py`|GJR-GARCH candidate fitting, BIC selection, Ljung-Box, parameter extraction|
|`var\_backtest.py`|Rolling VaR backtester; Kupiec + Christoffersen; symmetric-t vs skew-t|
|`ms\_garch.py`|Markov-switching GJR-GARCH from scratch (Hamilton filter, numba)|
|`ms\_var\_backtest.py`|MS-GARCH VaR via the regime-mixture predictive quantile|
|`egarch\_horserace.py`|GJR vs EGARCH out-of-sample contest (QLIKE + Diebold–Mariano)|
|`run\_all.py`|Staged entry point tying the whole pipeline together|
|`main.ipynb`|Notebook walkthrough of the same analysis|

Every module has a self-test under `if \_\_name\_\_ == "\_\_main\_\_":` — most validate
against simulated data with known properties (e.g. the MS-GARCH recovers known
parameters; the VaR backtester passes coverage on a correctly-specified
simulation).

\---

## Running it

```bash
pip install -r requirements.txt          # see note on numba below
python run\_all.py                        # fast: data quality + GJR-GARCH fits
python run\_all.py --stages var           # add the VaR skew-decomposition
python run\_all.py --all                  # everything (\~30-45 min; rolling backtests)
```

S\&P 500 and Prologis are pulled from Yahoo Finance (needs internet); WTI is read
from the bundled `RWTCd.csv`. Raw pulls are cached under `data\_cache/` so re-runs
are reproducible and don't depend on when you ran them.

`numba` JIT-compiles the Markov-switching hot loops (\~100× speedup); it is
required for the `msgarch` stage to run in reasonable time.

\---

## Data sources

* **S\&P 500 (^GSPC)** and **Prologis (PLD)** — Yahoo Finance, adjusted close.
* **WTI crude** — U.S. Energy Information Administration, Cushing WTI spot price
(series RWTC). EIA data is U.S. federal government work in the public domain;
the file is bundled for reproducibility.

\---

## Methods \& references

* Glosten, Jagannathan \& Runkle (1993) — GJR-GARCH (asymmetric volatility).
* Haas, Mittnik \& Paolella (2004) — the Markov-switching GARCH formulation that
avoids path dependence (the basis for `ms\_garch.py`).
* Kupiec (1995); Christoffersen (1998) — VaR coverage tests.
* Patton (2011) — robust volatility-forecast loss functions (QLIKE).
* Diebold \& Mariano (1995); Harvey, Leybourne \& Newbold (1997) — forecast
accuracy comparison and small-sample correction.

\---

## License

MIT — see `LICENSE`.

