"""
VaR backtesting for a GJR-GARCH(1,1,1) model with Student's-t innovations.

Workflow
--------
1. Roll a fixed estimation window through the series.
2. At each step, refit the model and forecast 1-day-ahead conditional variance.
3. Convert that variance into a 1-day VaR using the Student's-t quantile.
4. Flag a violation when the realized return breaches the VaR.
5. Test the violation sequence with Kupiec (POF) and Christoffersen (independence
   and conditional coverage).

Backtest 1-day VaR for clean, independent violations. Report the 10-day VaR/ES
separately as the headline risk number -- do not backtest the overlapping 10-day
series (violations are autocorrelated by construction and break the independence test).
"""

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model


# ----------------------------------------------------------------------
# 1. Rolling VaR forecast
# ----------------------------------------------------------------------
def rolling_var_forecast(returns, window=504, alpha=0.05, refit_every=5,
                         dist="skewt", scale=100.0):
    """
    Produce a 1-day-ahead VaR series via rolling GJR-GARCH.

    Methodology: PARAMETERS re-estimated every `refit_every` days; conditional
    VARIANCE re-filtered EVERY day with parameters fixed (skipping the daily
    refilter freezes the VaR between refits and manufactures clustered
    violations).

    dist : "t"     symmetric Student's-t innovations
           "skewt" Hansen skewed-t -- captures return asymmetry, which the
                   symmetric t cannot. Under-coverage of the downside VaR on
                   skewed assets is the symptom that motivates skewt.

    The left-tail quantile is taken from the FITTED standardized distribution,
    so for skewt the asymmetry flows straight into the VaR. It is cached at each
    refit (constant while parameters are held fixed).
    """
    r = returns.dropna() * scale
    n = len(r)
    if n <= window:
        raise ValueError("Series shorter than the estimation window.")

    var_pred = np.full(n, np.nan)
    params = None
    last_q = None
    mu = None

    for t in range(window, n):
        train = r.iloc[t - window:t]
        am = arch_model(train, mean="Constant", vol="GARCH",
                        p=1, o=1, q=1, dist=dist)

        if params is None or (t - window) % refit_every == 0:
            res = am.fit(disp="off")
            params = res.params
            mu = params["mu"]
            last_q = _standardized_quantile(res, dist, alpha)
            fc = res.forecast(horizon=1, reindex=False)
        else:
            fixed = am.fix(params)
            fc = fixed.forecast(horizon=1, reindex=False)

        sigma = np.sqrt(fc.variance.values[-1, 0])
        var_pred[t] = mu + sigma * last_q

    out = pd.DataFrame(
        {"return": r.values, "var": var_pred}, index=r.index
    ).iloc[window:]
    out["violation"] = (out["return"] < out["var"]).astype(int)
    out[["return", "var"]] /= scale
    return out


def _standardized_quantile(res, dist, alpha):
    """Left-tail quantile of the fitted UNIT-VARIANCE innovation distribution."""
    p = res.params
    if dist == "t":
        nu = p["nu"]
        # raw t has variance nu/(nu-2); rescale to unit variance
        return stats.t.ppf(alpha, df=nu) * np.sqrt((nu - 2.0) / nu)
    elif dist == "skewt":
        # arch's skew-t ppf already returns a standardized (unit-variance) quantile
        return float(res.model.distribution.ppf(alpha, [p["eta"], p["lambda"]]))
    else:
        raise ValueError(f"unsupported dist {dist!r}")


# ----------------------------------------------------------------------
# 2. Kupiec POF test (unconditional coverage)
# ----------------------------------------------------------------------
def kupiec_pof(violations, alpha=0.05):
    """LR test that the violation RATE equals alpha. chi-square, 1 df."""
    v = np.asarray(violations)
    n = len(v)
    x = int(v.sum())
    pi_hat = x / n if n else 0.0

    # guard the log-likelihood against log(0)
    if x == 0:
        ll_null = n * np.log(1 - alpha)
        ll_alt = 0.0
    elif x == n:
        ll_null = n * np.log(alpha)
        ll_alt = 0.0
    else:
        ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll_alt = (n - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)

    lr = -2.0 * (ll_null - ll_alt)
    p = 1 - stats.chi2.cdf(lr, df=1)
    return {"n": n, "violations": x, "rate": pi_hat,
            "expected_rate": alpha, "LR_pof": lr, "p_value": p}


# ----------------------------------------------------------------------
# 3. Christoffersen independence + conditional coverage
# ----------------------------------------------------------------------
def christoffersen(violations, alpha=0.05):
    """
    Independence test (violations should not cluster) and the combined
    conditional-coverage test (= Kupiec + independence).
    """
    v = np.asarray(violations)

    # transition counts: n_ij = moves from state i to state j
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(v[:-1], v[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(p): return np.log(p) if p > 0 else 0.0

    ll_ind = (n00 * safe_log(1 - pi01) + n01 * safe_log(pi01) +
              n10 * safe_log(1 - pi11) + n11 * safe_log(pi11))
    ll_pooled = ((n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi))

    lr_ind = -2.0 * (ll_pooled - ll_ind)
    p_ind = 1 - stats.chi2.cdf(lr_ind, df=1)

    # conditional coverage = unconditional (Kupiec) + independence
    lr_uc = kupiec_pof(v, alpha)["LR_pof"]
    lr_cc = lr_uc + lr_ind
    p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)

    return {"LR_ind": lr_ind, "p_ind": p_ind,
            "LR_cc": lr_cc, "p_cc": p_cc,
            "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11}}


# ----------------------------------------------------------------------
# 4. One-call report
# ----------------------------------------------------------------------
def backtest_report(returns, window=504, alpha=0.05, refit_every=5,
                    dist="skewt", label=""):
    bt = rolling_var_forecast(returns, window, alpha, refit_every, dist)
    pof = kupiec_pof(bt["violation"], alpha)
    chr_ = christoffersen(bt["violation"], alpha)

    print(f"\n=== VaR backtest: {label} (95% 1-day VaR, dist={dist}) ===")
    print(f"Test days            : {pof['n']}")
    print(f"Violations           : {pof['violations']}")
    print(f"Observed rate        : {pof['rate']:.4f}  (expected {alpha:.2f})")
    print(f"Kupiec POF  LR={pof['LR_pof']:.3f}  p={pof['p_value']:.3f}  "
          f"-> {'PASS' if pof['p_value'] > 0.05 else 'FAIL'}")
    print(f"Independence LR={chr_['LR_ind']:.3f}  p={chr_['p_ind']:.3f}  "
          f"-> {'PASS' if chr_['p_ind'] > 0.05 else 'FAIL'}")
    print(f"Cond. coverage LR={chr_['LR_cc']:.3f}  p={chr_['p_cc']:.3f}  "
          f"-> {'PASS' if chr_['p_cc'] > 0.05 else 'FAIL'}")
    return bt, pof, chr_


def compare_distributions(returns, window=504, alpha=0.05, refit_every=5,
                          label=""):
    """
    Backtest the same asset under symmetric-t vs skew-t innovations.
    The contrast (symmetric under-covers, skew-t restores coverage) is the
    headline finding: it shows the coverage failure is about return ASYMMETRY,
    not a coding artifact.
    """
    rows = []
    for d in ("t", "skewt"):
        bt = rolling_var_forecast(returns, window, alpha, refit_every, d)
        pof = kupiec_pof(bt["violation"], alpha)
        chr_ = christoffersen(bt["violation"], alpha)
        rows.append({
            "asset": label, "dist": d,
            "violations": pof["violations"], "rate": round(pof["rate"], 4),
            "kupiec_p": round(pof["p_value"], 3),
            "indep_p": round(chr_["p_ind"], 3),
            "cc_p": round(chr_["p_cc"], 3),
            "kupiec": "PASS" if pof["p_value"] > 0.05 else "FAIL",
            "cc": "PASS" if chr_["p_cc"] > 0.05 else "FAIL",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Self-test on simulated data
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate a GJR-GARCH(1,1,1)-t process so we KNOW the model is correct;
    # a correctly specified backtest should then PASS the coverage tests.
    rng = np.random.default_rng(7)
    n = 2000
    omega, a, gamma, beta, nu = 0.02, 0.03, 0.15, 0.88, 6.0  # persistence 0.985 < 1
    eps = np.zeros(n); sig2 = np.zeros(n); sig2[0] = omega / (1 - a - beta - gamma/2)
    z = stats.t.rvs(df=nu, size=n, random_state=rng) * np.sqrt((nu - 2) / nu)
    for t in range(1, n):
        shock = eps[t-1]
        sig2[t] = (omega + (a + gamma * (shock < 0)) * shock**2
                   + beta * sig2[t-1])
        eps[t] = np.sqrt(sig2[t]) * z[t]
    dates = pd.bdate_range("2016-01-01", periods=n)
    sim_returns = pd.Series(eps / 100.0, index=dates)  # decimal returns

    backtest_report(sim_returns, window=504, alpha=0.05,
                    refit_every=10, label="Simulated GJR-t")
