"""
Out-of-sample variance-forecast horse race: GJR-GARCH vs EGARCH.

Both are asymmetric volatility models, but they encode the leverage effect
differently:
    GJR    : h_t = omega + (alpha + gamma*1[eps<0]) eps_{t-1}^2 + beta h_{t-1}
             -- asymmetry as an extra ARCH term on negative shocks; needs
             positivity constraints on the parameters.
    EGARCH : log h_t = omega + alpha(|z|-E|z|) + gamma z_{t-1} + beta log h_{t-1}
             -- models LOG-variance, so positivity is automatic and the
             asymmetry enters multiplicatively. Different functional form,
             so the comparison is informative rather than two relabelings.

We compare them the way a desk would: OUT-OF-SAMPLE one-day-ahead variance
forecasts, scored by QLIKE loss against a realized proxy, with a
Diebold-Mariano test for whether any difference is statistically real.

Why QLIKE and not MSE:
    The realized proxy (squared return) is a noisy but conditionally-unbiased
    estimate of the true variance. QLIKE,
        L(sigma2_proxy, h) = sigma2_proxy/h - log(sigma2_proxy/h) - 1,
    is "robust" in Patton's (2011) sense: its expected minimizer is the true
    conditional variance even though the proxy is noisy. MSE on squared returns
    is distorted by that noise and can rank models incorrectly. QLIKE is also
    scale-free (depends only on the ratio sigma2_proxy/h), so units cancel.

Methodology mirrors the VaR backtester: parameters re-estimated every
`refit_every` days; the conditional variance is re-filtered EVERY day with
parameters held fixed.

Note: EGARCH's SLSQP optimizer occasionally fails to converge on windows with
extreme moves (ConvergenceWarning, code 4). This is handled deliberately --
non-converged refits are discarded and the last good parameters retained -- and
the warning is suppressed below so it doesn't clutter output. The n_nonconv
counter tracks how often it happens.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

SCALE = 100.0


# ----------------------------------------------------------------------
# Rolling 1-day-ahead variance forecasts for one volatility spec
# ----------------------------------------------------------------------
def rolling_variance_forecast(returns, vol="GARCH", p=1, o=1, q=1, dist="t",
                              window=504, refit_every=5, scale=SCALE,
                              h_floor=1e-2, h_ceil=1e4):
    """
    Returns a DataFrame with the realized proxy and the 1-day-ahead variance
    forecast, aligned on the forecast target date.

      vol="GARCH", o=1 -> GJR-GARCH(1,1,1)
      vol="EGARCH", o=1 -> EGARCH(1,1,1)

    proxy_t = (r_t - mu_hat)^2  : squared 1-step forecast error, the standard
    realized-variance proxy for daily data.

    Robustness: EGARCH estimation is less stable than GJR. If a refit fails to
    converge, we KEEP the previous (good) parameters rather than forecasting
    from garbage. Forecasts are clipped to a sane variance band; the count of
    non-converged refits and clipped forecasts is returned for transparency.
    """
    r = returns.dropna() * scale
    n = len(r)
    if n <= window:
        raise ValueError("Series shorter than the estimation window.")

    h_fore = np.full(n, np.nan)
    mu_fore = np.full(n, np.nan)
    params = None
    mu = None
    n_nonconv = 0
    n_clipped = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("ignore", ConvergenceWarning)
        for t in range(window, n):
            train = r.iloc[t - window:t]
            am = arch_model(train, mean="Constant", vol=vol,
                            p=p, o=o, q=q, dist=dist, rescale=False)
            if params is None or (t - window) % refit_every == 0:
                res = am.fit(disp="off", options={"maxiter": 2000})
                if int(getattr(res, "convergence_flag", 0)) == 0 or params is None:
                    params = res.params
                    mu = params["mu"]
                else:
                    n_nonconv += 1            # keep previous good params
            fc = am.fix(params).forecast(horizon=1, reindex=False)
            h = fc.variance.values[-1, 0]
            if not np.isfinite(h) or h < h_floor or h > h_ceil:
                n_clipped += 1
                h = min(max(h if np.isfinite(h) else h_floor, h_floor), h_ceil)
            h_fore[t] = h
            mu_fore[t] = mu

    out = pd.DataFrame({"r": r.values, "mu": mu_fore, "h": h_fore},
                       index=r.index).iloc[window:]
    out["proxy"] = (out["r"] - out["mu"]) ** 2
    out.attrs["n_nonconv"] = n_nonconv
    out.attrs["n_clipped"] = n_clipped
    return out[["proxy", "h"]]


# ----------------------------------------------------------------------
# QLIKE loss
# ----------------------------------------------------------------------
def qlike(proxy, h):
    """Patton (2011) robust QLIKE loss, elementwise. Lower is better."""
    ratio = proxy / h
    return ratio - np.log(ratio) - 1.0


# ----------------------------------------------------------------------
# Diebold-Mariano test
# ----------------------------------------------------------------------
def diebold_mariano(loss_a, loss_b, lag=1, hln=True):
    """
    Test equal predictive accuracy of model A vs model B.

    d_t = loss_a - loss_b ;  H0: E[d]=0.
    Variance of d_bar uses a Newey-West HAC estimator (loss differentials can be
    autocorrelated). `hln` applies the Harvey-Leybourne-Newbold small-sample
    correction and uses a t-distribution.

    Returns dict with the statistic, p-value, mean differential, and the verdict.
    A NEGATIVE mean differential => model A has the lower loss (A wins).
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    d_bar = d.mean()

    # Newey-West long-run variance of the mean
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)                 # Bartlett weight
        cov = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv += 2.0 * w * cov
    var_dbar = lrv / n

    dm = d_bar / np.sqrt(var_dbar)

    if hln:
        # Harvey-Leybourne-Newbold correction (h=1 step here)
        h_steps = 1
        corr = np.sqrt((n + 1 - 2*h_steps + h_steps*(h_steps-1)/n) / n)
        dm *= corr
        p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    else:
        p = 2 * (1 - stats.norm.cdf(abs(dm)))

    return {"dm_stat": dm, "p_value": p, "mean_diff": d_bar, "n": n}


# ----------------------------------------------------------------------
# Full horse race for one asset
# ----------------------------------------------------------------------
def horse_race(returns, label="", window=504, refit_every=5, dist="t",
               lag=1):
    gjr = rolling_variance_forecast(returns, vol="GARCH", o=1, dist=dist,
                                    window=window, refit_every=refit_every)
    egarch = rolling_variance_forecast(returns, vol="EGARCH", o=1, dist=dist,
                                       window=window, refit_every=refit_every)

    idx = gjr.index.intersection(egarch.index)
    proxy = gjr.loc[idx, "proxy"].values
    l_gjr = qlike(proxy, gjr.loc[idx, "h"].values)
    l_eg = qlike(proxy, egarch.loc[idx, "h"].values)

    dm = diebold_mariano(l_gjr, l_eg, lag=lag)
    mean_gjr, mean_eg = np.nanmean(l_gjr), np.nanmean(l_eg)
    # robustness: is any gap broad-based or driven by a few blow-up days?
    diff = l_gjr - l_eg
    med_diff = np.nanmedian(diff)
    gjr_winrate = np.mean(diff < 0)            # fraction of days GJR has lower loss
    if dm["p_value"] > 0.05:
        verdict = "no significant difference"
    else:
        verdict = "GJR better" if dm["mean_diff"] < 0 else "EGARCH better"

    print(f"\n=== Horse race: {label} (out-of-sample, {len(idx)} days) ===")
    print(f"  mean QLIKE   GJR={mean_gjr:.4f}   EGARCH={mean_eg:.4f}")
    print(f"  median diff (GJR-EGARCH) = {med_diff:+.4f}   "
          f"GJR daily win-rate = {gjr_winrate:.1%}")
    print(f"  EGARCH refits non-converged={egarch.attrs['n_nonconv']}, "
          f"forecasts clipped={egarch.attrs['n_clipped']}  "
          f"(GJR clipped={gjr.attrs['n_clipped']})")
    print(f"  Diebold-Mariano (mean loss) stat={dm['dm_stat']:.3f}  "
          f"p={dm['p_value']:.3f}")
    print(f"  verdict (mean-loss DM): {verdict}")
    if (verdict != "no significant difference" and
            np.sign(med_diff) != np.sign(dm["mean_diff"])):
        print(f"  [caution] median and mean disagree in sign -> the mean-loss "
              f"verdict is driven by a few extreme days, not the typical day.")
    return {"label": label, "qlike_gjr": mean_gjr, "qlike_egarch": mean_eg,
            "median_diff": med_diff, "gjr_winrate": gjr_winrate,
            "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"], "verdict": verdict}


def horse_race_table(results):
    return pd.DataFrame(results)[
        ["label", "qlike_gjr", "qlike_egarch", "dm_stat", "dm_p", "verdict"]
    ].round(4)


# ----------------------------------------------------------------------
# Self-test on real WTI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from data_pipeline import load_asset, log_returns
    wti = log_returns(load_asset("WTI", use_cache=False))
    horse_race(wti, label="WTI Crude", refit_every=21)  # 21 for a quick self-test