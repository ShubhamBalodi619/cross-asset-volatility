"""
Out-of-sample VaR backtest for the Markov-switching GARCH.

The one-day-ahead predictive distribution of an MS-GARCH is a MIXTURE over
regimes: weight each regime's Student-t (mean mu, variance h_{k,t+1}) by the
one-step predicted regime probability, then read the alpha-quantile off the
mixture. A mixture has no closed-form quantile, so we invert its CDF numerically
(Brent). This is the correct MS-GARCH VaR -- not "VaR of the most-likely regime."

Methodology mirrors the single-regime backtester: parameters re-estimated every
`refit_every` days; regime probabilities and variances re-filtered every day.
The whole point is to compare, on the SAME harness, whether regime-switching
fixes the violation clustering that skew-t single-regime could not.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats, optimize

from ms_garch import (fit_msgarch, _variance_paths, _std_t_logpdf,
                      _hamilton_filter, _trans_matrix, _stationary_dist, SCALE)
from var_backtest import kupiec_pof, christoffersen


def _mixture_cdf(x, probs, mu, sigmas, nu):
    s = np.sqrt((nu - 2.0) / nu)
    z = (x - mu) / (sigmas * s)
    cdf = stats.t.cdf(z, df=nu)
    return float(np.sum(probs * cdf))


def _mixture_var(probs, mu, sigmas, nu, alpha):
    """alpha-quantile of the regime mixture (left tail). Units: percent.

    Bracket by the component VaRs: each regime's own t-VaR is
    mu + sigma_k * q (q<0); the mixture quantile must lie between the most and
    least extreme of these, which gives a guaranteed-valid, tight bracket and
    avoids degenerate wide brackets when a regime variance is large.
    """
    s = np.sqrt((nu - 2.0) / nu)
    q = stats.t.ppf(alpha, df=nu) * s
    comp_var = mu + sigmas * q            # per-regime VaR (more negative = larger sigma)
    lo = comp_var.min() - 1e-6
    hi = comp_var.max() + 1e-6
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        # fall back to the probability-weighted component VaR
        return float(np.sum(probs * comp_var))
    return optimize.brentq(
        lambda x: _mixture_cdf(x, probs, mu, sigmas, nu) - alpha, lo, hi,
        xtol=1e-8, maxiter=200)


def _one_step_var(window_pct, mu, regs, P, nu, alpha, backcast):
    """1-day-ahead mixture VaR for the day after `window_pct` (percent units)."""
    eps = window_pct - mu
    h = _variance_paths(eps, regs, backcast)            # (T,K), capped in jit
    z = eps[:, None] / np.sqrt(h)
    loglik_kt = _std_t_logpdf(z, nu) - 0.5 * np.log(h)
    loglik_kt = np.nan_to_num(loglik_kt, nan=-1e6, posinf=-1e6, neginf=-1e6)
    _, filt, _ = _hamilton_filter(loglik_kt, P, _stationary_dist(P))

    neg = 1.0 if eps[-1] < 0 else 0.0
    e2 = eps[-1] ** 2
    h_next = np.array([om + (al + ga * neg) * e2 + be * h[-1, k]
                       for k, (om, al, ga, be) in enumerate(regs)])
    h_next = np.clip(np.nan_to_num(h_next, nan=backcast), 1e-8, 1e8)
    pred_next = filt[-1] @ P                            # one-step regime probs
    pred_next = np.clip(pred_next, 0.0, 1.0)
    pred_next = pred_next / pred_next.sum()
    sigmas = np.sqrt(h_next)
    return _mixture_var(pred_next, mu, sigmas, nu, alpha)


def rolling_msgarch_var(returns, window=504, alpha=0.05, refit_every=21,
                        n_starts=3, seed=0):
    """
    Rolling MS-GARCH VaR. Parameters re-estimated every `refit_every` days
    (MS-GARCH fits are expensive, so monthly by default); regime probabilities
    and variances re-filtered every day. Returns return/var/violation frame.
    """
    r = returns.dropna() * SCALE
    n = len(r)
    var_pred = np.full(n, np.nan)
    mu = regs = P = nu = None

    for t in range(window, n):
        win = r.iloc[t - window:t].values
        if mu is None or (t - window) % refit_every == 0:
            res = fit_msgarch(r.iloc[t - window:t] / SCALE,
                              n_starts=n_starts, seed=seed)
            mu, regs, P, nu = res.mu, res.regimes, res.P, res.nu
        bc = np.var(win)
        var_pred[t] = _one_step_var(win, mu, regs, P, nu, alpha, bc)

    out = pd.DataFrame({"return": r.values, "var": var_pred},
                       index=r.index).iloc[window:]
    out["violation"] = (out["return"] < out["var"]).astype(int)
    out[["return", "var"]] /= SCALE
    return out


def msgarch_backtest_report(returns, window=504, alpha=0.05, refit_every=21,
                            label=""):
    bt = rolling_msgarch_var(returns, window, alpha, refit_every)
    pof = kupiec_pof(bt["violation"], alpha)
    chr_ = christoffersen(bt["violation"], alpha)
    print(f"\n=== MS-GARCH VaR backtest: {label} (95% 1-day) ===")
    print(f"Test days     : {pof['n']}")
    print(f"Violations    : {pof['violations']}  rate={pof['rate']:.4f} (exp {alpha})")
    print(f"Kupiec POF    : p={pof['p_value']:.3f} "
          f"{'PASS' if pof['p_value']>0.05 else 'FAIL'}")
    print(f"Independence  : p={chr_['p_ind']:.3f} "
          f"{'PASS' if chr_['p_ind']>0.05 else 'FAIL'}")
    print(f"Cond. coverage: p={chr_['p_cc']:.3f} "
          f"{'PASS' if chr_['p_cc']>0.05 else 'FAIL'}")
    return bt, pof, chr_
