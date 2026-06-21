"""
Markov-switching GJR-GARCH, Haas-Mittnik-Paolella (2004) formulation.

Why this formulation (the interview question):
    A naive MS-GARCH is PATH-DEPENDENT: h_t depends on the entire unobserved
    regime history, so the likelihood needs a sum over K^T regime paths --
    intractable. Haas et al. instead run K INDEPENDENT GARCH recursions in
    parallel, each driven by the common shock but evolving its own variance:

        h_{k,t} = omega_k + (alpha_k + gamma_k * 1[eps_{t-1}<0]) eps_{t-1}^2
                  + beta_k * h_{k,t-1}

    Each h_{k,t} is a deterministic function of the return history alone -- no
    regime path -- so all K variance paths are precomputable and the Hamilton
    filter handles the rest in O(T*K^2). That is what makes estimation feasible.

Model:
    r_t = mu + eps_t,   eps_t = sqrt(h_{s_t, t}) * z_t,   z_t ~ standardized t(nu)
    s_t in {1..K} follows a Markov chain with transition matrix P.

Estimation: maximize the Hamilton-filter log-likelihood. The one-step density
is a regime mixture, which is itself fat-tailed and asymmetric even with a
single nu -- the mixing does work a single-regime model cannot.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln
from numba import njit

SCALE = 100.0


# ----------------------------------------------------------------------
# JIT-compiled hot loops
# ----------------------------------------------------------------------
@njit(cache=True)
def _variance_paths_jit(eps, om, al, ga, be, backcast):
    """Parallel GJR-GARCH recursions, one per regime. Returns h (T,K)."""
    T = eps.shape[0]
    K = om.shape[0]
    HUGE = 1.0e8                              # cap: stops near-integrated regimes
    h = np.empty((T, K))                      # from overflowing to inf over a window
    for k in range(K):
        h[0, k] = backcast
    for t in range(1, T):
        e2 = eps[t-1] * eps[t-1]
        neg = 1.0 if eps[t-1] < 0 else 0.0
        for k in range(K):
            v = om[k] + (al[k] + ga[k]*neg)*e2 + be[k]*h[t-1, k]
            if v > HUGE:
                v = HUGE
            elif v < 1e-12:
                v = 1e-12
            h[t, k] = v
    return h


@njit(cache=True)
def _hamilton_jit(loglik_kt, P, pi0):
    """Hamilton filter. Returns total loglik, filtered (T,K), predicted (T,K)."""
    T, K = loglik_kt.shape
    filt = np.empty((T, K))
    pred = np.empty((T, K))
    ll = 0.0
    for t in range(T):
        # predicted regime probs
        if t == 0:
            for k in range(K):
                pred[t, k] = pi0[k]
        else:
            for j in range(K):
                acc = 0.0
                for i in range(K):
                    acc += filt[t-1, i] * P[i, j]
                pred[t, j] = acc
        m = loglik_kt[t, 0]
        for k in range(1, K):
            if loglik_kt[t, k] > m:
                m = loglik_kt[t, k]
        s = 0.0
        for k in range(K):
            filt[t, k] = pred[t, k] * np.exp(loglik_kt[t, k] - m)
            s += filt[t, k]
        ll += np.log(s) + m
        for k in range(K):
            filt[t, k] /= s
    return ll, filt, pred


from scipy.special import gammaln as _gammaln

def _std_t_logpdf(z, nu):
    """log density of a unit-variance standardized Student-t (analytic, fast)."""
    s = np.sqrt((nu - 2.0) / nu)             # T = Z/s ~ standard t_nu
    u = z / s
    c = _gammaln((nu + 1.0) / 2.0) - _gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
    return c - 0.5 * (nu + 1.0) * np.log1p(u * u / nu) - np.log(s)


def _reg_arrays(reg_params):
    rp = np.asarray(reg_params, dtype=float)
    return rp[:, 0].copy(), rp[:, 1].copy(), rp[:, 2].copy(), rp[:, 3].copy()


def _variance_paths(eps, reg_params, backcast):
    om, al, ga, be = _reg_arrays(reg_params)
    return _variance_paths_jit(np.ascontiguousarray(eps), om, al, ga, be,
                               float(backcast))


def _hamilton_filter(loglik_kt, P, pi0):
    return _hamilton_jit(np.ascontiguousarray(loglik_kt),
                         np.ascontiguousarray(P),
                         np.ascontiguousarray(pi0))


# ----------------------------------------------------------------------
# Parameter packing  (unconstrained optimizer space <-> model space)
# ----------------------------------------------------------------------
# theta layout for K=2, t-dist:
#   [mu, lo1, a1, g1, b1, lo2, a2, g2, b2, q11, q22, lnu]
# omega = exp(lo); diagonal transition probs via logistic(q); nu = 2+exp(lnu)
def _unpack(theta, K=2):
    mu = theta[0]
    regs = []
    i = 1
    for _ in range(K):
        om = np.exp(theta[i]); al = theta[i+1]; ga = theta[i+2]; be = theta[i+3]
        regs.append((om, al, ga, be)); i += 4
    p_diag = 1.0 / (1.0 + np.exp(-theta[i:i+K])); i += K
    nu = 2.0 + np.exp(theta[i])
    return mu, regs, p_diag, nu


def _trans_matrix(p_diag):
    K = len(p_diag)
    P = np.empty((K, K))
    for i in range(K):
        P[i, i] = p_diag[i]
        off = (1.0 - p_diag[i]) / (K - 1)
        for j in range(K):
            if j != i:
                P[i, j] = off
    return P


def _stationary_dist(P):
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


def _neg_loglik(theta, eps, backcast, K=2):
    mu, regs, p_diag, nu = _unpack(theta, K)
    # guard rails -> penalty (keeps optimizer in a sane region)
    for (om, al, ga, be) in regs:
        if om <= 0 or al < 0 or be < 0 or be >= 1 or (al + ga/2) < 0:
            return 1e10
        if al + be + ga/2 >= 1.0:        # per-regime stationarity
            return 1e10
    e = eps - mu
    h = _variance_paths(e, regs, backcast)
    if not np.all(np.isfinite(h)) or np.any(h <= 0):
        return 1e10
    z = e[:, None] / np.sqrt(h)
    loglik_kt = _std_t_logpdf(z, nu) - 0.5 * np.log(h)  # includes Jacobian 1/sqrt(h)
    P = _trans_matrix(p_diag)
    pi0 = _stationary_dist(P)
    ll, _, _ = _hamilton_filter(loglik_kt, P, pi0)
    return -ll if np.isfinite(ll) else 1e10


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------
@dataclass
class MSGARCHResult:
    mu: float
    regimes: list           # [(omega, alpha, gamma, beta), ...] sorted low->high vol
    P: np.ndarray
    nu: float
    loglik: float
    n_params: int
    nobs: int
    filtered: np.ndarray    # (T, K)
    predicted: np.ndarray
    h: np.ndarray           # (T, K) regime variances (percent^2)
    uncond_vol: np.ndarray  # per-regime annualized vol (%)

    @property
    def aic(self): return -2*self.loglik + 2*self.n_params
    @property
    def bic(self): return -2*self.loglik + self.n_params*np.log(self.nobs)


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------
def fit_msgarch(returns, K=2, n_starts=8, seed=0):
    r = returns.dropna().values * SCALE
    eps = r
    backcast = np.var(r)
    var = np.var(r)

    # data-driven regime scales: low/high percentiles of rolling variance.
    # Initializing the crisis regime from the data's actual high-vol periods
    # steers the optimizer away from the degenerate 1-day "spike catcher"
    # local optimum.
    rv = np.convolve((r - r.mean())**2, np.ones(22)/22, mode="same")
    calm_var = np.percentile(rv, 35)
    crisis_var = np.percentile(rv, 90)

    rng = np.random.default_rng(seed)
    best = None

    for s in range(n_starts):
        j = rng.normal(0, 0.15, size=4) if s else np.zeros(4)
        # calm: high persistence; crisis: high omega, moderate persistence
        theta0 = [
            np.mean(r),
            np.log(max(calm_var * 0.03, 1e-3)) + j[0], 0.03, 0.05, 0.90,
            np.log(max(crisis_var * 0.10, 1e-2)) + j[1], 0.06, 0.12, 0.80,
            2.9 + j[2], 2.2 + j[3],                  # logistic -> P11~.95, P22~.90
            np.log(6.0),
        ]
        try:
            res = optimize.minimize(
                _neg_loglik, theta0, args=(eps, backcast, K),
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-7},
            )
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res

    mu, regs, p_diag, nu = _unpack(best.x, K)
    P = _trans_matrix(p_diag)
    e = eps - mu
    h = _variance_paths(e, regs, backcast)
    z = e[:, None] / np.sqrt(h)
    loglik_kt = _std_t_logpdf(z, nu) - 0.5*np.log(h)
    ll, filt, pred = _hamilton_filter(loglik_kt, P, _stationary_dist(P))

    # robust per-regime vol: filtered-prob-weighted mean of h (always finite,
    # unlike omega/(1-persistence) which diverges for near-unit-persistence
    # crisis regimes). Annualized, in percent. Then sort calm -> crisis.
    wsum = filt.sum(axis=0)
    wmean_var = np.divide((filt * h).sum(axis=0), wsum,
                          out=np.full(filt.shape[1], np.nan), where=wsum > 0)
    order = np.argsort(wmean_var)
    regs = [regs[i] for i in order]
    wmean_var = wmean_var[order]
    P = P[np.ix_(order, order)]
    filt = filt[:, order]; pred = pred[:, order]; h = h[:, order]
    ann_vol = np.sqrt(wmean_var * 252)       # percent

    n_params = 1 + 4*K + K + 1               # mu + GARCH + diag(P) + nu
    return MSGARCHResult(mu, regs, P, nu, ll, n_params, len(r),
                         filt, pred, h, ann_vol)


def print_msgarch(res: MSGARCHResult, label=""):
    print(f"\n=== MS-GARCH(2-regime) GJR-t : {label} ===")
    for k, (om, al, ga, be) in enumerate(res.regimes):
        tag = "calm " if k == 0 else "crisis"
        pers = al + be + ga/2
        flag = "" if pers < 1 else "  (locally non-stationary; transient)"
        print(f"  regime {k} [{tag}]: omega={om:.4f} alpha={al:.4f} "
              f"gamma={ga:.4f} beta={be:.4f}")
        print(f"      persistence={pers:.4f}{flag}  typical ann.vol={res.uncond_vol[k]:.1f}%")
    print(f"  transition P:  stay calm P11={res.P[0,0]:.3f}   "
          f"stay crisis P22={res.P[1,1]:.3f}")
    exp_dur = 1.0/(1.0 - np.diag(res.P))
    print(f"  expected regime duration (days): calm={exp_dur[0]:.0f}, "
          f"crisis={exp_dur[1]:.0f}")
    print(f"  nu={res.nu:.2f}   loglik={res.loglik:.1f}   "
          f"AIC={res.aic:.1f}   BIC={res.bic:.1f}")


# ----------------------------------------------------------------------
# Self-test: parameter recovery on simulated 2-regime MS-GARCH
# ----------------------------------------------------------------------
def _simulate(n, mu, regs, P, nu, seed=0):
    rng = np.random.default_rng(seed)
    K = len(regs)
    s = np.zeros(n, dtype=int)
    h = np.array([np.var([1.0])] * K, dtype=float)
    h = np.array([regs[k][0]/(1-regs[k][1]-regs[k][3]-regs[k][2]/2) for k in range(K)])
    r = np.zeros(n); eps_prev = 0.0; hk = h.copy()
    pi = _stationary_dist(P); s[0] = rng.choice(K, p=pi)
    z = stats.t.rvs(df=nu, size=n, random_state=rng) * np.sqrt((nu-2)/nu)
    for t in range(1, n):
        s[t] = rng.choice(K, p=P[s[t-1]])
        for k in range(K):
            om, al, ga, be = regs[k]
            hk[k] = om + (al + ga*(eps_prev < 0))*eps_prev**2 + be*hk[k]
        eps = np.sqrt(hk[s[t]]) * z[t]
        r[t] = mu + eps
        eps_prev = eps
    import pandas as pd
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(r/SCALE, index=idx), s


if __name__ == "__main__":
    import pandas as pd
    true_regs = [(0.02, 0.03, 0.05, 0.90),     # calm
                 (0.30, 0.10, 0.15, 0.78)]      # crisis
    true_P = np.array([[0.985, 0.015],
                       [0.060, 0.940]])
    r_sim, s_true = _simulate(4000, mu=0.03, regs=true_regs, P=true_P, nu=7, seed=1)

    res = fit_msgarch(r_sim, n_starts=8, seed=3)
    print_msgarch(res, "SIMULATED (recovery check)")
    print("\n  TRUE calm  : omega=0.02 alpha=0.03 gamma=0.05 beta=0.90")
    print("  TRUE crisis: omega=0.30 alpha=0.10 gamma=0.15 beta=0.78")
    print("  TRUE P11=0.985  P22=0.940  nu=7")

    # regime identification accuracy (filtered MAP vs true)
    map_reg = res.filtered.argmax(axis=1)
    acc = (map_reg == s_true).mean()
    print(f"\n  regime classification accuracy (filtered MAP): {acc:.1%}")
