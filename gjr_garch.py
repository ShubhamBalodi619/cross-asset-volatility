"""
GJR-GARCH model layer for the cross-asset volatility project.

Reproduces and hardens the deck's methodology:
    - Fit four candidate specifications per asset.
    - Select the best by BIC (penalizes complexity; favors parsimony).
    - Validate the winner with Ljung-Box on standardized residuals (mean
      dynamics captured?) and squared standardized residuals (variance
      dynamics captured?).
    - Extract the structural parameters (omega, alpha, gamma, beta, nu) that
      tell the cross-asset story.

The four candidates isolate the two modeling choices the deck made:
    symmetric vs asymmetric volatility (GARCH vs GJR), and
    normal vs fat-tailed innovations (Normal vs Student's-t).
If GJR-t wins on BIC, you've *shown* that both the leverage effect and the
fat tails earn their keep -- you haven't just assumed them.

Interview-defense notes:
    - arch is fit on returns in PERCENT (returns*100). GARCH optimizers behave
      poorly when the data is ~1e-2; scaling to ~O(1) is standard. Parameters
      omega scales with the variance units; alpha/beta/gamma/nu are unit-free.
    - gamma > 0 and significant => bad news raises vol more than good news of
      the same size (the leverage effect).
    - Persistence = alpha + beta + gamma/2 must be < 1 for stationarity. The
      gamma/2 enters because the leverage term is active half the time on
      average (negative shocks).
"""

from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

SCALE = 100.0   # fit on percent returns


# ----------------------------------------------------------------------
# Candidate specifications
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    label: str
    p: int          # ARCH (alpha) order
    o: int          # asymmetry (gamma) order; o=0 -> symmetric GARCH
    q: int          # GARCH (beta) order
    dist: str       # "normal" | "t"

CANDIDATES = [
    ModelSpec("GARCH(1,1)-Normal",     1, 0, 1, "normal"),
    ModelSpec("GARCH(1,1)-t",          1, 0, 1, "t"),
    ModelSpec("GJR-GARCH(1,1,1)-Normal", 1, 1, 1, "normal"),
    ModelSpec("GJR-GARCH(1,1,1)-t",      1, 1, 1, "t"),
]


# ----------------------------------------------------------------------
# Fit one specification
# ----------------------------------------------------------------------
def fit_one(returns: pd.Series, spec: ModelSpec):
    y = returns.dropna() * SCALE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(y, mean="Constant", vol="GARCH",
                        p=spec.p, o=spec.o, q=spec.q, dist=spec.dist)
        res = am.fit(disp="off")
    return res


# ----------------------------------------------------------------------
# Fit all candidates, rank by BIC
# ----------------------------------------------------------------------
def fit_candidates(returns: pd.Series) -> pd.DataFrame:
    rows = []
    for spec in CANDIDATES:
        res = fit_one(returns, spec)
        rows.append({
            "model": spec.label, "spec": spec,
            "loglik": res.loglikelihood, "aic": res.aic, "bic": res.bic,
            "result": res,
        })
    df = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
    df["delta_bic"] = df["bic"] - df["bic"].min()   # 0 for the winner
    return df


def select_best(candidate_df: pd.DataFrame):
    best = candidate_df.iloc[0]
    return best["result"], best["spec"], best


# ----------------------------------------------------------------------
# Ljung-Box validation
# ----------------------------------------------------------------------
def ljung_box(res, lags: int = 10) -> dict:
    """
    Test the fitted model's standardized residuals.
      - level resids autocorrelated  -> mean spec misses structure
      - squared resids autocorrelated -> variance spec (the GARCH) misses
        structure; this is the one that matters for a vol model.
    Null hypothesis: no autocorrelation. p > 0.05 => PASS (good).
    """
    z = res.std_resid.dropna()
    lb_level = acorr_ljungbox(z, lags=[lags], return_df=True)
    lb_sq = acorr_ljungbox(z**2, lags=[lags], return_df=True)
    p_level = float(lb_level["lb_pvalue"].iloc[0])
    p_sq = float(lb_sq["lb_pvalue"].iloc[0])
    return {
        "lags": lags,
        "p_level": p_level, "level_ok": p_level > 0.05,
        "p_squared": p_sq, "squared_ok": p_sq > 0.05,
    }


# ----------------------------------------------------------------------
# Parameter extraction
# ----------------------------------------------------------------------
def extract_params(res, spec: ModelSpec) -> dict:
    p = res.params
    out = {
        "mu": p.get("mu", np.nan),
        "omega": p.get("omega", np.nan),
        "alpha": p.get("alpha[1]", np.nan),
        "gamma": p.get("gamma[1]", 0.0) if spec.o > 0 else 0.0,
        "beta": p.get("beta[1]", np.nan),
        "nu": p.get("nu", np.nan) if spec.dist == "t" else np.nan,
    }
    # stationarity / persistence
    out["persistence"] = out["alpha"] + out["beta"] + (out["gamma"] / 2.0
                                                       if spec.o > 0 else 0.0)
    # is the leverage term statistically significant?
    if spec.o > 0 and "gamma[1]" in res.pvalues:
        out["gamma_pvalue"] = float(res.pvalues["gamma[1]"])
    else:
        out["gamma_pvalue"] = np.nan
    return out


# ----------------------------------------------------------------------
# Full per-asset workflow
# ----------------------------------------------------------------------
@dataclass
class AssetFit:
    key: str
    name: str
    candidates: pd.DataFrame
    best_spec: ModelSpec
    result: object
    params: dict
    diagnostics: dict

def fit_asset(key: str, name: str, returns: pd.Series) -> AssetFit:
    cands = fit_candidates(returns)
    res, spec, _ = select_best(cands)
    params = extract_params(res, spec)
    diag = ljung_box(res)
    return AssetFit(key, name, cands, spec, res, params, diag)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def print_bic_table(fit: AssetFit):
    print(f"\n--- {fit.name}: model selection (BIC) ---")
    show = fit.candidates[["model", "loglik", "bic", "delta_bic"]].copy()
    for _, row in show.iterrows():
        star = "  <- selected" if row["delta_bic"] == 0 else ""
        print(f"  {row['model']:<26} loglik={row['loglik']:>10.1f} "
              f"BIC={row['bic']:>9.1f}  dBIC={row['delta_bic']:>6.1f}{star}")

def print_params(fit: AssetFit):
    p = fit.params
    print(f"--- {fit.name}: {fit.best_spec.label} ---")
    print(f"  omega={p['omega']:.4f}  alpha={p['alpha']:.4f}  "
          f"gamma={p['gamma']:.4f}  beta={p['beta']:.4f}  nu={p['nu']:.4f}")
    print(f"  persistence(alpha+beta+gamma/2) = {p['persistence']:.4f}"
          f"  {'(stationary)' if p['persistence'] < 1 else '(NON-STATIONARY!)'}")
    if not np.isnan(p["gamma_pvalue"]):
        sig = "significant" if p["gamma_pvalue"] < 0.05 else "NOT significant"
        print(f"  leverage gamma p-value = {p['gamma_pvalue']:.4f} ({sig})")
    d = fit.diagnostics
    print(f"  Ljung-Box(L={d['lags']}): level p={d['p_level']:.3f} "
          f"{'PASS' if d['level_ok'] else 'FAIL'}, "
          f"squared p={d['p_squared']:.3f} "
          f"{'PASS' if d['squared_ok'] else 'FAIL'}")

def summary_table(fits: dict):
    """Cross-asset parameter table -- the resume-bullet payload."""
    rows = []
    for f in fits.values():
        p = f.params
        rows.append({
            "asset": f.name, "model": f.best_spec.label,
            "omega": p["omega"], "alpha": p["alpha"], "gamma": p["gamma"],
            "beta": p["beta"], "nu": p["nu"], "persistence": p["persistence"],
        })
    return pd.DataFrame(rows).set_index("asset").round(4)


# ----------------------------------------------------------------------
# Self-test: real WTI from CSV + synthetic SPX/PLD (no network needed)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from data_pipeline import load_asset, log_returns

    fits = {}

    # real crude from the uploaded CSV
    wti = log_returns(load_asset("WTI", use_cache=False))
    fits["WTI"] = fit_asset("WTI", "WTI Crude", wti)

    # synthetic equity/REIT stand-ins so the full workflow runs offline
    rng = np.random.default_rng(1)
    for key, name, vol, lev in [("SPX", "S&P 500", 0.011, 0.25),
                                ("PLD", "Prologis", 0.016, 0.06)]:
        n = 2200
        eps = np.zeros(n); s2 = np.zeros(n); s2[0] = 1.0
        z = rng.standard_t(df=6, size=n) * np.sqrt(4/6)
        for t in range(1, n):
            sh = eps[t-1]
            s2[t] = 0.02 + (0.03 + lev*(sh < 0))*sh**2 + 0.88*s2[t-1]
            eps[t] = np.sqrt(s2[t]) * z[t]
        idx = pd.bdate_range("2015-01-01", periods=n)
        fits[key] = fit_asset(key, name, pd.Series(eps/100*vol/0.011, index=idx))

    for f in fits.values():
        print_bic_table(f)
        print_params(f)

    print("\n=== CROSS-ASSET SUMMARY ===")
    print(summary_table(fits).to_string())
