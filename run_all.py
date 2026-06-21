"""
Entry point for the cross-asset volatility project.

Runs the full pipeline in stages. Each stage is independent and prints its own
section, so you can run the fast stages for a quick look or `--all` for the
complete analysis (the backtests re-estimate models on rolling windows and take
several minutes each).

Usage
-----
    python run_all.py                 # fast: data quality + GJR-GARCH fits
    python run_all.py --stages var    # add the VaR skew-decomposition backtest
    python run_all.py --all           # everything (slow: ~30-45 min total)
    python run_all.py --stages data gjr egarch

Stages
------
    data    data quality report for all three assets
    gjr     GJR-GARCH model selection (BIC) + cross-asset parameter table
    var     rolling 95% VaR backtest, symmetric-t vs skew-t (the skew finding)
    msgarch MS-GARCH VaR backtest (the violation-clustering finding on the REIT)
    egarch  GJR vs EGARCH out-of-sample forecast horse race (QLIKE + DM)

SPX and PLD fetch from Yahoo (needs internet); WTI reads the local RWTCd.csv.
"""

from __future__ import annotations
import argparse

from data_pipeline import build_dataset, print_reports

ASSET_NAMES = {"SPX": "S&P 500", "WTI": "WTI Crude", "PLD": "Prologis"}
KEYS = ["SPX", "WTI", "PLD"]
ALL_STAGES = ["data", "gjr", "var", "msgarch", "egarch"]


def _header(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def stage_gjr(returns):
    from gjr_garch import (fit_asset, print_bic_table, print_params,
                           summary_table)
    _header("GJR-GARCH MODEL SELECTION & FITS")
    fits = {}
    for k in KEYS:
        fits[k] = fit_asset(k, ASSET_NAMES[k], returns[k])
        print_bic_table(fits[k])
        print_params(fits[k])
    _header("CROSS-ASSET PARAMETER SUMMARY")
    print(summary_table(fits).to_string())
    return fits


def stage_var(returns):
    """Skew decomposition: symmetric-t under-covers; skew-t restores the rate."""
    import pandas as pd
    from var_backtest import compare_distributions
    _header("VaR BACKTEST: symmetric-t vs skew-t")
    table = pd.concat(
        [compare_distributions(returns[k], label=ASSET_NAMES[k]) for k in KEYS],
        ignore_index=True)
    print(table.to_string(index=False))
    return table


def stage_msgarch(returns):
    """Regime-switching: fixes the violation CLUSTERING the skew fix could not."""
    from ms_var_backtest import msgarch_backtest_report
    _header("MS-GARCH VaR BACKTEST (regime-switching)")
    out = {}
    for k in KEYS:
        out[k] = msgarch_backtest_report(returns[k], label=ASSET_NAMES[k],
                                         refit_every=21)
    return out


def stage_egarch(returns):
    """Out-of-sample GJR vs EGARCH forecast contest."""
    from egarch_horserace import horse_race, horse_race_table
    _header("EGARCH HORSE RACE (QLIKE + Diebold-Mariano)")
    results = [horse_race(returns[k], label=ASSET_NAMES[k], refit_every=21)
               for k in KEYS]
    _header("HORSE RACE SUMMARY")
    print(horse_race_table(results).to_string(index=False))
    return results


def main(stages):
    # data is always loaded (everything downstream needs the returns)
    returns, panel, reports = build_dataset()
    if "data" in stages:
        _header("DATA QUALITY")
        print_reports(reports)

    results = {"returns": returns}
    if "gjr" in stages:
        results["gjr"] = stage_gjr(returns)
    if "var" in stages:
        results["var"] = stage_var(returns)
    if "msgarch" in stages:
        results["msgarch"] = stage_msgarch(returns)
    if "egarch" in stages:
        results["egarch"] = stage_egarch(returns)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-asset volatility pipeline.")
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES,
                    default=["data", "gjr"],
                    help="stages to run (default: data gjr)")
    ap.add_argument("--all", action="store_true",
                    help="run every stage (slow: ~30-45 min)")
    args = ap.parse_args()
    stages = ALL_STAGES if args.all else args.stages
    main(stages)
