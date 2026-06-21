"""
Data pipeline for the cross-asset GJR-GARCH project.

Responsibilities (one per section):
    1. CONFIG      -- declarative asset specs, no logic
    2. LOADERS     -- source-specific fetchers (Yahoo, EIA, CSV)
    3. CLEAN       -- gaps, duplicates, non-positive prices
    4. RETURNS     -- log returns
    5. VALIDATE    -- data-quality assertions that fail loudly
    6. PIPELINE    -- orchestration + on-disk caching for reproducibility

Design notes for interview defense:
    - Adjusted close for equities (handles splits/dividends; raw close would
      inject artificial jumps into returns on ex-div dates).
    - Log returns r_t = ln(P_t / P_{t-1}): additive across time, the convention
      in the GARCH literature.
    - Raw downloads are cached to disk. A re-run reads the cache instead of
      re-hitting the network, so your reported numbers are reproducible and
      not silently dependent on when you ran it.
    - Each asset keeps its full history. A COMMON_WINDOW is defined separately
      for cross-asset work (backtest comparability, DCC-GARCH alignment).
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class AssetSpec:
    name: str
    source: str                 # "yahoo" | "eia" | "csv"
    identifier: str             # ticker, EIA series id, or csv path
    price_col: str = "Adj Close"
    start: str = "2009-01-01"
    end: str = "2026-01-01"

ASSETS = {
    "SPX":      AssetSpec("S&P 500",  "yahoo", "^GSPC", start="2015-01-01", end="2025-12-31"),
    "WTI": AssetSpec("WTI Crude", "csv", "RWTCd.csv", price_col="Price",
                     start="2010-01-01", end="2026-02-02"),
    "PLD":      AssetSpec("Prologis",  "yahoo", "PLD",   start="2009-01-01", end="2023-12-31"),
}

# Overlap window for cross-asset comparison / DCC-GARCH (all three trade here).
COMMON_WINDOW = ("2015-01-01", "2023-12-31")

CACHE_DIR = "data_cache"
EIA_API_KEY = os.environ.get("EIA_API_KEY", "")   # free key from eia.gov; optional


# ----------------------------------------------------------------------
# 2. LOADERS
# ----------------------------------------------------------------------
def _load_yahoo(spec: AssetSpec) -> pd.Series:
    import yfinance as yf
    df = yf.download(spec.identifier, start=spec.start, end=spec.end,
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {spec.identifier}")
    col = spec.price_col if spec.price_col in df.columns else "Close"
    s = df[col]
    if isinstance(s, pd.DataFrame):      # yfinance sometimes returns a 1-col frame
        s = s.iloc[:, 0]
    return s.rename(spec.name)


def _load_eia(spec: AssetSpec) -> pd.Series:
    """WTI spot via the EIA v2 API. Needs a free EIA_API_KEY env var."""
    if not EIA_API_KEY:
        raise RuntimeError(
            "No EIA_API_KEY set. Either export one (free at eia.gov), "
            "download the RWTC CSV manually and use source='csv', "
            "or switch this asset to yahoo 'CL=F' as a documented proxy.")
    import requests
    url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
    params = {
        "api_key": EIA_API_KEY, "frequency": "daily",
        "data[0]": "value", "facets[series][]": "RWTC",
        "start": spec.start, "end": spec.end,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
        "length": 5000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    s = pd.Series({pd.Timestamp(d["period"]): float(d["value"]) for d in rows})
    return s.sort_index().rename(spec.name)


def _load_csv(spec: AssetSpec) -> pd.Series:
    """
    Load the EIA RWTC CSV (Date,Price). Handles:
      - utf-8-sig: strips the BOM Excel/EIA prepend
      - date format 'Jan 02, 1986'
      - a trailing blank row (coerced to NaN, then dropped)
      - trims to the configured [start, end] window (file goes back to 1986)
    """
    df = pd.read_csv(spec.identifier, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]

    date_col = df.columns[0]
    price_col = spec.price_col if spec.price_col in df.columns else df.columns[-1]

    s = pd.Series(
        pd.to_numeric(df[price_col], errors="coerce").values,
        index=pd.to_datetime(df[date_col], format="%b %d, %Y", errors="coerce"),
        name=spec.name,
    )
    s = s[s.index.notna()].dropna().sort_index()
    return s.loc[spec.start:spec.end]


_LOADERS = {"yahoo": _load_yahoo, "eia": _load_eia, "csv": _load_csv}


# ----------------------------------------------------------------------
# 3. CLEAN
# ----------------------------------------------------------------------
def clean_prices(s: pd.Series) -> pd.Series:
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s = s.dropna()
    s = s[s > 0]                          # non-positive prices are data errors
    return s


# ----------------------------------------------------------------------
# 4. RETURNS
# ----------------------------------------------------------------------
def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna().rename(prices.name)


# ----------------------------------------------------------------------
# 5. VALIDATE
# ----------------------------------------------------------------------
@dataclass
class QualityReport:
    name: str
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    max_gap_days: int
    n_extreme: int            # |return| > 5 sigma
    annualized_vol: float

def validate(returns: pd.Series, max_gap: int = 7, sigma_k: float = 5.0) -> QualityReport:
    idx = returns.index
    gaps = idx.to_series().diff().dt.days.fillna(0)
    max_gap_days = int(gaps.max())
    sd = returns.std()
    n_extreme = int((returns.abs() > sigma_k * sd).sum())

    # loud failures: these are bugs, not findings
    assert returns.notna().all(), f"{returns.name}: NaNs survived cleaning"
    assert max_gap_days < 30, f"{returns.name}: {max_gap_days}-day gap suggests missing data"
    if max_gap_days > max_gap:
        print(f"  [warn] {returns.name}: largest gap {max_gap_days}d (holidays/weekends ok)")

    return QualityReport(
        name=returns.name, n_obs=len(returns),
        start=idx.min(), end=idx.max(),
        max_gap_days=max_gap_days, n_extreme=n_extreme,
        annualized_vol=float(sd * np.sqrt(252)),
    )


# ----------------------------------------------------------------------
# 6. PIPELINE
# ----------------------------------------------------------------------
def load_asset(key: str, use_cache: bool = True) -> pd.Series:
    """Return cleaned PRICES for one asset, caching the raw pull to disk."""
    spec = ASSETS[key]
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{key}_raw.csv")

    if use_cache and os.path.exists(cache):
        raw = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        raw.name = spec.name
    else:
        raw = _LOADERS[spec.source](spec)
        raw.to_frame().to_csv(cache)      # cache the raw pull
    return clean_prices(raw)


def build_dataset(keys=None, use_cache: bool = True):
    """
    Returns
    -------
    returns : dict[str, pd.Series]   per-asset log returns (full history)
    panel   : pd.DataFrame           aligned returns over COMMON_WINDOW (for DCC)
    reports : dict[str, QualityReport]
    """
    keys = keys or list(ASSETS)
    returns, reports = {}, {}
    for k in keys:
        prices = load_asset(k, use_cache)
        r = log_returns(prices)
        returns[k] = r
        reports[k] = validate(r)

    lo, hi = COMMON_WINDOW
    panel = pd.DataFrame({k: v for k, v in returns.items()}).loc[lo:hi].dropna()
    return returns, panel, reports


def print_reports(reports: dict):
    print(f"\n{'asset':<10}{'obs':>7}{'start':>13}{'end':>13}"
          f"{'maxgap':>8}{'extreme':>9}{'ann.vol':>9}")
    for r in reports.values():
        print(f"{r.name:<10}{r.n_obs:>7}{str(r.start.date()):>13}"
              f"{str(r.end.date()):>13}{r.max_gap_days:>8}"
              f"{r.n_extreme:>9}{r.annualized_vol:>9.2%}")


# ----------------------------------------------------------------------
# Self-test: synthetic prices, exercises clean/returns/validate (no network)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", "2023-12-31")
    out = {}
    for name, vol in [("S&P 500", 0.011), ("WTI Crude", 0.025), ("Prologis", 0.016)]:
        rets = rng.normal(0, vol, len(dates))
        prices = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates, name=name)
        prices.iloc[50] = np.nan          # inject a hole
        prices.iloc[100] = -1.0           # inject a bad price
        clean = clean_prices(prices)
        r = log_returns(clean)
        out[name] = validate(r)
    print_reports(out)
    print("\nSelf-test passed: clean/returns/validate handle holes and bad prices.")
