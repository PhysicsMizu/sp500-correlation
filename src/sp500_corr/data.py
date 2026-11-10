"""Download S&P 500 constituents and daily prices, with on-disk caching."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def get_constituents(data_dir: Path = DATA_DIR, refresh: bool = False) -> pd.DataFrame:
    """Current S&P 500 members with GICS sector, scraped from Wikipedia.

    Note: using *today's* members over a historical window introduces
    survivorship bias. That's fine for studying correlation structure, but
    don't read the returns themselves as a backtest of "the S&P 500".
    """
    path = data_dir / "constituents.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)

    html = requests.get(
        WIKI_URL, headers={"User-Agent": "sp500-corr/0.1 (research script)"}, timeout=30
    ).text
    df = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    df = df.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )[["ticker", "name", "sector", "sub_industry"]]
    # Yahoo uses dashes for share classes: BRK.B -> BRK-B
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def get_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    data_dir: Path = DATA_DIR,
    refresh: bool = False,
    name: str = "prices",
) -> pd.DataFrame:
    """Daily dividend/split-adjusted closes (dates x tickers), cached as parquet."""
    import yfinance as yf

    path = data_dir / f"{name}_{start}_{end or 'latest'}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=True, threads=True
    )
    prices = raw["Close"].sort_index()
    prices = prices.dropna(axis=1, how="all")

    data_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return prices


FACTOR_ETFS = ["SPY", "TLT", "USO", "GLD", "UUP", "IWM", "IVE", "IVW", "MTUM", "USMV",
               "JETS", "SMH", "KRE", "HYG", "^VIX"]


def get_factor_returns(start: str, end: str | None = None, data_dir: Path = DATA_DIR,
                       refresh: bool = False) -> pd.DataFrame:
    """Daily log returns of ETF proxies for macro and style factors.

    Style factors are long-short spreads against SPY so they aren't just
    "the market" again.
    """
    px = get_prices(FACTOR_ETFS, start, end, data_dir=data_dir, refresh=refresh,
                    name="factors")
    r = np.log(px).diff()
    return pd.DataFrame(
        {
            "Market (SPY)": r["SPY"],
            "Volatility (VIX)": r["^VIX"],
            "Credit (HYG)": r["HYG"],
            "Long bonds (TLT)": r["TLT"],
            "Oil (USO)": r["USO"],
            "Gold (GLD)": r["GLD"],
            "Dollar (UUP)": r["UUP"],
            "Small − large (IWM−SPY)": r["IWM"] - r["SPY"],
            "Value − growth (IVE−IVW)": r["IVE"] - r["IVW"],
            "Momentum (MTUM−SPY)": r["MTUM"] - r["SPY"],
            "Low vol (USMV−SPY)": r["USMV"] - r["SPY"],
            "Semis (SMH−SPY)": r["SMH"] - r["SPY"],
            "Regional banks (KRE−SPY)": r["KRE"] - r["SPY"],
            "Airlines (JETS−SPY)": r["JETS"] - r["SPY"],
        }
    ).iloc[1:]


def log_returns(prices: pd.DataFrame, min_coverage: float = 0.99) -> pd.DataFrame:
    """Daily log returns on a balanced panel.

    A correlation matrix must come from a *common* set of dates to be positive
    semi-definite, so we drop tickers with too much missing history (recent IPOs,
    spin-offs) and then drop any remaining dates with gaps.
    """
    coverage = prices.notna().mean()
    prices = prices.loc[:, coverage >= min_coverage]
    rets = np.log(prices).diff().iloc[1:]
    rets = rets.dropna(axis=0, how="any")
    # Remove constant columns (halted/bad data), which would give NaN correlations
    rets = rets.loc[:, rets.std() > 0]
    return rets
