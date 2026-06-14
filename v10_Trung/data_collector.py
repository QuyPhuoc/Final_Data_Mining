from __future__ import annotations

import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_COLS = ["date", "open", "high", "low", "close", "volume"]

_VN_PATTERN = re.compile(r"^[A-Z]{3}$")
_CRYPTO_PAIRS = {"BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC"}

SourceType = Literal["vnstock", "yfinance_stock", "yfinance_crypto"]

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Source Detection
# ---------------------------------------------------------------------------

def detect_source(ticker: str) -> SourceType:
    t = ticker.strip().upper()

    if "-USD" in t or any(c in t for c in _CRYPTO_PAIRS):
        return "yfinance_crypto"

    if _VN_PATTERN.match(t):
        return "vnstock"

    return "yfinance_stock"


# ---------------------------------------------------------------------------
# VNStock Fetch
# ---------------------------------------------------------------------------

def _fetch_vnstock(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        from vnstock import Quote

        quote = Quote(symbol=ticker.upper(), source="VCI")
        raw = quote.history(start=start, end=end, interval="1d")

        if raw is None or raw.empty:
            quote = Quote(symbol=ticker.upper(), source="TCBS")
            raw = quote.history(start=start, end=end, interval="1d")

    except Exception as e:
        logger.warning(f"VCI fail: {e}, fallback stock_historical_data")

        try:
            from vnstock import stock_historical_data

            raw = stock_historical_data(
                symbol=ticker.upper(),
                start_date=start,
                end_date=end,
                resolution="1D",
            )
        except Exception as e2:
            raise ImportError(f"vnstock failed completely: {e2}")

    if raw is None or raw.empty:
        raise ValueError(f"No data for {ticker}")

    df = raw.copy()
    df = df.reset_index()

    df = df.rename(columns={
        "time": "date",
        "Date": "date",
        "trading_date": "date",
    })

    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[(df["date"] >= start) & (df["date"] <= end)]

    if df.empty:
        raise ValueError(f"No data in range {start} → {end} for {ticker}")

    return df


# ---------------------------------------------------------------------------
# YFinance Fetch
# ---------------------------------------------------------------------------

def _fetch_yfinance(ticker: str, start: str, end: str, source_type: SourceType) -> pd.DataFrame:
    import yfinance as yf

    t = ticker.upper()

    if source_type == "yfinance_crypto" and not t.endswith("-USD"):
        t = f"{t}-USD"

    raw = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)

    if raw is None or raw.empty:
        raise ValueError(f"yfinance empty for {t}")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = df.columns.str.lower()

    return df


# ---------------------------------------------------------------------------
# Validation & Cleaning
# ---------------------------------------------------------------------------

def validate_and_clean(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()

    missing = [c for c in STANDARD_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker} missing columns: {missing}")

    df = df[STANDARD_COLS]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # drop invalid rows
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df[["open", "high", "low", "close"]] > 0].dropna()

    df["volume"] = df["volume"].fillna(0).astype("int64")

    df = df.sort_values("date").drop_duplicates("date")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, source: SourceType) -> Path:
    return CACHE_DIR / f"{ticker}_{source}.parquet"


def _csv_path(ticker: str, source: SourceType) -> Path:
    return CACHE_DIR / f"{ticker}_{source}.csv"


def _save_cache(df: pd.DataFrame, ticker: str, source: SourceType):
    try:
        df.to_parquet(_cache_path(ticker, source), index=False)
    except Exception as e:
        logger.warning(f"Parquet save failed: {e}")
        _save_csv(df, ticker, source)


def _save_csv(df: pd.DataFrame, ticker: str, source: SourceType):
    try:
        df.to_csv(_csv_path(ticker, source), index=False, encoding="utf-8")
    except Exception as e:
        logger.warning(f"CSV save failed: {e}")


def _load_cache(ticker: str, source: SourceType, start: str, end: str):
    path = _cache_path(ticker, source)

    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
        result = df.loc[mask]

        return result.reset_index(drop=True) if not result.empty else None

    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def fetch_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:

    today = datetime.today().strftime("%Y-%m-%d")

    end = end or today
    start = start or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    ticker = ticker.strip().upper()
    source = detect_source(ticker)

    # cache
    if use_cache:
        cached = _load_cache(ticker, source, start, end)
        if cached is not None and not cached.empty:
            logger.info(f"{ticker}: using cache")
            return cached

    # fetch
    if source == "vnstock":
        raw = _fetch_vnstock(ticker, start, end)
    else:
        raw = _fetch_yfinance(ticker, start, end, source)

    df = validate_and_clean(raw, ticker)

    # save
    if not df.empty:
        _save_cache(df, ticker, source)
        if save_csv:
            _save_csv(df, ticker, source)

    return df