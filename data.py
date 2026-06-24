"""
data.py
-------
Everything related to getting raw market data and turning it into the
technical indicators the rest of the app uses.

IMPORTANT — why fetching looks more involved than a plain yf.download():
Yahoo Finance now rate-limits and blocks plain requests, which makes yfinance
return an empty result for perfectly valid tickers. The fix the community has
settled on is to send requests through a browser-impersonating curl_cffi
session. We do that here, try two fetch methods, and retry with backoff.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
except Exception:          # curl_cffi not installed for some reason
    cffi_requests = None


def _make_session():
    """A Chrome-impersonating session gets past Yahoo's blocking. If curl_cffi
    isn't available we return None and yfinance uses its own default."""
    if cffi_requests is None:
        return None
    try:
        return cffi_requests.Session(impersonate="chrome")
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# 1. Download raw prices  (robust — this is the part that used to fail)
# ────────────────────────────────────────────────────────────────────
def fetch_data(ticker, period="10y"):
    """
    Download daily OHLCV data for a ticker and return a clean DataFrame.
    Tries Ticker().history first (most reliable), then yf.download as a
    fallback, retrying each. Returns None only if everything failed.
    """
    session = _make_session()

    df = _history(ticker, period, session)
    if df is None or df.empty:
        df = _download(ticker, period, session)
    if df is None or df.empty:
        return None

    # Newer yfinance returns MultiIndex columns even for one ticker,
    # e.g. ('Close', 'AAPL'). Flatten them so we always get 'Close'.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    # Some methods return a timezone-aware index; drop the tz so resampling
    # and plotting behave consistently.
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.dropna(inplace=True)
    return df


def _history(ticker, period, session, retries=3):
    """Primary method: Ticker(...).history — works best with a session."""
    for attempt in range(retries):
        try:
            try:
                tk = yf.Ticker(ticker, session=session)
            except TypeError:
                # Some yfinance versions don't accept the session kwarg
                tk = yf.Ticker(ticker)
            df = tk.history(period=period, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1 + attempt)        # simple backoff
    return None


def _download(ticker, period, session, retries=2):
    """Fallback method: yf.download."""
    for attempt in range(retries):
        try:
            kwargs = dict(period=period, auto_adjust=True,
                          progress=False, threads=False)
            try:
                df = yf.download(ticker, session=session, **kwargs)
            except TypeError:
                df = yf.download(ticker, **kwargs)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        time.sleep(1 + attempt)
    return None


# ────────────────────────────────────────────────────────────────────
# 2. Technical indicators
# ────────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """
    Relative Strength Index (RSI). A momentum oscillator from 0–100.
    Above 70 is often called 'overbought', below 30 'oversold'.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Moving Average Convergence Divergence (MACD).
    Returns the MACD line, its signal line, and the histogram (their gap).
    A histogram crossing above zero is read as bullish momentum.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist


def add_indicators(df):
    """Attach EMA / RSI / MACD columns to the price DataFrame."""
    df = df.copy()
    close = df["Close"]

    for span in (20, 50, 100, 200):
        df[f"EMA{span}"] = close.ewm(span=span, adjust=False).mean()

    df["RSI"] = compute_rsi(close, 14)
    macd, signal_line, hist = compute_macd(close)
    df["MACD"] = macd
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = hist
    return df


# ────────────────────────────────────────────────────────────────────
# 3. Resampling for weekly / monthly views
# ────────────────────────────────────────────────────────────────────
def resample_ohlc(df, rule):
    """
    Aggregate daily candles into weekly or monthly candles so the user can
    zoom out to a longer-term view.

    `rule` accepts friendly names 'W' (weekly) or 'M' (monthly). We resolve
    monthly to the right pandas alias automatically, because pandas renamed
    'M' to 'ME' in version 2.2 — this keeps the app working on any version.
    """
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    cols = [c for c in agg if c in df.columns]
    plan = {c: agg[c] for c in cols}

    candidates = ["ME", "M"] if rule.upper().startswith("M") else [rule]
    last_err = None
    for freq in candidates:
        try:
            return df[cols].resample(freq).agg(plan).dropna()
        except ValueError as err:
            last_err = err
    raise last_err


# ────────────────────────────────────────────────────────────────────
# 4. Helpers to turn DataFrames into JSON-friendly lists for the charts
# ────────────────────────────────────────────────────────────────────
def ohlc_records(df):
    """Convert an OHLC DataFrame into a list of dicts the frontend can plot."""
    out = df.reset_index()
    date_col = out.columns[0]
    records = []
    for _, row in out.iterrows():
        records.append({
            "t": pd.Timestamp(row[date_col]).strftime("%Y-%m-%d"),
            "o": round(float(row["Open"]), 2),
            "h": round(float(row["High"]), 2),
            "l": round(float(row["Low"]), 2),
            "c": round(float(row["Close"]), 2),
            "v": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
        })
    return records


def series_records(index, values):
    """Pair a date index with a value series, skipping NaNs."""
    out = []
    for dt, val in zip(index, values):
        if pd.isna(val):
            continue
        out.append({
            "t": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "v": round(float(val), 4),
        })
    return out
