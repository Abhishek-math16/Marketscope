"""
app.py
------
The web layer. It wires together data.py, model.py and database.py and
exposes a few clean endpoints. The heavy lifting lives in the other modules;
this file just orchestrates and returns JSON the frontend can render.

Endpoints
  GET  /                 → the single-page dashboard
  POST /api/predict      → run the full pipeline for a ticker, return JSON
  GET  /api/recent       → recently searched tickers
  GET  /download/<file>  → download a saved CSV dataset
"""

import os
import traceback

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, abort

import data as datalib
import model as modellib
import database as db

app = Flask(__name__)

for d in ("models", "datasets", "data", "static"):
    os.makedirs(d, exist_ok=True)

db.init_db()


# ────────────────────────────────────────────────────────────────────
# Plain-English trading signals (the beginner-friendly bit)
# ────────────────────────────────────────────────────────────────────
def build_signals(df):
    """Translate the latest indicator values into plain-English signals."""
    last = df.iloc[-1]
    signals = []

    # Short-term trend
    if last["EMA20"] > last["EMA50"]:
        signals.append({
            "label": "Short-term trend",
            "value": "Upward",
            "sentiment": "positive",
            "explain": "The 20-day average price is above the 50-day average, "
                       "which usually means recent buying interest is stronger "
                       "than the slightly longer-term trend.",
        })
    else:
        signals.append({
            "label": "Short-term trend",
            "value": "Downward",
            "sentiment": "negative",
            "explain": "The 20-day average price is below the 50-day average, "
                       "which usually means recent selling pressure.",
        })

    # Long-term trend (Golden / Death cross)
    if last["EMA100"] > last["EMA200"]:
        signals.append({
            "label": "Long-term trend",
            "value": "Golden cross",
            "sentiment": "positive",
            "explain": "The 100-day average is above the 200-day average — a "
                       "'Golden Cross', traditionally seen as a longer-term "
                       "bullish (rising) signal.",
        })
    else:
        signals.append({
            "label": "Long-term trend",
            "value": "Death cross",
            "sentiment": "negative",
            "explain": "The 100-day average is below the 200-day average — a "
                       "'Death Cross', traditionally seen as a longer-term "
                       "bearish (falling) signal.",
        })

    # RSI
    rsi = float(last["RSI"])
    if rsi >= 70:
        rsi_sent, rsi_word = "negative", "Overbought"
        rsi_exp = ("RSI above 70 can mean the stock has risen quickly and may "
                   "be due for a pause or pullback.")
    elif rsi <= 30:
        rsi_sent, rsi_word = "positive", "Oversold"
        rsi_exp = ("RSI below 30 can mean the stock has fallen quickly and may "
                   "be due for a bounce.")
    else:
        rsi_sent, rsi_word = "neutral", "Neutral"
        rsi_exp = ("RSI between 30 and 70 suggests momentum is balanced — "
                   "neither overbought nor oversold.")
    signals.append({
        "label": f"Momentum (RSI {rsi:.0f})",
        "value": rsi_word,
        "sentiment": rsi_sent,
        "explain": rsi_exp,
    })

    # MACD
    if last["MACD_hist"] > 0:
        signals.append({
            "label": "MACD momentum",
            "value": "Positive",
            "sentiment": "positive",
            "explain": "MACD is above its signal line, pointing to building "
                       "upward momentum.",
        })
    else:
        signals.append({
            "label": "MACD momentum",
            "value": "Negative",
            "sentiment": "negative",
            "explain": "MACD is below its signal line, pointing to building "
                       "downward momentum.",
        })

    return signals


# ────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/recent")
def api_recent():
    return jsonify({"tickers": db.recent_searches()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    ticker = (payload.get("ticker") or "").upper().strip()
    if not ticker:
        return jsonify({"error": "Please enter a ticker symbol."}), 400

    try:
        # 1. Data
        raw = datalib.fetch_data(ticker)
        if raw is None:
            return jsonify({
                "error": f"Couldn't fetch data for '{ticker}'. This is usually "
                         "one of two things: the symbol is wrong (Indian stocks "
                         "need a suffix — RELIANCE.NS for NSE, RELIANCE.BO for "
                         "BSE), or Yahoo's data service is temporarily blocking "
                         "requests. Wait a few seconds and try again."
            }), 502
        if len(raw) < 200:
            return jsonify({
                "error": f"Only {len(raw)} days of history for '{ticker}' — "
                         "not enough to train on. Try a stock with a longer "
                         "track record."
            }), 422

        df = datalib.add_indicators(raw)
        db.log_search(ticker)

        # 2. Save a CSV for download
        csv_name = f"{ticker.replace('/', '_')}_data.csv"
        raw.to_csv(os.path.join("datasets", csv_name))

        # 3. Train or reuse a saved model
        trained_now = False
        if modellib.model_exists(ticker) and db.model_is_fresh(ticker):
            model, scalers = modellib.load_cached(ticker)
            evaluation = modellib._evaluate(
                model, scalers, df,
                modellib.make_features(df),
                int(len(modellib.make_features(df)) * 0.80),
            )
        else:
            model, scalers, evaluation = modellib.train(df)
            modellib.save(ticker, model, scalers)
            db.mark_model_trained(ticker)
            trained_now = True

        # 4. Forecast
        fc = modellib.forecast(model, scalers, df, evaluation["resid_std"], days=30)

        # 5. Persist metrics
        last_price = float(df["Close"].iloc[-1])
        db.save_prediction(ticker, evaluation["metrics"], last_price)

        # 6. Build chart data (raw is already daily)
        weekly = datalib.resample_ohlc(raw, "W")
        monthly = datalib.resample_ohlc(raw, "M")

        prev_close = float(df["Close"].iloc[-2])
        change_pct = (last_price - prev_close) / prev_close * 100

        response = {
            "ticker": ticker,
            "last_price": round(last_price, 2),
            "change_pct": round(change_pct, 2),
            "rows": int(len(raw)),
            "start": raw.index[0].strftime("%Y-%m-%d"),
            "end": raw.index[-1].strftime("%Y-%m-%d"),
            "trained_now": trained_now,

            "ohlc": {
                "daily": datalib.ohlc_records(raw),
                "weekly": datalib.ohlc_records(weekly),
                "monthly": datalib.ohlc_records(monthly),
            },
            "ema": {
                "ema20": datalib.series_records(df.index, df["EMA20"]),
                "ema50": datalib.series_records(df.index, df["EMA50"]),
                "ema100": datalib.series_records(df.index, df["EMA100"]),
                "ema200": datalib.series_records(df.index, df["EMA200"]),
            },
            "rsi": datalib.series_records(df.index, df["RSI"]),
            "macd": {
                "macd": datalib.series_records(df.index, df["MACD"]),
                "signal": datalib.series_records(df.index, df["MACD_signal"]),
                "hist": datalib.series_records(df.index, df["MACD_hist"]),
            },

            "prediction": evaluation["test"],
            "metrics": evaluation["metrics"],
            "forecast": fc,
            "signals": build_signals(df),
            "csv": csv_name,
        }
        return jsonify(response)

    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({
            "error": "Something went wrong while building the analysis. "
                     f"Technical detail: {exc}"
        }), 500


@app.route("/download/<path:filename>")
def download_file(filename):
    path = os.path.join("datasets", filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    abort(404)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
