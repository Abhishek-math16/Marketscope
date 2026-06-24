"""
model.py
--------
The machine-learning core.

KEY DESIGN DECISION — we predict the next day's *return* (percentage change),
not the next day's raw price.

Why this matters: if you train a model on raw prices, it quickly learns the
laziest possible trick — "tomorrow's price ≈ today's price". The prediction
chart then looks almost perfect, but the model has learned nothing useful; it
is just echoing the last value one day late. By predicting the *return* we
remove that shortcut and force the model to learn something about direction
and magnitude. It also makes the model comparable across stocks at very
different price levels.

We always compare against a NAIVE BASELINE ("tomorrow = today"). If the LSTM
cannot beat that baseline, we say so honestly — that is a mature, defensible
result to present in a viva.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

import data as datalib

SEQ_LEN = 60                       # how many past days the model looks at
FEATURES = ["ret", "rsi", "macd_hist", "vol_chg"]
MODEL_DIR = "models"


# ────────────────────────────────────────────────────────────────────
# Feature engineering
# ────────────────────────────────────────────────────────────────────
def make_features(df):
    """
    Build the model's input features from a price DataFrame that already has
    indicators attached (via data.add_indicators).

    Features:
      ret        – log return of close (the main signal)
      rsi        – RSI scaled to 0–1
      macd_hist  – MACD histogram normalised by recent volatility
      vol_chg    – day-over-day change in volume (clipped)
    Target:
      next day's log return.
    """
    out = pd.DataFrame(index=df.index)
    out["ret"] = np.log(df["Close"] / df["Close"].shift(1))
    out["rsi"] = df["RSI"] / 100.0
    out["macd_hist"] = df["MACD_hist"] / (df["Close"].rolling(20).std() + 1e-8)
    out["vol_chg"] = df["Volume"].pct_change().clip(-3, 3)
    out["target"] = out["ret"].shift(-1)
    out = out.dropna()
    return out


def _make_sequences(feat_scaled, target_scaled, seq_len):
    X, y = [], []
    for i in range(seq_len, len(feat_scaled)):
        X.append(feat_scaled[i - seq_len:i])
        y.append(target_scaled[i])
    return np.array(X), np.array(y)


def _build_model(n_features):
    model = Sequential([
        Input(shape=(SEQ_LEN, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ────────────────────────────────────────────────────────────────────
# Save / load so we don't retrain on every single request
# ────────────────────────────────────────────────────────────────────
def _paths(ticker):
    safe = ticker.replace(".", "_").replace("/", "_")
    return (
        os.path.join(MODEL_DIR, f"{safe}.keras"),
        os.path.join(MODEL_DIR, f"{safe}_scalers.pkl"),
    )


def model_exists(ticker):
    mp, sp = _paths(ticker)
    return os.path.exists(mp) and os.path.exists(sp)


def load_cached(ticker):
    mp, sp = _paths(ticker)
    model = load_model(mp)
    scalers = joblib.load(sp)
    return model, scalers


# ────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────
def train(df, epochs=20):
    """
    Train the LSTM on a price+indicator DataFrame.
    Returns (model, scalers, evaluation_dict).
    """
    feat = make_features(df)
    feature_matrix = feat[FEATURES].values
    target = feat["target"].values.reshape(-1, 1)

    # 80 / 20 split — fit scalers ONLY on the training portion to avoid leakage
    split = int(len(feat) * 0.80)

    fscaler = StandardScaler().fit(feature_matrix[:split])
    tscaler = StandardScaler().fit(target[:split])

    feat_scaled = fscaler.transform(feature_matrix)
    target_scaled = tscaler.transform(target).flatten()

    X, y = _make_sequences(feat_scaled, target_scaled, SEQ_LEN)

    # Re-align the split to the sequence arrays
    seq_split = split - SEQ_LEN
    X_train, y_train = X[:seq_split], y[:seq_split]
    X_test, y_test = X[seq_split:], y[seq_split:]

    model = _build_model(len(FEATURES))
    early = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[early],
    )

    scalers = {"feature": fscaler, "target": tscaler}
    evaluation = _evaluate(model, scalers, df, feat, split)
    return model, scalers, evaluation


def save(ticker, model, scalers):
    os.makedirs(MODEL_DIR, exist_ok=True)
    mp, sp = _paths(ticker)
    model.save(mp)
    joblib.dump(scalers, sp)


# ────────────────────────────────────────────────────────────────────
# Evaluate (in real price units, with a baseline)
# ────────────────────────────────────────────────────────────────────
def _evaluate(model, scalers, df, feat, split):
    """
    Produce honest, price-space metrics on the test set.

    We turn the predicted RETURN back into a predicted PRICE using the actual
    previous close as the anchor (one-step-ahead prediction):
        predicted_price[t+1] = actual_close[t] * exp(predicted_return[t+1])
    """
    fscaler, tscaler = scalers["feature"], scalers["target"]
    feat_scaled = fscaler.transform(feat[FEATURES].values)
    target_scaled = tscaler.transform(feat["target"].values.reshape(-1, 1)).flatten()

    X, _ = _make_sequences(feat_scaled, target_scaled, SEQ_LEN)
    seq_split = split - SEQ_LEN
    X_test = X[seq_split:]

    pred_scaled = model.predict(X_test, verbose=0)
    pred_returns = tscaler.inverse_transform(pred_scaled).flatten()

    # The dates these predictions correspond to. feat is offset by SEQ_LEN.
    test_dates = feat.index[SEQ_LEN + seq_split:]
    close = df["Close"].reindex(feat.index)

    # actual_close[t] is the anchor; actual_close[t+1] is the truth
    prev_close = close.shift(1).reindex(test_dates).values   # close at t
    actual_price = close.reindex(test_dates).values          # close at t+1 (truth)
    predicted_price = prev_close * np.exp(pred_returns)

    mask = ~np.isnan(actual_price) & ~np.isnan(predicted_price)
    actual_price = actual_price[mask]
    predicted_price = predicted_price[mask]
    test_dates = test_dates[mask]

    rmse = math.sqrt(mean_squared_error(actual_price, predicted_price))
    mae = mean_absolute_error(actual_price, predicted_price)
    mape = float(np.mean(np.abs((actual_price - predicted_price) / (actual_price + 1e-8))) * 100)

    # Naive baseline: predict tomorrow = today
    baseline_price = prev_close[mask]
    baseline_rmse = math.sqrt(mean_squared_error(actual_price, baseline_price))

    # Residual volatility in RETURN space, used to build the forecast cone
    actual_returns = np.log(actual_price / prev_close[mask])
    resid_std = float(np.std(actual_returns - pred_returns[mask]))

    return {
        "metrics": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "baseline_rmse": round(baseline_rmse, 4),
            "beats_baseline": bool(rmse < baseline_rmse),
        },
        "test": {
            "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test_dates],
            "actual": [round(float(v), 2) for v in actual_price],
            "predicted": [round(float(v), 2) for v in predicted_price],
        },
        "resid_std": resid_std,
    }


# ────────────────────────────────────────────────────────────────────
# 30-day recursive forecast with a widening confidence cone
# ────────────────────────────────────────────────────────────────────
def forecast(model, scalers, df, resid_std, days=30):
    """
    Forecast the next `days` business days.

    This is recursive: each predicted day is fed back in to predict the next.
    Errors compound, so we widen a confidence band as we look further out —
    the band grows with sqrt(time), the standard way uncertainty accumulates
    in a random-walk-like process.
    """
    fscaler, tscaler = scalers["feature"], scalers["target"]

    work = df[["Close", "Volume"]].copy()
    last_volume = float(work["Volume"].iloc[-1])
    prices = []

    for _ in range(days):
        # We only carry Close + Volume forward, so recompute the indicators
        # make_features() needs on this growing synthetic series.
        feat = make_features(_minimal_for_features(work))
        if len(feat) < SEQ_LEN:
            break

        window = fscaler.transform(feat[FEATURES].values)[-SEQ_LEN:]
        x_input = window.reshape(1, SEQ_LEN, len(FEATURES))
        pred_scaled = model.predict(x_input, verbose=0)
        pred_return = float(tscaler.inverse_transform(pred_scaled)[0, 0])

        last_close = float(work["Close"].iloc[-1])
        next_close = last_close * math.exp(pred_return)
        prices.append(next_close)

        next_date = work.index[-1] + pd.tseries.offsets.BDay(1)
        work.loc[next_date] = {"Close": next_close, "Volume": last_volume}

    future_dates = pd.date_range(
        start=df.index[-1] + pd.tseries.offsets.BDay(1),
        periods=len(prices),
        freq="B",
    )

    prices = np.array(prices)
    steps = np.arange(1, len(prices) + 1)
    band = 1.96 * resid_std * np.sqrt(steps)          # 95% cone, widening over time
    lower = prices * np.exp(-band)
    upper = prices * np.exp(band)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "price": [round(float(p), 2) for p in prices],
        "lower": [round(float(p), 2) for p in lower],
        "upper": [round(float(p), 2) for p in upper],
    }


def _minimal_for_features(work):
    """
    During recursive forecasting we only track Close and Volume. make_features
    needs RSI and MACD too, so recompute them on the synthetic close series.
    """
    df = work.copy()
    close = df["Close"]
    df["RSI"] = datalib.compute_rsi(close, 14)
    macd, signal_line, hist = datalib.compute_macd(close)
    df["MACD"] = macd
    df["MACD_signal"] = signal_line
    df["MACD_hist"] = hist
    return df
