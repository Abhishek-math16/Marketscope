# Marketscope — Stock Trend Explorer

An educational web app that downloads ten years of stock history, draws fully
interactive charts, trains an LSTM model to predict price movement, evaluates it
honestly against a baseline, and explains every finance term in plain English —
so a complete beginner can understand what they're looking at.

> **Not financial advice.** This is a learning project. Real markets move on news
> and events no price-only model can see.

---

## What's new vs. the original version

| Area | Before | Now |
|---|---|---|
| Charts | Static PNG images | **Interactive Plotly** — zoom, pan, hover, 1M/6M/1Y/5Y buttons, range slider |
| Timeframes | Daily only | **Daily / Weekly / Monthly** candlestick toggle |
| Model target | Raw price (learns to echo yesterday) | **Predicts returns** + multi-feature input (RSI, MACD, volume) |
| Honesty | "Looks accurate" | **Naive baseline comparison** + real uncertainty cone |
| Speed | Retrains every click (30–90s) | **Saves & reuses models** — instant on repeat tickers |
| Confidence band | Fake ±4% | **Statistical cone** that widens with time |
| Data store | None | **SQLite** — search history, metrics, model cache |
| Beginner help | Term names only | **Plain-English signals, glossary, info buttons** |
| Structure | One big file | **Modular**: data / model / database / app |

---

## Project structure

```
stock-app/
├── app.py            # Flask routes + plain-English signal logic
├── data.py           # download prices, compute EMA/RSI/MACD, resample
├── model.py          # LSTM: returns-based prediction, save/load, forecast
├── database.py       # SQLite: searches, predictions, model cache
├── templates/
│   └── index.html    # single-page dashboard
├── static/
│   ├── css/style.css # theming (light/dark), layout
│   └── js/app.js     # AJAX + Plotly interactive charts
├── models/           # saved models appear here at runtime
├── datasets/         # CSV exports appear here
├── data/             # SQLite database lives here
├── requirements.txt
├── Procfile          # for Render / Railway / Heroku-style hosts
└── runtime.txt       # pins the Python version for hosts
```

---

## 1. Run it locally (step by step)

You need **Python 3.10 or 3.11** installed.

```bash
# 1. Open a terminal in the project folder
cd stock-app

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install the dependencies (TensorFlow is large — give it a few minutes)
pip install -r requirements.txt

# 4. Start the app
python app.py
```

Open **http://127.0.0.1:5000** in your browser and type a ticker (e.g. `AAPL`,
`TCS.NS`). The **first** lookup of a ticker trains a model (30–90s); after that
it's cached and loads in a second or two.

---

## 2. Make it reachable from any device (quick demo)

**ngrok** gives your local app a public URL in seconds — ideal for showing a
guide or examiner. It disappears when you close it.

```bash
# install from https://ngrok.com/download, then with app.py running:
ngrok http 5000
```

Share the `https://....ngrok-free.app` link it prints.

---

## 3. Deploy it permanently on the web

A few honest realities first:

- **TensorFlow is heavy.** Free hosting tiers often have ~512 MB RAM. Training a
  model inside a web request can be slow or run out of memory there.
- **The robust pattern** is to *pre-train models locally for the tickers you'll
  demo, commit them, and let the server only load them.* See section 4.

### Option A — Render (recommended, persistent free URL)

1. Push this folder to a **GitHub** repo.
2. On https://render.com → **New → Web Service** → connect the repo.
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --timeout 300 --workers 1`
4. Deploy. You get a permanent `https://your-app.onrender.com` URL.

> The included `Procfile` already contains the start command, so most hosts
> auto-detect it. The long `--timeout 300` matters because training takes time.

### Option B — Railway

Connect the GitHub repo at https://railway.app; it reads the `Procfile`
automatically. Similar free-tier RAM caveats apply.

### Option C — Hugging Face Spaces (best for heavy ML)

More generous resources for ML demos. Create a Space, upload the files, and run
either Flask via a Docker Space or a small Gradio/Streamlit wrapper.

> Free-tier limits on all of these change often — check the current plan before
> you rely on it for a submission deadline.

**Before deploying anywhere**, set debug off in `app.py`:
```python
app.run(debug=False, host="0.0.0.0", port=5000)
```
(Gunicorn ignores `app.run`, so this only affects local runs — but it's good hygiene.)

---

## 4. Pre-train models so the public site is fast (optional but recommended)

Run this once locally for the tickers you'll show. It trains and saves each
model into `models/`, which you then commit (remove those lines from
`.gitignore` first).

```python
# pretrain.py
import data as d, model as m, database as db
db.init_db()
for t in ["AAPL", "MSFT", "RELIANCE.NS", "TCS.NS"]:
    print("training", t)
    raw = d.fetch_data(t); df = d.add_indicators(raw)
    model, scalers, _ = m.train(df)
    m.save(t, model, scalers); db.mark_model_trained(t)
print("done")
```

Now the deployed site loads those instantly instead of training on request.

---

## 5. How the model works (for your report / viva)

**The key idea:** the model predicts the *next day's return* (percentage change),
not the next day's raw price. If you train on raw prices, the model learns the
lazy trick of repeating today's price — the chart looks perfect but it has
learned nothing. Predicting returns removes that shortcut.

- **Inputs (per day, 60-day window):** log return, RSI, volatility-normalised
  MACD histogram, and volume change.
- **Network:** two stacked LSTM layers (64 units) with dropout, then dense
  layers to one output. Trained with early stopping on a validation split.
- **Evaluation:** predictions are converted back to prices using the actual
  previous close, then compared to the truth with RMSE / MAE / MAPE — and
  against a **naive baseline** (tomorrow = today). The app states clearly whether
  the model beats that baseline.
- **Forecast:** 30 days, recursive (each day feeds the next). Because errors
  compound, the shaded uncertainty cone widens with the square root of time.

**A mature point to make:** daily stock prices behave close to a random walk, so
beating the naive baseline by a lot is genuinely hard. Reporting that honestly —
with the baseline shown — is stronger science than a chart that merely *looks*
accurate.

---

## Indicators in plain English

- **EMA** — a fast-reacting average of price; crossovers hint at trend changes.
- **RSI** — momentum 0–100; >70 overbought, <30 oversold.
- **MACD** — momentum from two averages; above zero = upward push.
- **RMSE / MAE / MAPE** — how wrong the predictions were (lower is better).

---

## Ideas to take it further

- Add a **backtest**: simulate buying on EMA golden crosses and report returns.
- Add **news-sentiment** as an extra feature (a real differentiator).
- **Compare multiple tickers** side by side (portfolio view).
- Cache the JSON response so repeat loads skip recomputation entirely.
