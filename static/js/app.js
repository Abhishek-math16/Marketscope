/* ============================================================
   Marketscope — frontend logic
   Talks to the Flask API, then renders everything with Plotly so
   the charts are interactive (zoom, pan, range buttons, hover).
   ============================================================ */

const $ = (sel) => document.querySelector(sel);
let LAST = null;   // cache the latest API response for the timeframe toggle

/* ---- Theme ---------------------------------------------------- */
function currentTheme() {
  return document.documentElement.getAttribute("data-theme");
}
function plotColors() {
  const dark = currentTheme() === "dark";
  return {
    paper: "rgba(0,0,0,0)",
    plot: "rgba(0,0,0,0)",
    font: dark ? "#98a3b3" : "#5b6472",
    grid: dark ? "#283040" : "#e4e9f0",
    up: dark ? "#34d399" : "#15a34a",
    down: dark ? "#f87171" : "#dc2626",
    line: dark ? "#2dd4bf" : "#0f766e",
    actual: dark ? "#60a5fa" : "#2563eb",
    pred: dark ? "#fbbf24" : "#d97706",
  };
}
function baseLayout(extra = {}) {
  const c = plotColors();
  return Object.assign({
    paper_bgcolor: c.paper,
    plot_bgcolor: c.plot,
    font: { family: "Inter, sans-serif", color: c.font, size: 12 },
    margin: { l: 50, r: 20, t: 20, b: 30 },
    xaxis: { gridcolor: c.grid, zeroline: false },
    yaxis: { gridcolor: c.grid, zeroline: false },
    showlegend: true,
    legend: { orientation: "h", y: 1.08, x: 0 },
    hovermode: "x unified",
  }, extra);
}
const CONFIG = { responsive: true, displayModeBar: true, displaylogo: false,
                 modeBarButtonsToRemove: ["lasso2d", "select2d"] };

$("#themeBtn").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  $("#themeBtn").textContent = next === "dark" ? "☀" : "☾";
  if (LAST) renderCharts(LAST);   // recolour existing charts
});

/* ---- Helpers -------------------------------------------------- */
function xy(records) {
  return { x: records.map(r => r.t), y: records.map(r => r.v) };
}

/* ---- Run a prediction ---------------------------------------- */
async function analyze(ticker) {
  ticker = (ticker || $("#tickerInput").value || "").trim();
  if (!ticker) return;
  $("#tickerInput").value = ticker;

  $("#results").hidden = true;
  $("#errorBox").hidden = true;
  $("#loading").hidden = false;
  cycleLoadingMessages();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ticker }),
    });
    const data = await res.json();
    $("#loading").hidden = true;

    if (!res.ok || data.error) {
      showError(data.error || "Could not complete the request.");
      return;
    }
    LAST = data;
    renderSnapshot(data);
    renderSignals(data.signals);
    renderMetrics(data.metrics);
    renderCharts(data);
    $("#downloadLink").href = `/download/${data.csv}`;
    $("#results").hidden = false;
    loadRecent();
    window.scrollTo({ top: $("#results").offsetTop - 60, behavior: "smooth" });
  } catch (err) {
    $("#loading").hidden = true;
    showError("Network error — is the server running? " + err.message);
  }
}

function showError(msg) {
  const box = $("#errorBox");
  box.textContent = "⚠ " + msg;
  box.hidden = false;
}

let loadingTimer = null;
function cycleLoadingMessages() {
  const msgs = [
    "Downloading ten years of price history…",
    "Calculating moving averages and momentum…",
    "Training the model on past data…",
    "Testing it on data it has never seen…",
    "Drawing your interactive charts…",
  ];
  let i = 0;
  $("#loadingMsg").textContent = msgs[0];
  clearInterval(loadingTimer);
  loadingTimer = setInterval(() => {
    i = (i + 1) % msgs.length;
    $("#loadingMsg").textContent = msgs[i];
  }, 4000);
}

/* ---- Snapshot ------------------------------------------------- */
function renderSnapshot(d) {
  $("#snapTicker").textContent = d.ticker;
  $("#snapPrice").textContent = d.last_price.toLocaleString();
  const chg = $("#snapChange");
  const up = d.change_pct >= 0;
  chg.textContent = `${up ? "▲" : "▼"} ${Math.abs(d.change_pct).toFixed(2)}%`;
  chg.className = "snap-change " + (up ? "up" : "down");
  $("#snapMeta").textContent =
    `${d.rows.toLocaleString()} trading days · ${d.start} → ${d.end}` +
    (d.trained_now ? " · model trained just now" : " · using cached model");
}

/* ---- Signals -------------------------------------------------- */
function renderSignals(signals) {
  const grid = $("#signalGrid");
  grid.innerHTML = "";
  signals.forEach(s => {
    const card = document.createElement("div");
    card.className = "signal-card " + s.sentiment;
    card.title = s.explain;
    card.innerHTML = `
      <div class="signal-label">${s.label}</div>
      <div class="signal-value ${s.sentiment}">${s.value}</div>
      <div class="signal-explain">${s.explain}</div>`;
    grid.appendChild(card);
  });
}

/* ---- Metrics -------------------------------------------------- */
function renderMetrics(m) {
  const grid = $("#metricGrid");
  grid.innerHTML = "";
  const cards = [
    ["RMSE", m.rmse, "Typical error size, in price units"],
    ["MAE", m.mae, "Average error, in price units"],
    ["MAPE", m.mape + "%", "Average error as a percentage"],
    ["Baseline RMSE", m.baseline_rmse, "Error of a 'tomorrow = today' guess"],
  ];
  cards.forEach(([label, val, sub]) => {
    const c = document.createElement("div");
    c.className = "metric-card";
    c.title = sub;
    c.innerHTML = `<div class="metric-label">${label}</div><div class="metric-value">${val}</div>`;
    grid.appendChild(c);
  });

  const note = $("#baselineNote");
  if (m.beats_baseline) {
    note.className = "baseline-note good";
    note.textContent = "✓ The model beats the naive baseline — its predictions are, on average, closer to reality than simply assuming tomorrow equals today.";
  } else {
    note.className = "baseline-note bad";
    note.textContent = "This run did not beat the naive 'tomorrow = today' baseline. That is common and honest for daily stock prediction — short-term prices are close to a random walk. Worth discussing as a finding rather than hiding.";
  }
}

/* ---- Charts --------------------------------------------------- */
function renderCharts(d) {
  drawPrice(d, document.querySelector(".tf.active")?.dataset.tf || "daily");
  drawRSI(d);
  drawMACD(d);
  drawPrediction(d);
  drawForecast(d);
}

function rangeButtons() {
  return {
    buttons: [
      { count: 1, label: "1M", step: "month", stepmode: "backward" },
      { count: 6, label: "6M", step: "month", stepmode: "backward" },
      { count: 1, label: "1Y", step: "year", stepmode: "backward" },
      { count: 5, label: "5Y", step: "year", stepmode: "backward" },
      { step: "all", label: "All" },
    ],
  };
}

function drawPrice(d, tf) {
  const c = plotColors();
  const candles = d.ohlc[tf];
  const traces = [{
    type: "candlestick",
    x: candles.map(r => r.t),
    open: candles.map(r => r.o),
    high: candles.map(r => r.h),
    low: candles.map(r => r.l),
    close: candles.map(r => r.c),
    name: "Price",
    increasing: { line: { color: c.up } },
    decreasing: { line: { color: c.down } },
  }];

  // EMA overlays only make sense on the daily series
  if (tf === "daily") {
    const emas = [["ema20", "EMA 20", c.line], ["ema50", "EMA 50", c.pred],
                  ["ema100", "EMA 100", c.actual], ["ema200", "EMA 200", c.down]];
    emas.forEach(([key, name, color]) => {
      const s = xy(d.ema[key]);
      traces.push({ type: "scatter", mode: "lines", x: s.x, y: s.y, name,
                    line: { width: 1.4, color }, opacity: .9 });
    });
  }

  const layout = baseLayout({
    xaxis: {
      gridcolor: c.grid, rangeslider: { visible: true, thickness: 0.06 },
      rangeselector: rangeButtons(),
    },
    yaxis: { gridcolor: c.grid, title: "Price" },
  });
  Plotly.react("priceChart", traces, layout, CONFIG);
}

function drawRSI(d) {
  const c = plotColors();
  const s = xy(d.rsi);
  const traces = [{ type: "scatter", mode: "lines", x: s.x, y: s.y,
                    name: "RSI", line: { color: c.line, width: 1.4 } }];
  const layout = baseLayout({
    showlegend: false,
    yaxis: { gridcolor: c.grid, range: [0, 100] },
    shapes: [
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: 70, y1: 70,
        line: { color: c.down, width: 1, dash: "dot" } },
      { type: "line", xref: "paper", x0: 0, x1: 1, y0: 30, y1: 30,
        line: { color: c.up, width: 1, dash: "dot" } },
    ],
  });
  Plotly.react("rsiChart", traces, layout, CONFIG);
}

function drawMACD(d) {
  const c = plotColors();
  const macd = xy(d.macd.macd), sig = xy(d.macd.signal), hist = xy(d.macd.hist);
  const traces = [
    { type: "bar", x: hist.x, y: hist.y, name: "Histogram",
      marker: { color: hist.y.map(v => v >= 0 ? c.up : c.down) }, opacity: .55 },
    { type: "scatter", mode: "lines", x: macd.x, y: macd.y, name: "MACD",
      line: { color: c.line, width: 1.4 } },
    { type: "scatter", mode: "lines", x: sig.x, y: sig.y, name: "Signal",
      line: { color: c.pred, width: 1.4 } },
  ];
  Plotly.react("macdChart", traces, baseLayout(), CONFIG);
}

function drawPrediction(d) {
  const c = plotColors();
  const p = d.prediction;
  const traces = [
    { type: "scatter", mode: "lines", x: p.dates, y: p.actual, name: "Actual",
      line: { color: c.actual, width: 1.6 } },
    { type: "scatter", mode: "lines", x: p.dates, y: p.predicted, name: "Predicted (test set)",
      line: { color: c.pred, width: 1.6 } },
  ];
  const layout = baseLayout({
    yaxis: { gridcolor: c.grid, title: "Price" },
    xaxis: { gridcolor: c.grid, rangeslider: { visible: true, thickness: 0.06 } },
  });
  Plotly.react("predChart", traces, layout, CONFIG);
}

function drawForecast(d) {
  const c = plotColors();
  const f = d.forecast;
  // recent actual tail (last ~120 daily closes) for context
  const tail = d.ohlc.daily.slice(-120);
  const traces = [
    { type: "scatter", mode: "lines", x: tail.map(r => r.t), y: tail.map(r => r.c),
      name: "Recent actual", line: { color: c.actual, width: 1.6 } },
    // confidence cone (upper then lower with fill)
    { type: "scatter", mode: "lines", x: f.dates, y: f.upper, name: "Upper bound",
      line: { width: 0 }, showlegend: false },
    { type: "scatter", mode: "lines", x: f.dates, y: f.lower, name: "Uncertainty range",
      fill: "tonexty", fillcolor: "rgba(15,118,110,.15)", line: { width: 0 } },
    { type: "scatter", mode: "lines", x: f.dates, y: f.price, name: "Forecast",
      line: { color: c.line, width: 2, dash: "dash" } },
  ];
  Plotly.react("forecastChart", traces, baseLayout({
    yaxis: { gridcolor: c.grid, title: "Price" },
  }), CONFIG);
}

/* ---- Timeframe toggle ---------------------------------------- */
document.querySelectorAll(".tf").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tf").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    if (LAST) drawPrice(LAST, btn.dataset.tf);
  });
});

/* ---- Recently viewed ----------------------------------------- */
async function loadRecent() {
  try {
    const res = await fetch("/api/recent");
    const { tickers } = await res.json();
    if (!tickers || !tickers.length) return;
    $("#recentWrap").hidden = false;
    const wrap = $("#recentChips");
    wrap.innerHTML = "";
    tickers.forEach(t => {
      const b = document.createElement("button");
      b.className = "chip";
      b.textContent = t;
      b.addEventListener("click", () => analyze(t));
      wrap.appendChild(b);
    });
  } catch (_) { /* ignore */ }
}

/* ---- Glossary ------------------------------------------------- */
const GLOSSARY = {
  ema: ["EMA (Exponential Moving Average)",
    "A running average of price that gives more weight to recent days, so it reacts faster than a plain average. Shorter EMAs (20, 50) track recent moves; longer ones (100, 200) show the big-picture trend. When a short EMA crosses above a long one, traders read it as a strengthening uptrend."],
  rsi: ["RSI (Relative Strength Index)",
    "A momentum gauge from 0 to 100. Above 70 suggests a stock has climbed fast and might be 'overbought' (due for a rest); below 30 suggests it has dropped fast and might be 'oversold' (due for a bounce). It is a hint, not a guarantee."],
  macd: ["MACD (Moving Average Convergence Divergence)",
    "Compares a fast and a slow moving average to measure momentum. When the MACD line rises above its signal line (histogram bars turn positive), momentum is building upward; the reverse signals downward momentum."],
  metrics: ["RMSE, MAE, MAPE & baseline",
    "These measure how wrong the model's predictions were on data it never trained on. RMSE and MAE are average error sizes in price units (lower is better); MAPE is that error as a percentage. The baseline is a dumb guess that tomorrow equals today — a good model should beat it."],
  forecast: ["30-day forecast & the uncertainty cone",
    "The model predicts one day ahead, then feeds that guess back in to predict the next, and so on for 30 days. Because each step builds on the last, errors pile up — so the shaded cone widens the further out you look. Treat the far end as a rough scenario, not a precise number."],
};
function buildGlossary() {
  const body = $("#glossaryBody");
  body.innerHTML = "";
  Object.values(GLOSSARY).forEach(([term, def]) => {
    const div = document.createElement("div");
    div.className = "gloss-term";
    div.innerHTML = `<h3>${term}</h3><p>${def}</p>`;
    body.appendChild(div);
  });
}
function openGlossary(term) {
  buildGlossary();
  $("#glossary").hidden = false;
  if (term && GLOSSARY[term]) {
    const idx = Object.keys(GLOSSARY).indexOf(term);
    $("#glossaryBody").children[idx]?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}
$("#glossaryBtn").addEventListener("click", () => openGlossary());
$("#glossaryClose").addEventListener("click", () => ($("#glossary").hidden = true));
$("#glossary").addEventListener("click", (e) => {
  if (e.target.id === "glossary") $("#glossary").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("#glossary").hidden = true;
});
document.addEventListener("click", (e) => {
  if (e.target.classList.contains("info")) openGlossary(e.target.dataset.term);
});

/* ---- Wire up the search -------------------------------------- */
$("#goBtn").addEventListener("click", () => analyze());
$("#tickerInput").addEventListener("keydown", (e) => { if (e.key === "Enter") analyze(); });
document.querySelectorAll(".seed").forEach(b =>
  b.addEventListener("click", () => analyze(b.dataset.t)));

loadRecent();
