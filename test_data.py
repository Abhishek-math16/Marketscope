"""
test_data.py — run this FIRST to check that data fetching works.

    python test_data.py

If you see a table of prices, your data layer is fine and you can run the app.
If you see "FAILED", the problem is Yahoo/yfinance on your machine/network,
not the app — the script prints what to try next.
"""

import data as d

TICKERS = ["AAPL", "RELIANCE.NS", "SUZLON.NS"]

print("Testing data fetch (this routes through a browser-impersonating "
      "session to get past Yahoo's blocking)...\n")

any_ok = False
for t in TICKERS:
    df = d.fetch_data(t, period="1y")   # short period = fast test
    if df is not None and not df.empty:
        any_ok = True
        last = df["Close"].iloc[-1]
        print(f"  OK   {t:<14} {len(df):>4} rows, last close = {last:.2f}")
    else:
        print(f"  FAIL {t:<14} no data returned")

print()
if any_ok:
    print("Data layer works. You can now run:  python app.py")
else:
    print("All fetches failed. Try, in order:")
    print("  1.  pip install --upgrade yfinance curl_cffi")
    print("  2.  Wait 2–3 minutes (Yahoo may be rate-limiting your IP) and retry")
    print("  3.  Try a different network / disable VPN")
    print("  4.  Delete the yfinance cache:")
    print("        Windows: %LOCALAPPDATA%\\py-yfinance")
    print("        Mac/Linux: ~/.cache/py-yfinance")
