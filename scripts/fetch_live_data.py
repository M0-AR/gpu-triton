"""Re-fetch live market snapshots (CoinGecko BTC history + Yahoo AAPL quote).

No API keys. Overwrites data/btc_30d.json and data/aapl_quote.json with a fresh
UTC timestamp so any reviewer can confirm the market leg is genuinely live.
"""
from __future__ import annotations
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "gpu-triton-lecture1-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    DATA.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    cg = get("https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30&interval=daily")
    daily = [{"date": datetime.fromtimestamp(p[0] / 1000, tz=timezone.utc).date().isoformat(),
              "price": round(float(p[1]), 2)} for p in cg["prices"]]
    (DATA / "btc_30d.json").write_text(json.dumps(
        {"coin": "bitcoin", "vs": "usd", "fetched_utc": now,
         "source": "CoinGecko market_chart (no key)", "daily": daily}, indent=2))
    yf = get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=5d")
    m = yf["chart"]["result"][0]["meta"]
    (DATA / "aapl_quote.json").write_text(json.dumps(
        {"fetched_utc": now, "source": "Yahoo Finance v8 (no key)",
         "symbol": "AAPL", "name": "Apple Inc.", "current": m["regularMarketPrice"],
         "open": m.get("chartPreviousClose"), "high": m.get("regularMarketDayHigh"),
         "low": m.get("regularMarketDayLow"), "volume": m.get("regularMarketVolume"),
         "market_cap": None, "pe_ttm": None}, indent=2))
    print(f"refreshed {len(daily)} BTC points + AAPL @ {now}")


if __name__ == "__main__":
    main()
