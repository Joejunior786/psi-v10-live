import os
import asyncio

import httpx
from fastapi import FastAPI, Query
from fastmcp import FastMCP

app = FastAPI(
    title="Psi V10 Live Scanner",
    version="1.1.0",
    description="Read-only Binance Spot market scanner for the Psi-V10 workflow."
)

BASE = os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
CONCURRENCY = int(os.getenv("CONCURRENCY", "12"))
MIN_QUOTE_VOLUME = float(os.getenv("MIN_QUOTE_VOLUME", "500000"))
SEM = asyncio.Semaphore(CONCURRENCY)


async def api_get(client: httpx.AsyncClient, path: str, params=None):
    async with SEM:
        response = await client.get(BASE + path, params=params, timeout=20)
        response.raise_for_status()
        return response.json()


def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def atr(rows, period=14):
    if len(rows) < period + 1:
        return None
    ranges = []
    for i in range(1, len(rows)):
        high = float(rows[i][2])
        low = float(rows[i][3])
        prev_close = float(rows[i - 1][4])
        ranges.append(max(high-low, abs(high-prev_close), abs(low-prev_close)))
    return sum(ranges[-period:]) / period


async def analyse_symbol(client: httpx.AsyncClient, symbol: str):
    rows = await api_get(
        client, "/api/v3/klines",
        {"symbol": symbol, "interval": "4h", "limit": 220}
    )
    closed = rows[:-1]
    closes = [float(row[4]) for row in closed]
    ma200 = sma(closes, 200)
    ma50 = sma(closes, 50)

    if ma200 is None:
        return None

    price = float(rows[-1][4])
    atr14 = atr(closed, 14)
    distance_pct = (price - ma200) / ma200 * 100.0

    volumes = [float(row[5]) for row in closed]
    avg20_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None
    volume_acceleration = (
        volumes[-1] / avg20_volume if avg20_volume and avg20_volume > 0 else None
    )

    last = closed[-1]
    total_volume = float(last[5])
    taker_buy_volume = float(last[9])

    return {
        "symbol": symbol,
        "price": price,
        "sma200_4h": ma200,
        "sma50_4h": ma50,
        "distance_pct": distance_pct,
        "abs_distance_pct": abs(distance_pct),
        "side": "ABOVE" if distance_pct >= 0 else "BELOW",
        "atr14_4h": atr14,
        "distance_atr": ((price - ma200) / atr14) if atr14 else None,
        "last_closed_volume_vs_20avg": volume_acceleration,
        "last_closed_taker_buy_ratio": (
            taker_buy_volume / total_volume if total_volume else None
        ),
        "last_closed_trade_count": int(last[8]),
    }


def classify(row):
    distance = row["abs_distance_pct"]
    if distance <= 0.5:
        row["ma_state"] = "TOUCH"
        row["ma_layer"] = "PASS_4H_SMA200"
    elif distance <= 1.0:
        row["ma_state"] = "ALMOST"
        row["ma_layer"] = "WATCH"
    else:
        row["ma_state"] = "APPROACHING"
        row["ma_layer"] = "WATCH"

    # MA proximity alone is not a complete V10 BUY signal.
    row["final_v10_state"] = "MA_CANDIDATE"
    return row


async def run_scan(limit: int = 10, max_distance: float = 2.0):
    async with httpx.AsyncClient(
        headers={"User-Agent": "psi-v10-live-scanner/1.1"}
    ) as client:
        exchange_info, tickers = await asyncio.gather(
            api_get(client, "/api/v3/exchangeInfo"),
            api_get(client, "/api/v3/ticker/24hr"),
        )

        ticker_map = {row["symbol"]: row for row in tickers}
        universe = []

        for market in exchange_info["symbols"]:
            symbol = market["symbol"]
            if market.get("status") != "TRADING":
                continue
            if market.get("quoteAsset") != "USDT":
                continue
            if not market.get("isSpotTradingAllowed", False):
                continue

            quote_volume = float(
                ticker_map.get(symbol, {}).get("quoteVolume", 0) or 0
            )
            if quote_volume >= MIN_QUOTE_VOLUME:
                universe.append((symbol, quote_volume))

        universe.sort(key=lambda item: item[1], reverse=True)

        raw = await asyncio.gather(
            *(analyse_symbol(client, symbol) for symbol, _ in universe),
            return_exceptions=True,
        )

    results = [
        classify(row) for row in raw
        if isinstance(row, dict)
        and row["abs_distance_pct"] <= max_distance
    ]
    results.sort(key=lambda row: row["abs_distance_pct"])

    return {
        "source": "Binance public Spot API",
        "timeframe": "4h",
        "indicator": "SMA200",
        "sma_definition": "200 completed 4-hour candles",
        "minimum_24h_quote_volume_usdt": MIN_QUOTE_VOLUME,
        "max_distance_pct": max_distance,
        "returned": min(limit, len(results)),
        "results": results[:limit],
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "psi-v10-live-scanner",
        "mcp": "/mcp",
    }


@app.get("/scan")
async def scan(
    limit: int = Query(10, ge=1, le=50),
    max_distance: float = Query(2.0, ge=0.05, le=20.0),
):
    return await run_scan(limit, max_distance)


@app.get("/")
async def root():
    return {
        "service": "Psi V10 Live Scanner",
        "health": "/health",
        "scan": "/scan?limit=10&max_distance=2",
        "mcp": "/mcp",
    }


mcp = FastMCP("Psi V10 Live Scanner")


@mcp.tool()
async def scanner_health() -> dict:
    """Check scanner connectivity and identify the live data source."""
    return {
        "ok": True,
        "service": "psi-v10-live-scanner",
        "source": "Binance public Spot API",
        "mcp": True,
    }


@mcp.tool()
async def scan_top(limit: int = 10, max_distance: float = 2.0) -> dict:
    """Return Binance USDT spot pairs nearest the 4H SMA200.

    This is the MA candidate layer and not, by itself, a complete V10 BUY signal.
    """
    limit = max(1, min(int(limit), 50))
    max_distance = max(0.05, min(float(max_distance), 20.0))
    return await run_scan(limit, max_distance)


@mcp.tool()
async def symbol_detail(symbol: str) -> dict:
    """Return current 4H SMA200/SMA50 telemetry for one Binance spot symbol."""
    symbol = symbol.upper().strip()
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    async with httpx.AsyncClient(
        headers={"User-Agent": "psi-v10-live-scanner/1.1"}
    ) as client:
        try:
            row = await analyse_symbol(client, symbol)
        except httpx.HTTPStatusError as exc:
            return {
                "ok": False,
                "symbol": symbol,
                "error": f"Binance HTTP {exc.response.status_code}",
            }

    if row is None:
        return {
            "ok": False,
            "symbol": symbol,
            "error": "Insufficient data or unsupported symbol",
        }

    return classify(row)


# Streamable HTTP MCP transport, available publicly at /mcp.
mcp_app = mcp.http_app(path="/")
app.mount("/mcp", mcp_app)
