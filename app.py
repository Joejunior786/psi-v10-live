import asyncio
import json
import math
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

BINANCE_REST = os.getenv("BINANCE_REST", "https://data-api.binance.vision")
BINANCE_WS = os.getenv("BINANCE_WS", "wss://data-stream.binance.vision/ws")
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "150"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "40"))
LOOKBACK_SECONDS = int(os.getenv("LOOKBACK_SECONDS", "60"))
NEAR_BOOK_USD = float(os.getenv("NEAR_BOOK_USD", "25000"))
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "2.0"))
VOL_ACCEL_THRESHOLD = float(os.getenv("VOL_ACCEL_THRESHOLD", "1.5"))
BUY_IMBALANCE_THRESHOLD = float(os.getenv("BUY_IMBALANCE_THRESHOLD", "0.20"))
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"

states: dict[str, dict[str, Any]] = {}
paper_signals = deque(maxlen=500)
tasks = []
http_session = None
started_at = time.time()


class CommandBody(BaseModel):
    command: str


def now_ms():
    return int(time.time() * 1000)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def ensure_state(symbol):
    if symbol not in states:
        states[symbol] = {
            "symbol": symbol,
            "trades": deque(maxlen=3000),
            "bids": [],
            "asks": [],
            "price": 0.0,
            "updated_ms": 0,
        }
    return states[symbol]


async def get_json(path):
    async with http_session.get(BINANCE_REST + path, timeout=20) as r:
        r.raise_for_status()
        return await r.json()


async def discover_symbols():
    info, tickers = await asyncio.gather(
        get_json("/api/v3/exchangeInfo"),
        get_json("/api/v3/ticker/24hr"),
    )
    volume = {x["symbol"]: safe_float(x.get("quoteVolume")) for x in tickers}
    found = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("isSpotTradingAllowed") is False:
            continue
        found.append((s["symbol"], volume.get(s["symbol"], 0.0)))
    found.sort(key=lambda x: x[1], reverse=True)
    return [symbol.lower() for symbol, _ in found[:MAX_SYMBOLS]]


def trim_trades(st):
    cutoff = now_ms() - LOOKBACK_SECONDS * 1000
    while st["trades"] and st["trades"][0]["ts"] < cutoff:
        st["trades"].popleft()


def calc(st):
    trim_trades(st)
    trades = list(st["trades"])
    bids = st["bids"]
    asks = st["asks"]
    if not trades or not bids or not asks:
        return None

    buy_notional = sum(t["notional"] for t in trades if t["side"] == "buy")
    sell_notional = sum(t["notional"] for t in trades if t["side"] == "sell")
    total = buy_notional + sell_notional
    buy_imbalance = (buy_notional - sell_notional) / total if total else 0.0

    near_ask = min(NEAR_BOOK_USD, sum(p * q for p, q in asks))
    near_bid = min(NEAR_BOOK_USD, sum(p * q for p, q in bids))
    psi = buy_notional / max(near_ask, 1.0)

    book_total = near_bid + near_ask
    book_imbalance = (near_bid - near_ask) / book_total if book_total else 0.0

    mid = len(trades) // 2
    first = sum(t["notional"] for t in trades[:mid]) if mid else 0.0
    second = sum(t["notional"] for t in trades[mid:]) if mid else 0.0
    volume_acceleration = second / max(first, 1.0)

    prices = [t["price"] for t in trades]
    price_change = (prices[-1] / prices[0] - 1) * 100 if prices[0] else 0.0

    bid_top = bids[0][0]
    ask_top = asks[0][0]
    spread_pct = ((ask_top - bid_top) / bid_top * 100) if bid_top else 0.0

    score = (
        math.log1p(max(psi, 0))
        + 1.5 * max(buy_imbalance, 0)
        + 0.75 * math.log1p(max(volume_acceleration, 0))
        + 0.5 * max(book_imbalance, 0)
    )

    signal = (
        psi >= PSI_THRESHOLD
        and volume_acceleration >= VOL_ACCEL_THRESHOLD
        and buy_imbalance > BUY_IMBALANCE_THRESHOLD
        and len(trades) >= 20
    )

    return {
        "symbol": st["symbol"].upper(),
        "price": round(st["price"], 10),
        "psi": round(psi, 4),
        "buy_imbalance": round(buy_imbalance, 4),
        "book_imbalance": round(book_imbalance, 4),
        "volume_acceleration": round(volume_acceleration, 4),
        "price_change_1m_pct": round(price_change, 4),
        "near_ask_usd": round(near_ask, 2),
        "near_bid_usd": round(near_bid, 2),
        "spread_pct": round(spread_pct, 5),
        "buy_notional_usd": round(buy_notional, 2),
        "sell_notional_usd": round(sell_notional, 2),
        "trade_count": len(trades),
        "score": round(score, 4),
        "signal": signal,
        "signal_reason": (
            "aggressive buying + accelerating flow + thin near-ask liquidity"
            if signal else ""
        ),
        "updated_ms": st["updated_ms"],
    }


def v10_decision(st, base):
    trades = list(st["trades"])
    if len(trades) < 3:
        return {**base, "v10_confluence_pct": 0.0, "v10_state": "FLAT"}

    prices = [t["price"] for t in trades]
    buy_cvd = sum(t["notional"] for t in trades if t["side"] == "buy")
    sell_cvd = sum(t["notional"] for t in trades if t["side"] == "sell")
    total_notional = buy_cvd + sell_cvd

    vwap = (
        sum(t["price"] * t["notional"] for t in trades) / total_notional
        if total_notional else prices[-1]
    )
    mean_price = sum(prices) / len(prices)
    variance = sum((p - mean_price) ** 2 for p in prices) / max(len(prices) - 1, 1)
    std_price = math.sqrt(variance)
    zscore = (prices[-1] - vwap) / std_price if std_price > 0 else 0.0

    cvd_usd = buy_cvd - sell_cvd
    cvd_ratio = cvd_usd / max(total_notional, 1.0)

    recent_high = max(prices)
    recent_low = min(prices)
    range_pct = ((recent_high - recent_low) / recent_low * 100) if recent_low else 0.0
    breakout_proximity_pct = (
        max(0.0, (recent_high - prices[-1]) / recent_high * 100)
        if recent_high else 100.0
    )

    book_pressure = base["book_imbalance"]
    ofi_proxy = 0.55 * base["buy_imbalance"] + 0.45 * book_pressure

    spread_ok = 0 <= base["spread_pct"] <= 0.15
    liquidity_ok = min(base["near_bid_usd"], base["near_ask_usd"]) >= 5000
    flow_ok = base["buy_imbalance"] >= 0.20 and cvd_ratio >= 0.20
    book_ok = book_pressure >= 0.10
    accel_ok = base["volume_acceleration"] >= 1.50
    psi_ok = base["psi"] >= 1.50
    breakout_ok = breakout_proximity_pct <= 0.35 and prices[-1] >= vwap
    anti_chase_ok = base["price_change_1m_pct"] <= 2.0 and zscore <= 2.5
    activity_ok = base["trade_count"] >= 20

    factors = {
        "aggressive_flow": flow_ok,
        "order_book_pressure": book_ok,
        "volume_acceleration": accel_ok,
        "psi_liquidity_pressure": psi_ok,
        "breakout_proximity": breakout_ok,
        "spread_quality": spread_ok,
        "liquidity_quality": liquidity_ok,
        "anti_chase": anti_chase_ok,
        "trade_activity": activity_ok,
    }

    passed = sum(1 for ok in factors.values() if ok)
    confluence = 100.0 * passed / len(factors)

    if not anti_chase_ok or not spread_ok or not liquidity_ok:
        state = "FLAT"
    elif confluence >= 85 and flow_ok and breakout_ok and activity_ok:
        state = "BUY NOW"
    elif confluence >= 75:
        state = "ARMED"
    elif confluence >= 60:
        state = "WATCH"
    elif confluence >= 45:
        state = "PRE-IGNITION"
    else:
        state = "FLAT"

    return {
        **base,
        "vwap_1m": round(vwap, 10),
        "vwap_zscore": round(zscore, 4),
        "cvd_usd_1m": round(cvd_usd, 2),
        "cvd_ratio": round(cvd_ratio, 4),
        "ofi_proxy": round(ofi_proxy, 4),
        "range_1m_pct": round(range_pct, 4),
        "breakout_proximity_pct": round(breakout_proximity_pct, 4),
        "anti_chase_pass": anti_chase_ok,
        "v10_factors": factors,
        "v10_factors_passed": passed,
        "v10_factors_total": len(factors),
        "v10_confluence_pct": round(confluence, 1),
        "v10_state": state,
        "derivatives_confirmation": "NOT_AVAILABLE_IN_SPOT_FEED",
    }


def snapshot(limit=20):
    rows = []
    for st in states.values():
        row = calc(st)
        if row:
            rows.append(v10_decision(st, row))
    state_rank = {"BUY NOW": 5, "ARMED": 4, "WATCH": 3, "PRE-IGNITION": 2, "FLAT": 1}
    rows.sort(
        key=lambda x: (
            state_rank.get(x["v10_state"], 0),
            x["v10_confluence_pct"],
            x["score"],
        ),
        reverse=True,
    )
    return rows[:max(1, min(limit, 200))]


def deep_dive(symbol):
    st = states.get(symbol.lower())
    if not st:
        raise HTTPException(404, f"{symbol.upper()} is not currently tracked")
    row = calc(st)
    if row:
        row = v10_decision(st, row)
    if not row:
        raise HTTPException(503, f"Insufficient live data for {symbol.upper()}")
    recent = list(st["trades"])[-20:]
    buys = sum(1 for t in recent if t["side"] == "buy")
    return {
        "metrics": row,
        "recent_20_trade_buy_count": buys,
        "recent_20_trade_sell_count": len(recent) - buys,
        "top_bid": st["bids"][0] if st["bids"] else None,
        "top_ask": st["asks"][0] if st["asks"] else None,
        "top_5_bids": st["bids"][:5],
        "top_5_asks": st["asks"][:5],
    }


async def trade_worker(symbols):
    streams = [f"{s}@aggTrade" for s in symbols]
    while True:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(BINANCE_WS, heartbeat=20, receive_timeout=60) as ws:
                    await ws.send_json({
                        "method": "SUBSCRIBE",
                        "params": streams,
                        "id": int(time.time()),
                    })
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        if "result" in data:
                            continue
                        payload = data.get("data", data)
                        if payload.get("e") != "aggTrade":
                            continue
                        symbol = payload.get("s", "").lower()
                        if not symbol:
                            continue
                        st = ensure_state(symbol)
                        price = safe_float(payload.get("p"))
                        qty = safe_float(payload.get("q"))
                        side = "sell" if payload.get("m") else "buy"
                        st["price"] = price
                        st["updated_ms"] = now_ms()
                        st["trades"].append({
                            "ts": safe_float(payload.get("T"), now_ms()),
                            "price": price,
                            "qty": qty,
                            "notional": price * qty,
                            "side": side,
                        })
                        row = calc(st)
                        if row:
                            row = v10_decision(st, row)
                        if row and row["v10_state"] == "BUY NOW":
                            paper_signals.appendleft({"timestamp_ms": now_ms(), **row})
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(3)


async def depth_worker(symbols):
    streams = [f"{s}@depth20@100ms" for s in symbols]
    combined_url = BINANCE_WS.replace(
        "/ws",
        "/stream?streams=" + "/".join(streams),
    )
    while True:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(
                    combined_url, heartbeat=20, receive_timeout=60
                ) as ws:
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        stream = data.get("stream", "")
                        payload = data.get("data", {})
                        if "@depth" not in stream:
                            continue
                        symbol = stream.split("@", 1)[0].lower()
                        if not symbol:
                            continue
                        st = ensure_state(symbol)
                        bids = payload.get("bids", payload.get("b", []))
                        asks = payload.get("asks", payload.get("a", []))
                        st["bids"] = [
                            (safe_float(p), safe_float(q)) for p, q in bids
                        ]
                        st["asks"] = [
                            (safe_float(p), safe_float(q)) for p, q in asks
                        ]
                        st["updated_ms"] = now_ms()
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(3)


async def start_streams():
    global http_session
    http_session = aiohttp.ClientSession()
    symbols = await discover_symbols()
    for s in symbols:
        ensure_state(s)
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        tasks.append(asyncio.create_task(trade_worker(batch)))
        tasks.append(asyncio.create_task(depth_worker(batch)))


async def stop_streams():
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if http_session:
        await http_session.close()


@asynccontextmanager
async def lifespan(app):
    await start_streams()
    yield
    await stop_streams()


app = FastAPI(title="Ψ-V10 Live Scanner", version="10.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/dashboard")


@app.get("/health")
async def health():
    ages = [
        now_ms() - st["updated_ms"]
        for st in states.values()
        if st["updated_ms"]
    ]
    ready = sum(
        1 for st in states.values()
        if st["trades"] and st["bids"] and st["asks"]
    )
    return {
        "status": "ok",
        "tracked_symbols": len(states),
        "fresh_symbols_under_5s": sum(age < 5000 for age in ages),
        "scanner_ready_symbols": ready,
        "uptime_seconds": round(time.time() - started_at, 1),
        "live_trading_enabled": LIVE_TRADING,
        "execution_status": "READ_ONLY_MARKET_DATA",
    }


@app.get("/scanner")
async def scanner(limit: int = 50):
    rows = snapshot(limit)
    return {"count": len(rows), "results": rows}


@app.get("/scanner/top")
async def scanner_top(limit: int = 10):
    return {"results": snapshot(limit)}


@app.get("/scanner/symbol/{symbol}")
async def scanner_symbol(symbol: str):
    return deep_dive(symbol)


@app.get("/paper/signals")
async def paper():
    return {"live_trading": LIVE_TRADING, "signals": list(paper_signals)[:100]}


@app.get("/chat/scan")
async def chat_scan(limit: int = 20):
    return {"command": "SCAN", "results": snapshot(limit)}


@app.get("/chat/top")
async def chat_top(limit: int = 10):
    return {"command": f"TOP {limit}", "results": snapshot(limit)}


@app.get("/chat/deep-dive/{symbol}")
async def chat_deep_dive(symbol: str):
    return {"command": f"DEEP DIVE {symbol.upper()}", **deep_dive(symbol)}


@app.get("/chat/status")
async def chat_status():
    return {"command": "STATUS", **(await health())}


@app.post("/chat/command")
async def chat_command(body: CommandBody):
    c = body.command.strip().upper()
    if c == "SCAN":
        return await chat_scan(20)
    if c.startswith("TOP "):
        try:
            n = int(c.split()[1])
        except Exception:
            n = 10
        return await chat_top(max(1, min(n, 100)))
    if c.startswith("DEEP DIVE "):
        return await chat_deep_dive(c.replace("DEEP DIVE ", "", 1).strip())
    if c == "STATUS":
        return await chat_status()
    if c == "PAPER":
        return await paper()
    if c == "HELP":
        return {"commands": ["SCAN", "TOP 10", "DEEP DIVE BTCUSDT", "PAPER", "STATUS"]}
    raise HTTPException(400, "Unknown command")


DASHBOARD = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ψ-V10 Live Scanner</title>
<style>
body{font-family:system-ui,sans-serif;background:#080b10;color:#e8edf5;margin:0}
header{padding:18px;border-bottom:1px solid #202733}
main{padding:18px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
.card{background:#10151d;border:1px solid #202733;border-radius:12px;padding:12px}
.value{font-size:22px;font-weight:700}.muted{color:#8c98aa;font-size:13px}
table{width:100%;border-collapse:collapse;background:#10151d}
th,td{padding:8px;border-bottom:1px solid #202733;text-align:right;font-size:12px}
th:first-child,td:first-child{text-align:left}
@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:10px}}
</style>
</head>
<body>
<header><b>Ψ-V10 Live Scanner</b><div class="muted">Live Binance microstructure telemetry</div></header>
<main>
<div class="cards">
<div class="card"><div class="muted">Tracked</div><div id="tracked" class="value">—</div></div>
<div class="card"><div class="muted">Fresh &lt;5s</div><div id="fresh" class="value">—</div></div>
<div class="card"><div class="muted">Ready</div><div id="ready" class="value">—</div></div>
<div class="card"><div class="muted">Signals</div><div id="signals" class="value">—</div></div>
</div>
<table><thead><tr><th>Symbol</th><th>State</th><th>V10 %</th><th>Price</th><th>Ψ</th><th>CVD</th><th>OFI</th><th>Vol Accel.</th><th>Breakout %</th><th>Score</th></tr></thead><tbody id="rows"></tbody></table>
</main>
<script>
async function load(){
try{
const [h,s]=await Promise.all([
fetch('/health').then(r=>r.json()),
fetch('/scanner/top?limit=50').then(r=>r.json())
]);
tracked.textContent=h.tracked_symbols;
fresh.textContent=h.fresh_symbols_under_5s;
ready.textContent=h.scanner_ready_symbols;
signals.textContent=s.results.filter(x=>x.signal).length;
rows.innerHTML=s.results.map(x=>`<tr><td>${x.symbol}</td><td>${x.v10_state}</td><td>${x.v10_confluence_pct}</td><td>${x.price}</td><td>${x.psi}</td><td>${x.cvd_ratio}</td><td>${x.ofi_proxy}</td><td>${x.volume_acceleration}</td><td>${x.breakout_proximity_pct}</td><td>${x.score}</td></tr>`).join('');
}catch(e){}
}
load();setInterval(load,2000);
</script>
</body>
</html>"""


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return DASHBOARD
