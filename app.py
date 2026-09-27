
import asyncio, json, math, os, statistics, time
from collections import defaultdict, deque
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
REFRESH_SECONDS = float(os.getenv("REFRESH_SECONDS", "2"))
LOOKBACK_SECONDS = int(os.getenv("LOOKBACK_SECONDS", "60"))
NEAR_BOOK_USD = float(os.getenv("NEAR_BOOK_USD", "25000"))
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "2.0"))
VOL_ACCEL_THRESHOLD = float(os.getenv("VOL_ACCEL_THRESHOLD", "1.5"))
BUY_IMBALANCE_THRESHOLD = float(os.getenv("BUY_IMBALANCE_THRESHOLD", "0.20"))

# Live execution is deliberately OFF unless LIVE_TRADING=true AND credentials exist.
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

states: dict[str, dict[str, Any]] = {}
paper_signals: deque = deque(maxlen=500)
trade_tasks = []
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
            "signal": False,
            "signal_reason": "",
        }
    return states[symbol]

async def get_json(path, params=None):
    async with http_session.get(BINANCE_REST + path, params=params, timeout=15) as r:
        r.raise_for_status()
        return await r.json()

async def discover_symbols():
    info = await get_json("/api/v3/exchangeInfo")
    tickers = await get_json("/api/v3/ticker/24hr")
    vol = {x["symbol"]: safe_float(x.get("quoteVolume")) for x in tickers}
    symbols = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("isSpotTradingAllowed") is False:
            continue
        symbols.append((s["symbol"], vol.get(s["symbol"], 0.0)))
    symbols.sort(key=lambda x: x[1], reverse=True)
    return [x[0].lower() for x in symbols[:MAX_SYMBOLS]]

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
    buy_imb = (buy_notional - sell_notional) / total if total else 0.0

    near_ask = 0.0
    near_bid = 0.0
    ask_top = asks[0][0] if asks else 0
    bid_top = bids[0][0] if bids else 0
    if ask_top:
        for p, q in asks:
            n = p * q
            if near_ask + n <= NEAR_BOOK_USD:
                near_ask += n
            else:
                near_ask += max(0.0, NEAR_BOOK_USD - near_ask)
                break
    if bid_top:
        for p, q in bids:
            n = p * q
            if near_bid + n <= NEAR_BOOK_USD:
                near_bid += n
            else:
                near_bid += max(0.0, NEAR_BOOK_USD - near_bid)
                break

    psi = buy_notional / max(near_ask, 1.0)
    book_total = near_bid + near_ask
    book_imb = (near_bid - near_ask) / book_total if book_total else 0.0

    # Compare first half of the rolling trade window with the second half.
    mid = len(trades) // 2
    first = sum(t["notional"] for t in trades[:mid]) if mid else 0
    second = sum(t["notional"] for t in trades[mid:]) if mid else 0
    vol_accel = second / max(first, 1.0)

    prices = [t["price"] for t in trades]
    price_change = (prices[-1] / prices[0] - 1) * 100 if prices[0] else 0
    spread_pct = ((ask_top - bid_top) / bid_top * 100) if bid_top else 0

    score = (
        math.log1p(max(psi, 0))
        + 1.5 * max(buy_imb, 0)
        + 0.75 * math.log1p(max(vol_accel, 0))
        + 0.5 * max(book_imb, 0)
    )

    signal = (
        psi >= PSI_THRESHOLD
        and vol_accel >= VOL_ACCEL_THRESHOLD
        and buy_imb > BUY_IMBALANCE_THRESHOLD
        and len(trades) >= 20
    )

    return {
        "symbol": st["symbol"],
        "price": round(st["price"], 10),
        "psi": round(psi, 4),
        "buy_imbalance": round(buy_imb, 4),
        "book_imbalance": round(book_imb, 4),
        "volume_acceleration": round(vol_accel, 4),
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

def snapshot(limit=20):
    rows = []
    for st in states.values():
        x = calc(st)
        if x:
            rows.append(x)
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[:max(1, min(limit, 200))]

def deep_dive(symbol):
    key = symbol.upper()
    st = states.get(key.lower())
    if not st:
        # state keys are lowercase from discovery
        for k, v in states.items():
            if k.upper() == key:
                st = v
                break
    if not st:
        raise HTTPException(404, f"{key} is not currently tracked")
    x = calc(st)
    if not x:
        raise HTTPException(503, f"Insufficient live data for {key}")
    trades = list(st["trades"])
    recent = trades[-20:]
    buys = sum(1 for t in recent if t["side"] == "buy")
    sells = len(recent) - buys
    return {
        "metrics": x,
        "recent_20_trade_buy_count": buys,
        "recent_20_trade_sell_count": sells,
        "top_bid": st["bids"][0] if st["bids"] else None,
        "top_ask": st["asks"][0] if st["asks"] else None,
        "top_5_bids": st["bids"][:5],
        "top_5_asks": st["asks"][:5],
    }

async def ws_worker(symbols):
    streams = []
    for s in symbols:
        streams += [f"{s}@aggTrade", f"{s}@depth20@100ms"]
    url = BINANCE_WS
    while True:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(url, heartbeat=20, receive_timeout=60) as ws:
                    await ws.send_json({"method": "SUBSCRIBE", "params": streams, "id": int(time.time())})
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        if "result" in data:
                            continue
                        payload = data.get("data", data)
                        event = payload.get("e")
                        symbol = payload.get("s", "").lower()
                        if not symbol:
                            continue
                        st = ensure_state(symbol)
                        st["updated_ms"] = now_ms()

                        if event == "aggTrade":
                            price = safe_float(payload.get("p"))
                            qty = safe_float(payload.get("q"))
                            # m=true means buyer is market maker => seller/taker sell.
                            side = "sell" if payload.get("m") else "buy"
                            st["price"] = price
                            st["trades"].append({
                                "ts": safe_float(payload.get("T"), now_ms()),
                                "price": price,
                                "qty": qty,
                                "notional": price * qty,
                                "side": side,
                            })
                            x = calc(st)
                            if x and x["signal"]:
                                paper_signals.appendleft({
                                    "timestamp_ms": now_ms(),
                                    **x
                                })
                        elif event == "depthUpdate":
                            # Fallback handler for diff-depth messages if returned.
                            # Partial depth is preferred below; ignore updates we cannot safely apply.
                            continue
                        elif event in ("depth20",):
                            st["bids"] = [(safe_float(p), safe_float(q)) for p, q in payload.get("bids", [])]
                            st["asks"] = [(safe_float(p), safe_float(q)) for p, q in payload.get("asks", [])]
        except Exception:
            await asyncio.sleep(3)

async def depth_worker(symbols):
    # Dedicated partial-book streams; keeps top-20 books fresh.
    streams = [f"{s}@depth20@100ms" for s in symbols]
    while True:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(BINANCE_WS, heartbeat=20, receive_timeout=60) as ws:
                    await ws.send_json({"method": "SUBSCRIBE", "params": streams, "id": int(time.time())})
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        if "result" in data:
                            continue
                        payload = data.get("data", data)
                        symbol = payload.get("s", "").lower()
                        if not symbol:
                            continue
                        st = ensure_state(symbol)
                        st["bids"] = [(safe_float(p), safe_float(q)) for p, q in payload.get("b", payload.get("bids", []))]
                        st["asks"] = [(safe_float(p), safe_float(q)) for p, q in payload.get("a", payload.get("asks", []))]
                        st["updated_ms"] = now_ms()
        except Exception:
            await asyncio.sleep(3)

async def start_streams():
    global http_session
    http_session = aiohttp.ClientSession()
    symbols = await discover_symbols()
    for s in symbols:
        ensure_state(s)
    # One connection per batch for trades; one per batch for depth.
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        trade_tasks.append(asyncio.create_task(ws_worker(batch)))
        trade_tasks.append(asyncio.create_task(depth_worker(batch)))

async def stop_streams():
    for t in trade_tasks:
        t.cancel()
    if http_session:
        await http_session.close()

@asynccontextmanager
async def lifespan(app):
    await start_streams()
    yield
    await stop_streams()

app = FastAPI(title="Binance Quant Scanner V2", version="2.0.0", lifespan=lifespan)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/dashboard")

@app.get("/health")
async def health():
    ages = [now_ms() - st["updated_ms"] for st in states.values() if st["updated_ms"]]
    return {
        "status": "ok",
        "tracked_symbols": len(states),
        "fresh_symbols_under_5s": sum(a < 5000 for a in ages),
        "uptime_seconds": round(time.time() - started_at, 1),
        "live_trading_enabled": LIVE_TRADING,
        "execution_status": "NOT_IMPLEMENTED_IN_V2_SAFETY_BUILD",
    }

@app.get("/scanner")
async def scanner(limit: int = 50):
    return {"count": min(limit, len(states)), "results": snapshot(limit)}

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
    rows = snapshot(limit)
    return {"command": "SCAN", "results": rows}

@app.get("/chat/top")
async def chat_top(limit: int = 10):
    rows = snapshot(limit)
    return {"command": f"TOP {limit}", "results": rows}

@app.get("/chat/deep-dive/{symbol}")
async def chat_deep_dive(symbol: str):
    return {"command": f"DEEP DIVE {symbol.upper()}", **deep_dive(symbol)}

@app.get("/chat/paper")
async def chat_paper():
    return {"command": "PAPER", "live_trading": LIVE_TRADING, "signals": list(paper_signals)[:100]}

@app.get("/chat/status")
async def chat_status():
    return {"command": "STATUS", **(await health())}

@app.post("/chat/command")
async def chat_command(body: CommandBody):
    raw = body.command.strip()
    c = raw.upper()
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
    if c == "PAPER":
        return await chat_paper()
    if c == "STATUS":
        return await chat_status()
    if c == "HELP":
        return {
            "commands": [
                "SCAN", "TOP 10", "DEEP DIVE BTCUSDT", "PAPER", "STATUS"
            ]
        }
    raise HTTPException(400, "Unknown command. Use SCAN, TOP 10, DEEP DIVE SYMBOL, PAPER, STATUS, or HELP.")

DASHBOARD = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Binance Quant Scanner V2</title>
<style>
body{font-family:Inter,system-ui,sans-serif;background:#080b10;color:#e8edf5;margin:0}
header{padding:18px 22px;border-bottom:1px solid #202733;display:flex;justify-content:space-between;align-items:center}
h1{font-size:20px;margin:0}.muted{color:#8c98aa;font-size:13px}
main{padding:18px 22px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}
.card{background:#10151d;border:1px solid #202733;border-radius:12px;padding:14px}.value{font-size:22px;font-weight:700;margin-top:5px}
table{width:100%;border-collapse:collapse;background:#10151d;border:1px solid #202733;border-radius:12px;overflow:hidden}
th,td{text-align:right;padding:10px 8px;border-bottom:1px solid #1c232e;font-size:12px}th{text-align:right;color:#8c98aa}
th:first-child,td:first-child{text-align:left}.hot{font-weight:800}.yes{font-weight:800}.toolbar{display:flex;gap:8px;align-items:center;margin-bottom:12px}
button,input{background:#111821;border:1px solid #2a3442;color:#e8edf5;padding:8px 10px;border-radius:8px}
@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}table{font-size:11px}main{padding:12px}}
</style></head>
<body><header><div><h1>Binance Quant Scanner V2</h1><div class="muted">Ψ aggressive-buy pressure vs near-ask liquidity</div></div><div id="clock" class="muted">connecting…</div></header>
<main><div class="cards">
<div class="card"><div class="muted">Tracked</div><div id="tracked" class="value">—</div></div>
<div class="card"><div class="muted">Fresh &lt;5s</div><div id="fresh" class="value">—</div></div>
<div class="card"><div class="muted">Live Trading</div><div id="live" class="value">OFF</div></div>
<div class="card"><div class="muted">Signals</div><div id="signals" class="value">—</div></div>
</div>
<div class="toolbar"><button onclick="load()">Refresh</button><span class="muted">Auto-refresh every 2 seconds</span></div>
<table><thead><tr><th>Symbol</th><th>Price</th><th>Ψ</th><th>Buy Imb.</th><th>Book Imb.</th><th>Vol Accel.</th><th>1m %</th><th>Ask $</th><th>Score</th><th>Signal</th></tr></thead><tbody id="rows"></tbody></table>
</main>
<script>
async function load(){
 try{
  const [h,s]=await Promise.all([fetch('/health').then(r=>r.json()),fetch('/scanner/top?limit=50').then(r=>r.json())]);
  tracked.textContent=h.tracked_symbols; fresh.textContent=h.fresh_symbols_under_5s; live.textContent=h.live_trading_enabled?'ON':'OFF';
  signals.textContent=s.results.filter(x=>x.signal).length;
  rows.innerHTML=s.results.map(x=>`<tr><td>${x.symbol}</td><td>${x.price}</td><td class="${x.psi>=2?'hot':''}">${x.psi}</td><td>${x.buy_imbalance}</td><td>${x.book_imbalance}</td><td>${x.volume_acceleration}</td><td>${x.price_change_1m_pct}</td><td>${x.near_ask_usd}</td><td>${x.score}</td><td class="${x.signal?'yes':''}">${x.signal?'YES':''}</td></tr>`).join('');
  clock.textContent=new Date().toLocaleTimeString();
 }catch(e){clock.textContent='connection error'}
}
load();setInterval(load,2000);
</script></body></html>"""

@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return DASHBOARD
