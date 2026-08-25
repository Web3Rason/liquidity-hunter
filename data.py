"""
歷史 K 線資料抓取。

加密貨幣：Binance USDⓈ-M 永續合約 公開 REST /fapi/v1/klines（不需 API key）
官方文件：https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data
（用永續合約，貼合「貝格先生」分析用的 PERPETUAL 圖；K 線含 taker 主動買量欄位，未來做 CVD 確認可直接用）

回傳格式統一為 pandas DataFrame，欄位：
    time(秒, int)、open、high、low、close、volume、taker_buy_base
"""
import time
import requests
import pandas as pd

FAPI_BASE = "https://fapi.binance.com"
KLINES_PATH = "/fapi/v1/klines"

# 期貨單次最多回傳 1500 根
_MAX_LIMIT = 1500

_INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def fetch_binance_klines(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    """抓 Binance 永續合約歷史 K 線。

    Args:
        symbol: 交易對，如 "BTCUSDT"
        interval: "1h" / "4h" / "1d"
        bars: 想要的總根數（自動分頁湊齊）

    Returns:
        DataFrame（時間升序），欄位 time/open/high/low/close/volume/taker_buy_base
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f"不支援的 interval: {interval}")

    interval_ms = _INTERVAL_MS[interval]
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - bars * interval_ms

    rows = []
    cursor = start_ms
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "limit": _MAX_LIMIT,
        }
        # 節流退避：429=限流(等待重試)，418=IP 被 ban(長時間，直接中止)
        for attempt in range(4):
            resp = requests.get(FAPI_BASE + KLINES_PATH, params=params, timeout=20)
            if resp.status_code == 418:
                raise RuntimeError(f"Binance IP banned (418): {resp.text[:120]}")
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("Retry-After", 3)) + 2)
                continue
            resp.raise_for_status()
            break
        else:
            # 重試耗盡仍 429：必須拋錯中止，不能把錯誤 JSON 當 K 線、也不能繼續打 API
            raise RuntimeError(f"Binance 限流未恢復 (429 x4): {resp.text[:120]}")
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        next_cursor = last_open + interval_ms
        if next_cursor <= cursor or len(batch) < _MAX_LIMIT or next_cursor >= now_ms:
            break
        cursor = next_cursor
        time.sleep(0.2)

    # 期貨 kline 欄位：
    # [0]openTime [1]open [2]high [3]low [4]close [5]volume [6]closeTime
    # [7]quoteVol [8]trades [9]takerBuyBaseVol [10]takerBuyQuoteVol [11]ignore
    cols = ["openTime", "open", "high", "low", "close", "volume",
            "closeTime", "qav", "trades", "tbbav", "tbqav", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "taker_buy_base"])

    df = df.drop_duplicates(subset="openTime").sort_values("openTime")
    # 丟掉「未收盤」的最後一根：偵測器把 close 當收盤價判定突破/收復，
    # 用走到一半的 K 棒會產生隔天消失的假訊號
    closed_cutoff_ms = int(time.time() * 1000) - interval_ms
    df = df[df["openTime"] <= closed_cutoff_ms]
    out = pd.DataFrame({
        "time": (df["openTime"] // 1000).astype("int64"),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "taker_buy_base": df["tbbav"].astype(float),
    })
    return out.reset_index(drop=True)


# ---- 美股：yfinance ----
_STOCK_PERIOD = {"1h": "720d", "1d": "5y"}  # yfinance 1h 最多約 730 天


def _drop_open_stock_bar(h, interval: str):
    """美股盤中抓資料會含「未收盤」的最後一根，砍掉（與 crypto 同理，防假訊號）。
    日線：tz-aware(紐約午夜)起算 16h=16:00 ET 收盤；tz-naive 視為 UTC 午夜，保守用 21h(=16:00 EST)。
    """
    if h.empty:
        return h
    last_ts = h.index[-1]
    if interval == "1d":
        close_off = 16 * 3600 if last_ts.tz is not None else 21 * 3600
    else:  # 1h
        close_off = 3600
    if int(last_ts.value // 10**9) + close_off > time.time():
        h = h.iloc[:-1]
    return h


def fetch_stock(symbol: str, interval: str, bars: int = 0) -> pd.DataFrame:
    """抓美股歷史 K 線（yfinance）。bars 參數僅為與 binance 介面一致，實際依 period 抓滿後裁切。"""
    import yfinance as yf
    if interval not in _STOCK_PERIOD:
        raise ValueError(f"美股不支援的 interval: {interval}")
    h = yf.Ticker(symbol).history(period=_STOCK_PERIOD[interval], interval=interval)
    h = _drop_open_stock_bar(h, interval)
    if h.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "taker_buy_base"])
    h = h.reset_index()
    tcol = h.columns[0]  # Date 或 Datetime
    out = pd.DataFrame({
        "time": (h[tcol].astype("int64") // 10**9).astype("int64"),
        "open": h["Open"].astype(float),
        "high": h["High"].astype(float),
        "low": h["Low"].astype(float),
        "close": h["Close"].astype(float),
        "volume": h["Volume"].astype(float),
        "taker_buy_base": float("nan"),  # 美股無 taker 分解，留空
    })
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if bars:
        out = out.tail(bars).reset_index(drop=True)
    return out


def fetch_stocks_batch(symbols: list, period: str = "2y") -> dict:
    """一次批次抓多檔美股日線（yfinance 內部批次，單一程序）。回傳 {symbol: DataFrame}。"""
    import yfinance as yf
    out = {}
    if not symbols:
        return out
    raw = yf.download(symbols, period=period, interval="1d",
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)
    for sym in symbols:
        try:
            h = raw[sym] if len(symbols) > 1 else raw
            h = h.dropna(how="all")
            h = _drop_open_stock_bar(h, "1d")
            if h.empty:
                continue
            df = pd.DataFrame({
                "time": (h.index.astype("int64") // 10**9).astype("int64"),
                "open": h["Open"].astype(float),
                "high": h["High"].astype(float),
                "low": h["Low"].astype(float),
                "close": h["Close"].astype(float),
                "volume": h["Volume"].astype(float),
            }).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
            if not df.empty:
                out[sym] = df
        except Exception:
            continue
    return out


if __name__ == "__main__":
    for itv, n in [("1d", 30), ("1h", 50)]:
        d = fetch_binance_klines("BTCUSDT", itv, n)
        print(f"BTCUSDT {itv}: {len(d)} 根, 最後 close={d['close'].iloc[-1]}")
    m = fetch_stock("MU", "1d")
    print(f"MU 1d: {len(m)} 根, 最後 close={m['close'].iloc[-1]}")
