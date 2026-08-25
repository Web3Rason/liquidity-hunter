"""
取得掃描標的清單。
- 加密貨幣：Binance USDT 永續合約（fapi exchangeInfo）
- 美股：S&P500 + NASDAQ100（Wikipedia）
"""
import requests
import pandas as pd
from io import StringIO

_UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")}


def get_crypto_perps(min_qvol_usdt: float = 0) -> list:
    """所有 TRADING 中的 Binance USDT 永續合約代號（預設全掃 527 檔）。

    min_qvol_usdt > 0 才剔除低流動性標的（opt-in）：低流動性幣的 taker delta
    一筆市價單就能把 z-score 打爆，且本就不是「流動性獵取」該狩獵的池子。
    監控與回測共用本函式 → universe 一致；要乾淨池子兩邊都傳 5e6。
    官方文件：GET /fapi/v1/ticker/24hr，quoteVolume=24h 成交額(USDT,字串)，已實測回傳結構。
    """
    info = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=25).json()
    syms = {s["symbol"] for s in info["symbols"]
            if s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"}
    if min_qvol_usdt:
        try:
            tick = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=25).json()
            if not isinstance(tick, list):
                raise RuntimeError(f"非預期回應: {str(tick)[:100]}")
            liquid = {t["symbol"] for t in tick
                      if float(t.get("quoteVolume") or 0) >= min_qvol_usdt}
        except Exception as e:   # 過濾失敗就全掃，不讓整次掃描死掉
            print(f"[universe] 24h ticker 取得失敗，跳過流動性過濾：{e}")
        else:
            dropped = len(syms) - len(syms & liquid)
            syms &= liquid
            print(f"[universe] 剔除 24h 成交額 < {min_qvol_usdt/1e6:.0f}M USDT 共 {dropped} 檔")
    return sorted(syms)


def _wiki_tables(url: str) -> list:
    r = requests.get(url, headers=_UA, timeout=25)
    r.raise_for_status()
    return pd.read_html(StringIO(r.text))


def get_stock_universe() -> list:
    """S&P500 + NASDAQ100 去重；代號轉成 Yahoo 格式（. → -）。"""
    syms = set()
    try:
        sp = _wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        syms.update(sp["Symbol"].astype(str).tolist())
    except Exception as e:
        print(f"[universe] S&P500 取得失敗：{e}")
    try:
        for t in _wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
            hit = [c for c in t.columns if str(c) in ("Ticker", "Symbol")]
            if hit:
                syms.update(t[hit[0]].astype(str).tolist())
                break
    except Exception as e:
        print(f"[universe] NASDAQ100 取得失敗：{e}")
    out = sorted(s.replace(".", "-").strip() for s in syms if s and s != "nan")
    if len(out) < 300:   # 正常 S&P500+NASDAQ100 去重後約 520 檔
        print(f"[universe] 警告：美股清單僅 {len(out)} 檔，Wikipedia 表格結構可能已變動")
    return out


if __name__ == "__main__":
    c = get_crypto_perps()
    s = get_stock_universe()
    print(f"crypto perps: {len(c)} | stocks: {len(s)}")
