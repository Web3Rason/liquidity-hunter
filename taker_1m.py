"""
1 分 K taker 確認（crypto 專用）—— 方法論作者的實際做法：價格觸碰流動性後看小級別 taker。

日 K 把插破的恐慌賣壓與收復的買盤混在同一個 24h 容器（淨 delta 失真）；
1m 才能把「介入的瞬間」單獨抓出來。設計：

兩個精準節點（以做多 B低 為例）任一成立即 ⚡（節點名對應作者教法，與日線層的
「吸收/反轉」同名異義問題已修正——忠實度審查指出後改名）：
  - 極值節點（作者第 1 種進場：Stop Hunt 當下的反向強力 taker）：
    極值日「當日最低價那根 1m ±2 根」內，
    有 1m 同時滿足 量>當日PR99、買比>0.6、名目額>max(當日總額0.1%, 5萬U)
  - 收復節點（作者：「監測漲回/跌回區間時，是否有強力 taker 介入」）：
    收復日「首次 1m 收盤站回線上那根 +2 根」內，同上條件
做空 B高 鏡像（賣比>0.6、最高價節點、首次收破線）。
美股無分鐘級歷史 → 維持日線放量判定（taker.py）。
快取 .cache_klines/m1/{sym}_{day}.pkl，同標的同日只抓一次。
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

M1_CACHE = Path(__file__).parent / ".cache_klines" / "m1"

BUY_RATIO = 0.6          # 主動方佔比門檻（多=買比、空=賣比）
VOL_PCTL = 99            # 爆量門檻：當日 1m 量的百分位
MIN_QUOTE = 50_000       # 名目額底線（U）：防殭屍幣基期過低的假爆量
MIN_DAY_FRAC = 0.001     # 或至少佔當日總成交額 0.1%
WIN_EXTREME = 2          # 極值窗：極值根 ±2
WIN_RECLAIM = 2          # 收復窗：首次收復根 +2


def fetch_1m_day(symbol: str, day_ts: int) -> pd.DataFrame | None:
    """抓某 UTC 日的 1m（1440 根），帶磁碟快取；失敗回 None。"""
    M1_CACHE.mkdir(parents=True, exist_ok=True)
    f = M1_CACHE / f"{symbol}_{day_ts}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/klines",
                            params={"symbol": symbol, "interval": "1m",
                                    "startTime": day_ts * 1000, "limit": 1440}, timeout=30)
        resp.raise_for_status()
        b = resp.json()
        if not isinstance(b, list) or not b:
            return None
        b = [x for x in b if day_ts * 1000 <= x[0] < (day_ts + 86400) * 1000]  # 日界過濾：
        if not b:                                       # 交易所停機缺口時 limit=1440 可能吞到次日
            return None
        m = pd.DataFrame({"time": [x[0] // 1000 for x in b],
                          "close": [float(x[4]) for x in b],
                          "vol": [float(x[5]) for x in b],
                          "tbuy": [float(x[9]) for x in b]})
        m.to_pickle(f)
        time.sleep(0.25)   # 節流
        return m
    except Exception:
        return None


def _burst_in(m: pd.DataFrame, idxs, side: str) -> bool:
    """窗口內是否存在符合 爆量+方向佔比+名目底線 的 1m。"""
    vol = m["vol"].values
    tbuy = m["tbuy"].values
    quote = vol * m["close"].values
    pr = np.percentile(vol[vol > 0], VOL_PCTL) if (vol > 0).any() else float("inf")
    q_floor = max(MIN_QUOTE, quote.sum() * MIN_DAY_FRAC)
    for j in idxs:
        if not (0 <= j < len(m)) or vol[j] <= 0:
            continue
        ratio = tbuy[j] / vol[j] if side == "low" else 1 - tbuy[j] / vol[j]
        if vol[j] > pr and ratio > BUY_RATIO and quote[j] > q_floor:
            return True
    return False


def m1_confirm(symbol: str, side: str, level: float,
               extreme_day_ts: int, reclaim_day_ts: int) -> dict:
    """回傳 {ok, kind('吸收'/'反轉'/None), note}；資料抓不到時 ok=None（無法判定）。"""
    out = {"ok": None, "kind": None}
    m_ext = fetch_1m_day(symbol, extreme_day_ts)
    m_rec = m_ext if reclaim_day_ts == extreme_day_ts else fetch_1m_day(symbol, reclaim_day_ts)
    if m_ext is None and m_rec is None:
        return out
    # 吸收：極值日的極值根 ±WIN
    if m_ext is not None and len(m_ext):
        ei = int(m_ext["close"].idxmin()) if side == "low" else int(m_ext["close"].idxmax())
        # 用影線極值更準：close 序列沒有 high/low，1m 粒度下 close 極值已足夠貼近
        if _burst_in(m_ext, range(ei - WIN_EXTREME, ei + WIN_EXTREME + 1), side):
            out.update(ok=True, kind="極值")
            return out
    # 反轉：收復日首次站回線的那根 +WIN
    if m_rec is not None and len(m_rec):
        c = m_rec["close"].values
        if side == "low":
            cross = np.where((np.roll(c, 1) < level) & (c >= level))[0]
        else:
            cross = np.where((np.roll(c, 1) > level) & (c <= level))[0]
        cross = cross[cross > 0]
        if len(cross):
            ri = int(cross[0])
            if _burst_in(m_rec, range(ri, ri + WIN_RECLAIM + 1), side):
                out.update(ok=True, kind="收復")
                return out
    # 半盲防護：有任一日抓不到資料且未確認 → 無法判定(None)，不給硬否決
    out["ok"] = False if (m_ext is not None and m_rec is not None) else None
    return out


def m1_window_confirm(symbol: str, side: str,
                      dev_day_ts: int, reclaim_day_ts: int) -> dict:
    """線復活第二級(L2)專用的放寬 taker 閘（使用者裁決）：不限兩個精準節點，
    掃「插破→收復」整段窗口每一日的 1m，任一日存在符合 爆量+方向佔比+名目底線 的 1m 即確認。
    理由：CRV 型「漸進吸籌」的強力買盤分散在多日、不落在極值/收復的窄窗，嚴格節點會誤殺。
    回傳 {ok, kind}：ok=True(任一日有強力 taker) / False(各日皆抓到資料但無) / None(全段無資料)。"""
    out = {"ok": None, "kind": None}
    day = dev_day_ts
    any_data = False
    while day <= reclaim_day_ts:
        m = fetch_1m_day(symbol, day)
        day += 86400
        if m is None or not len(m):
            continue
        any_data = True
        if _burst_in(m, range(len(m)), side):     # 整日掃，非僅節點窗
            out.update(ok=True, kind="整段")
            return out
    out["ok"] = False if any_data else None
    return out
