"""
全市場日線掃描：找出「近 N 天內出現 B 型流動性獵取(假突破/假跌破收復)」的標的，
產生前端清單 + 每個命中可在頁內自繪 K 線圖（支撐/壓力線 + 觸碰點 + 獵取標記，不連 TradingView）。

範圍：Binance USDT 永續(全) + 美股 S&P500+NASDAQ100
條件：type=B、touches>=2、收復日(sweep_time) 在近 N 天內
依機器負載規則：單一程序序列執行（crypto 逐檔、stocks 批次但同程序）。

用法：
    python market_scan.py                 # 完整掃描
    python market_scan.py --days 7
    python market_scan.py --limit 30      # 測試：每個市場只掃前 N 檔
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

_STATE = Path(__file__).parent / "scan_state.json"
# 推播腳本：設環境變數 LH_NOTIFY_SCRIPT 指向自己的推送程式，
# 會以 `python <script> "<訊息>"` 呼叫。沒設就只印在畫面上。
_NOTIFY = os.environ.get("LH_NOTIFY_SCRIPT", "")


def _key(r):
    return f"{r['market']}|{r['symbol']}|{r['side']}|{r['sweep_time']}"


def notify_new(results, now):
    """跟上次比對：第一次只記錄+發啟動訊息；之後只推「新增」的命中。"""
    first_run = not _STATE.exists()
    seen = set()
    if not first_run:
        try:
            seen = set(json.loads(_STATE.read_text(encoding="utf-8")).get("seen", []))
        except Exception:
            seen = set()
    # 只推「近 days 天剛收復(is_new)且沒推過」的——清單含持倉中舊單，但推播只認新命中
    new = [r for r in results if r.get("is_new") and _key(r) not in seen]
    # 更新狀態：保留近 30 天的 key（避免無限膨脹）
    cut = now - 30 * 86400
    keep = {k for k in (seen | {_key(r) for r in results})
            if k.rsplit("|", 1)[-1].isdigit() and int(k.rsplit("|", 1)[-1]) >= cut}
    _STATE.write_text(json.dumps({"seen": sorted(keep)}), encoding="utf-8")

    if first_run:
        _send(f"5035 流動性獵取監控啟動：已記錄現有 {len(results)} 筆，之後只推新增命中")
        return
    if not new:
        print("無新增命中，不推播")
        return
    # 原文：「沒有強力 taker 介入，無法確認 Stop Hunt」→ 只列 ⚡ 明細，未確認彙總筆數
    cs = [f"{r['symbol']} {r['dir'][:2]}⚡" for r in new
          if r["market"] == "crypto" and r.get("taker_ok")][:20]
    ss = [f"{r['symbol']} {r['dir'][:2]}⚡" for r in new
          if r["market"] == "stock" and r.get("taker_ok")][:20]
    n_unconf = sum(1 for r in new if not r.get("taker_ok"))
    msg = f"5035 流動性獵取·新增 {len(new)} 筆"
    if cs:
        msg += f" | 加密: {'、'.join(cs)}"
    if ss:
        msg += f" | 美股: {'、'.join(ss)}"
    if n_unconf:
        msg += (f" | 另 {n_unconf} 筆無⚡確認(候選)見 scanner" if (cs or ss)
                else f" | 全部 {n_unconf} 筆皆無⚡確認(候選)，明細見 scanner")
    _send(msg)


def _send(msg):
    print(msg)
    if not _NOTIFY:
        return  # 沒設 LH_NOTIFY_SCRIPT，印出來就好
    try:
        subprocess.run(["python", _NOTIFY, msg], timeout=30)
    except Exception as e:
        print(f"[notify] 失敗：{e}")

import numpy as np
import pandas as pd

from data import fetch_binance_klines, fetch_stocks_batch
from detector import _swings, detect_sweeps, get_config
from taker import attach_taker, rearm_blocked
from taker_1m import m1_confirm, m1_window_confirm
from universe import get_crypto_perps, get_stock_universe


def apply_m1(df, signals):
    """crypto ⚡ 改由 1 分 K 判定（作者實際看法）；日線 z 降為輔助。
    每個命中多 1~2 個 API 請求（有快取）；抓不到資料時保留日線判定。"""
    lo = df["low"].values
    hi = df["high"].values
    tt = df["time"].values
    for s in signals:
        if s["type"] != "B":
            continue
        j0, j1 = s["dev_idx"], s["sweep_idx"]
        if int(s.get("rearm_level", 1)) == 2:
            # L2 放寬閘（使用者裁決）：掃插破→收復整段窗口，非僅極值/收復兩節點
            r = m1_window_confirm(s.get("_symbol", ""), s["side"],
                                  int(tt[j0]) // 86400 * 86400, int(tt[j1]) // 86400 * 86400)
        else:
            ed = j0 + (int(np.argmin(lo[j0:j1 + 1])) if s["side"] == "low"
                       else int(np.argmax(hi[j0:j1 + 1])))
            r = m1_confirm(s.get("_symbol", ""), s["side"], s["level_price"],
                           int(tt[ed]) // 86400 * 86400, int(tt[j1]) // 86400 * 86400)
        if r["ok"] is not None:
            s["taker_ok"] = bool(r["ok"])
            s["taker_kind"] = f"1m{r['kind']}" if r["ok"] else (s["taker_kind"] or "")
    return signals

CRYPTO_BARS = 400
MIN_TOUCHES = 2
VIEW_BARS = 160           # 預設視野根數（資料是全量，這只是初始縮放）
_LIB_PATH = Path(__file__).parent / "vendor" / "lightweight-charts.standalone.production.js"


def _atr14(df):
    """ATR(14)，純 numpy（與 detector 的 rolling(14).mean 同義；前 14 根為 NaN）。"""
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    pc = np.empty_like(c, dtype=float)
    pc[0] = np.nan
    pc[1:] = c[:-1]
    tr = np.maximum.reduce([h - lo, np.abs(h - pc), np.abs(lo - pc)])
    atr = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr[13:] = np.convolve(tr, np.ones(14) / 14, "valid")
    return atr


def _entry_decision(df, s, atr, cfg):
    """進場分流（鏡像 backtest，不洩漏：每根用該根收盤決定次根進場）。回傳 (status, ie)：
    entered(已進場，ie=進場根) / watching(觀察中，未進場) / abandoned(放棄)。
    強收復(站回≥reclaim_enter ATR)→立即進場；弱收復→觀察 watch_bars 根，站回達門檻→進場、破針尖→放棄。"""
    i = int(s["sweep_idx"])
    n = len(df)
    side = s["side"]
    c = df["close"].values
    lvl = s["level_price"]
    tip = float(s["extreme"])
    re = cfg["reclaim_enter"]
    wb = cfg["watch_bars"]

    def rec(k, ak):
        return (c[k] - lvl) / ak if side == "low" else (lvl - c[k]) / ak

    a_sig = atr[i]
    if not (np.isfinite(a_sig) and a_sig > 0):
        return ("abandoned", None)
    if rec(i, a_sig) >= re:
        return ("entered", i)                        # 強收復，立即(次根)進場
    for k in range(i + 1, min(n, i + 1 + wb)):
        if (c[k] < tip) if side == "low" else (c[k] > tip):
            return ("abandoned", None)               # 收盤破針尖＝證偽，放棄
        ak = atr[k]
        if np.isfinite(ak) and ak > 0 and rec(k, ak) >= re:
            return ("entered", k)                    # 觀察期內站回達門檻＝確認獵取
    if i + 1 + wb <= n:
        return ("abandoned", None)                   # 觀察窗已滿仍沒站高＝力道不足，放棄
    return ("watching", None)                        # 觀察窗未滿（最新訊號，仍觀察中）


def _v5_open(df, s, sw_h, sw_l, lr, ie):
    """v5 localhigh 結構移停出場模擬，回傳 (is_open, info)。ie=進場根(進場分流決定)。
    與 backtest.py 的 v5 模擬完全同構：初始 SL=插破極值(針尖，非收復根低點)；TP=下一個流動性聚集區(opp_level)；
    途中每出現一個更高的 swing low(多)/更低的 swing high(空)→SL 上移到「前一個 swing low/high」
    （使用者準則：沒 lower low 就一路跟；兩個 low 之間最高點=local high）。
    info["kind"]：holding(仍持倉) / tp(漲到聚集區) / trail(結構移停後被掃) / sl(初始停損)；
    出場時帶 exit_idx/exit_time（在 K 線中的位置與時間），持倉中則為 None。"""
    i = int(ie)                   # 進場根（進場分流決定；出場/移停/持倉天數皆以此為基準）
    n = len(df)
    side = s["side"]
    lo = df["low"].values
    h = df["high"].values
    tt = df["time"].values
    tgt = s.get("opp_level")
    stop0 = float(s["extreme"])   # 停損=跌破/突破流動線後的極值(作者「針尖高/低點」)
    stop_cur = stop0

    def _info(kind, j):           # j=None 表持倉中；否則為出場根 index
        return {"kind": kind, "exit_idx": (None if j is None else int(j)),
                "exit_time": (None if j is None else int(tt[j])),
                "armed": stop_cur != stop0, "cur_stop": round(stop_cur, 8),
                "held_bars": (n - 1 - i if j is None else j - i)}

    prev_sw = None                # 上一個已確認 swing low(多)/high(空)；higher-low 出現才上移 SL 到它
    for j in range(i + 1, n):
        if (lo[j] <= stop_cur) if side == "low" else (h[j] >= stop_cur):
            return False, _info("sl" if stop_cur == stop0 else "trail", j)   # 被 SL 掃出
        if tgt is not None and ((h[j] >= tgt) if side == "low" else (lo[j] <= tgt)):
            return False, _info("tp", j)        # 漲到聚集區 TP，出場
        pi = j - lr
        if pi > i and (sw_l[pi] if side == "low" else sw_h[pi]):
            cur = lo[pi] if side == "low" else h[pi]
            if prev_sw is not None:
                if side == "low" and cur > prev_sw and prev_sw > stop_cur:
                    stop_cur = float(prev_sw)         # higher-low → 鎖到前一個低點
                elif side == "high" and cur < prev_sw and prev_sw < stop_cur:
                    stop_cur = float(prev_sw)         # lower-high → 鎖到前一個高點
            prev_sw = cur
    return True, _info("holding", None)


def _open_B(df, signals, market="crypto", days=7):
    """顯示三類 B 訊號：(1)觀察中（弱收復、進場分流尚未確認進場）；
    (2)持倉中（已進場、v5 結構移停未出場，不受時間窗）；(3)近 days 天已出場（tp/trail/sl）。
    進場分流判定放棄(abandoned)者不顯示。"""
    cfg = get_config("1d", market)
    lr = cfg["pivot_lr"]
    sw_h, sw_l = _swings(df["high"].values, df["low"].values, lr)
    atr = _atr14(df)
    last_t = int(df["time"].iloc[-1])
    cutoff = last_t - days * 86400
    out = []
    for s in signals:
        if s["type"] != "B" or s["touches"] < MIN_TOUCHES:
            continue
        status, ie = _entry_decision(df, s, atr, cfg)
        if status == "abandoned":
            continue
        s["_entry_idx"] = ie
        if status == "watching":                    # 弱收復觀察中，尚未進場（必為近期訊號）
            s["_open"] = {"kind": "watching", "exit_idx": None, "exit_time": None,
                          "armed": False, "cur_stop": None, "held_bars": None}
            s["_status"] = "watching"
            out.append(s)
            continue
        is_open, info = _v5_open(df, s, sw_h, sw_l, lr, ie)
        if is_open:                                 # 持倉中：全顯示，不受時間窗限制
            s["_open"] = info
            s["_status"] = "holding"
            out.append(s)
        elif info["exit_time"] is not None and info["exit_time"] >= cutoff:
            s["_open"] = info                       # 近 days 天剛出場（停利/移停/停損）
            s["_status"] = info["kind"]
            out.append(s)
    return out


def _row(market, sym, df, s):
    seg = df          # 全部 K 棒進圖表（預設視野由前端聚焦訊號區，可往左拖看完整歷史）
    vols = seg["volume"].fillna(0.0)
    candles = [{"time": int(t), "open": round(o, 6), "high": round(h, 6),
                "low": round(l, 6), "close": round(c, 6), "volume": round(v, 2)}
               for t, o, h, l, c, v in zip(seg["time"], seg["open"], seg["high"],
                                           seg["low"], seg["close"], vols)]
    touch_times = [int(df["time"].iloc[ti]) for ti in s["touch_idxs"] if 0 <= ti < len(df)]
    stop = float(s["extreme"])   # 停損=跌破/突破流動線後的極值(作者「針尖高/低點」)
    return {
        "market": market, "symbol": sym,
        "side": s["side"], "dir": "B高(空)" if s["side"] == "high" else "B低(多)",
        "level": s["level_price"], "extreme": s["extreme"],
        "stop": round(stop, 6), "tgt": s.get("opp_level"),   # TP=下一個流動性聚集區（等高/顯著前高低，可能無）
        "sweep_time": s["sweep_time"], "touches": s["touches"],
        "taker_z": s["taker_z"], "taker_kind": s["taker_kind"], "taker_ok": s["taker_ok"],
        "held_days": (s.get("_open") or {}).get("held_bars"),   # 持倉天數（收復至出場/至今交易日）
        "armed": (s.get("_open") or {}).get("armed"),           # SL 是否已被結構移停上移過
        "cur_stop": (s.get("_open") or {}).get("cur_stop"),     # 目前停損水位（結構移停後）
        "status": s.get("_status", "holding"),                  # watching/holding/tp/trail/sl
        "exit_time": (s.get("_open") or {}).get("exit_time"),   # 出場時間（持倉中/觀察中為 None）
        "exit_kind": (s.get("_open") or {}).get("kind"),        # 出場類型
        "entry_time": (int(df["time"].iloc[s["_entry_idx"]])    # 進場時間（觀察中尚未進場=None）
                       if s.get("_entry_idx") is not None else None),
        "last_close": round(float(df["close"].iloc[-1]), 6),
        "chart": {"candles": candles, "level": s["level_price"],
                  "touch_times": touch_times, "sweep_time": s["sweep_time"]},
    }


SCAN_HISTORY = Path(__file__).parent / "scan_history"   # 每日掃描存檔（K線快照+結果+html，不覆蓋）


def _process_symbol(market, sym, df, days, cutoff):
    """單標的：偵測→進場分流/出場模擬→1m taker(僅crypto)→L2硬閘→組 row。
    抽成共用函式：正式掃描與 --cached 重算共用同一條處理路徑（保證一致）。"""
    sigs = attach_taker(df, detect_sweeps(df, get_config("1d", market)))
    recent = _open_B(df, sigs, market, days)
    for s in recent:
        s["_symbol"] = sym
    if market == "crypto":   # 1m：持倉中(省API) + 所有 L2(線復活硬閘須先定 taker)；美股無 1m
        apply_m1(df, [s for s in recent
                      if s.get("_status") == "holding" or int(s.get("rearm_level", 1)) == 2])
    recent = [s for s in recent if not rearm_blocked(s)]   # L2 硬閘：無 taker 不顯示
    rows = []
    for s in recent:
        row = _row(market, sym, df, s)
        row["is_new"] = s["sweep_time"] >= cutoff   # 近 days 天剛收復=新命中
        rows.append(row)
    return rows


def _archive(now, days, results, klines):
    """每日掃描存檔到 scan_history/<日期>/：原始 K 線快照(供 --cached 重算) + 結果 + meta。
    不覆蓋舊日期。同日重跑會覆蓋當日（最新版）。"""
    date = pd.Timestamp(now, unit="s").strftime("%Y-%m-%d")
    d = SCAN_HISTORY / date
    d.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(klines, d / "klines.pkl.gz")          # dict (market,sym)->df，壓縮存
    (d / "results.json").write_text(
        json.dumps({"results": results, "now": now, "days": days}, ensure_ascii=False),
        encoding="utf-8")
    return d


def scan(days: int, limit: int | None, skip_crypto=False, skip_stock=False, min_qvol=0.0):
    now = int(time.time())
    cutoff = now - days * 86400
    results = []
    klines = {}   # (market, sym) -> df：供當日存檔與 --cached 重算（免重抓網路）

    if not skip_crypto:
        perps = get_crypto_perps(min_qvol)   # 預設 0=全掃；可用 --min-qvol 設流動性門檻
        if limit:
            perps = perps[:limit]
        print(f"[crypto] 掃描 {len(perps)} 檔永續 ...")
        for i, sym in enumerate(perps, 1):
            try:
                df = fetch_binance_klines(sym, "1d", CRYPTO_BARS)
                if len(df) < 30:
                    continue
                klines[("crypto", sym)] = df
                results += _process_symbol("crypto", sym, df, days, cutoff)
            except RuntimeError as e:   # IP ban → 中止 crypto，保留已掃結果
                print(f"  crypto 中止：{e}")
                break
            except Exception:
                pass
            if i % 50 == 0:
                print(f"  crypto {i}/{len(perps)} ... 命中 {sum(1 for r in results if r['market']=='crypto')}")
            time.sleep(0.3)   # 節流，避免 Binance 權重超限被 ban

    if not skip_stock:
        stocks = get_stock_universe()
        if limit:
            stocks = stocks[:limit]
        print(f"[stock] 掃描 {len(stocks)} 檔 ...")
        CHUNK = 50
        for c in range(0, len(stocks), CHUNK):
            chunk = stocks[c:c + CHUNK]
            try:
                data = fetch_stocks_batch(chunk, period="2y")
            except Exception as e:
                print(f"  stock chunk {c} 失敗：{e}")
                continue
            for sym, df in data.items():
                if len(df) < 30:
                    continue
                klines[("stock", sym)] = df
                results += _process_symbol("stock", sym, df, days, cutoff)
            print(f"  stock {min(c+CHUNK,len(stocks))}/{len(stocks)} ... 命中 {sum(1 for r in results if r['market']=='stock')}")
            time.sleep(0.3)

    results.sort(key=lambda r: r["sweep_time"], reverse=True)
    return results, now, klines


def scan_cached(date: str | None, days: int):
    """從 scan_history 快照重算（不連網）：用存檔的原始 K 線重跑偵測/移停/L2/組 row。
    date=None 取最新一份快照。供反覆改程式時秒級重算驗證。"""
    if not SCAN_HISTORY.exists():
        raise FileNotFoundError("尚無任何 scan_history 快照，請先正式掃描一次")
    if date is None:
        dirs = sorted(p.name for p in SCAN_HISTORY.iterdir() if (p / "klines.pkl.gz").exists())
        if not dirs:
            raise FileNotFoundError("scan_history 內無含 klines 的快照")
        date = dirs[-1]
    snap = SCAN_HISTORY / date
    klines = pd.read_pickle(snap / "klines.pkl.gz")
    meta = json.loads((snap / "results.json").read_text(encoding="utf-8")) if (snap / "results.json").exists() else {}
    now = int(meta.get("now") or pd.Timestamp(date).timestamp())   # 用快照當日時點，holding/新命中視窗才對齊
    cutoff = now - days * 86400
    print(f"[cached] 用 {date} 快照重算 {len(klines)} 檔（不連網）...")
    results = []
    for (market, sym), df in klines.items():
        try:
            results += _process_symbol(market, sym, df, days, cutoff)
        except Exception:
            pass
    results.sort(key=lambda r: r["sweep_time"], reverse=True)
    return results, now


def build_html(results, now, out_path):
    payload = json.dumps({"results": results, "now": now, "view_bars": VIEW_BARS},
                         separators=(",", ":"))
    lib = _LIB_PATH.read_text(encoding="utf-8").replace("</script>", "<\\/script>") if _LIB_PATH.exists() else ""
    html = _HTML.replace("__LIB__", lib).replace("__DATA__", payload)
    Path(out_path).write_text(html, encoding="utf-8")


_HTML = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>5035 流動性獵取 — 全市場掃描</title>
<script>__LIB__</script>
<style>
:root{color-scheme:dark;}*{box-sizing:border-box;}
body{margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:#0e1117;color:#d1d4dc;height:100vh;display:flex;flex-direction:column;}
header{padding:10px 16px;border-bottom:1px solid #222;display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
h1{font-size:15px;margin:0;}.sub{font-size:12px;color:#8b949e;}
button{background:#161b22;color:#d1d4dc;border:1px solid #30363d;border-radius:6px;padding:5px 11px;cursor:pointer;font-size:13px;}
button.on{background:#1f6feb;border-color:#1f6feb;color:#fff;}
#main{flex:1;display:flex;min-height:0;}
#chartwrap{flex:1;position:relative;min-width:0;}
#chart{position:absolute;inset:0;}
#legend{position:absolute;top:6px;left:10px;z-index:5;pointer-events:none;font:12px ui-monospace,Consolas,monospace;color:#a8acb3;text-shadow:0 0 4px #000;}
#ovl{position:absolute;inset:0;z-index:4;pointer-events:none;}
#side{width:380px;border-left:1px solid #222;overflow-y:auto;}
.row{padding:8px 12px;border-bottom:1px solid #1b1f27;cursor:pointer;font-size:12px;}
.row:hover,.row.sel{background:#161b22;}
.row .top{display:flex;gap:8px;align-items:center;}
.high{color:#ef5350;}.low{color:#26a69a;}
.tag{font-size:10px;padding:1px 5px;border-radius:3px;background:#30363d;}
.d{color:#9aa4b2;}
</style></head><body>
<header>
  <h1>🎯 流動性獵取 · 觀察/持倉/出場</h1><span class="sub" id="meta"></span>
  <span style="flex:1"></span>
  <input id="q" placeholder="🔍 標的" style="background:#161b22;color:#d1d4dc;border:1px solid #30363d;border-radius:6px;padding:4px 8px;font-size:13px;width:84px">&nbsp;
  <button class="filt on" data-m="all">全部</button><button class="filt" data-m="crypto">加密</button><button class="filt" data-m="stock">美股</button>
  &nbsp;<button class="filt2 on" data-d="all">多空</button><button class="filt2" data-d="low">B低(多)</button><button class="filt2" data-d="high">B高(空)</button>
  &nbsp;<button class="filt3 on" data-t="all">全部</button><button class="filt3" data-t="ok" title="只顯示確認的命中（加密=taker 介入；美股=放量）">⚡確認</button>
  &nbsp;<button class="filt4 on" data-s="all">全狀態</button><button class="filt4" data-s="watching">👁觀察</button><button class="filt4" data-s="holding">🟢持倉</button><button class="filt4" data-s="tp">✅停利</button><button class="filt4" data-s="exit">🔴停損/移停</button>
  &nbsp;<button id="measBtn">📏 測量</button><label class="sub"><input type="checkbox" id="log"> 對數</label>
  &nbsp;<a href="backtest.html" style="background:#161b22;color:#d1d4dc;border:1px solid #30363d;border-radius:6px;padding:5px 11px;font-size:13px;text-decoration:none;">📊 回測</a>
</header>
<div id="main"><div id="chartwrap"><div id="chart"></div><div id="legend"></div><canvas id="ovl"></canvas></div><div id="side"></div></div>
<script>
const D=__DATA__;let mF='all',dF='all',tF='all',sF='all',qF='',cur=null;
const ST={watching:['👁觀察中','#9aa4b2'],holding:['🟢持倉','#26a69a'],tp:['✅停利','#26a69a'],trail:['🟠移停出場','#ff9800'],sl:['🔴停損','#ef5350']};
const tkTxt=r=>r.taker_z==null?'taker —':`${r.taker_kind==='放量'?'量':'taker'} ${r.taker_z>=0?'+':''}${r.taker_z.toFixed(2)} ${r.taker_kind}${r.taker_ok?' ⚡':''}`;
const fmtD=t=>new Date(t*1000).toISOString().slice(0,10);
const chart=LightweightCharts.createChart(document.getElementById('chart'),{
  layout:{background:{color:'#0e1117'},textColor:'#d1d4dc'},grid:{vertLines:{color:'#1b1f27'},horzLines:{color:'#1b1f27'}},
  timeScale:{timeVisible:false,borderColor:'#30363d'},rightPriceScale:{borderColor:'#30363d'},crosshair:{mode:0}});
const candle=chart.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'});
// 價格精度自適應：低價幣(0.0026/0.04)用更多小數，否則顯示 0.00 看不到價
const precFor=p=>{if(!(p>0))return 2;const d=Math.ceil(-Math.log10(p));return Math.min(8,Math.max(2,d+3));};
const applyPrec=p=>{const pr=precFor(p);candle.applyOptions({priceFormat:{type:'price',precision:pr,minMove:Math.pow(10,-pr)}});};
chart.priceScale('right').applyOptions({scaleMargins:{top:0.05,bottom:0.22}});
let pLines=[];
function clearPLines(){for(const pl of pLines)candle.removePriceLine(pl);pLines=[];}
const volSeries=chart.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'vol',lastValueVisible:false,priceLineVisible:false});
chart.priceScale('vol').applyOptions({scaleMargins:{top:0.82,bottom:0},visible:false});
let lvlSeries=null;

// ---- 開高低收 legend + 量尺測量 ----
const legend=document.getElementById('legend'),wrap=document.getElementById('chartwrap');
const ovl=document.getElementById('ovl'),octx=ovl.getContext('2d');
let measureOn=false,mp1=null,mp2=null,mprev=null,curCandles=[];
const dec=n=>{n=Math.abs(n);return n>=100?2:n>=1?3:6;};
const fmt=n=>Number(n).toFixed(dec(n));
const fmtV=v=>v>=1e9?(v/1e9).toFixed(2)+'B':v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':v.toFixed(0);
chart.subscribeCrosshairMove(p=>{
  const d=p&&p.time?p.seriesData.get(candle):null;
  if(d&&Number.isFinite(d.open)){const up=d.close>=d.open,t=up?'#26a69a':'#ef5350';
    const v=p.seriesData.get(volSeries);
    legend.innerHTML=['開','高','低','收'].map((k,i)=>`${k} <span style="color:${t}">${fmt([d.open,d.high,d.low,d.close][i])}</span>`).join('&nbsp;&nbsp;')
      +(v&&Number.isFinite(v.value)?`&nbsp;&nbsp;量 <span style="color:${t}">${fmtV(v.value)}</span>`:'');}
  else legend.innerHTML='';
  if(measureOn&&mp1&&!mp2&&p&&p.time&&p.point){const pr=candle.coordinateToPrice(p.point.y);if(pr!=null){mprev={time:p.time,price:pr};drawOvl();}}
});
chart.subscribeClick(p=>{
  if(!measureOn||!p||!p.time||!p.point)return;const pr=candle.coordinateToPrice(p.point.y);if(pr==null)return;
  const pt={time:p.time,price:pr};
  if(!mp1||mp2){mp1=pt;mp2=null;mprev=null;}else{mp2=pt;mprev=null;}
  drawOvl();
});
chart.timeScale().subscribeVisibleLogicalRangeChange(()=>drawOvl());
// 圖表不會自己跟容器重算（v4 autoSize 在此版面失效），必須手動補——漏了圖高會凍結在建立當下：
// #meta 文字是 createChart 之後才塞進 header，把工具列擠成兩排(52→91px)，容器縮了但圖沒縮，X 軸就掉出視窗底。（同 backtest.py sizeChart）
function sizeChart(){const w=wrap.clientWidth,h=wrap.clientHeight;if(w&&h)chart.applyOptions({width:w,height:h});}
function sizeOvl(){const r=wrap.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  ovl.width=r.width*dpr;ovl.height=r.height*dpr;ovl.style.width=r.width+'px';ovl.style.height=r.height+'px';
  octx.setTransform(dpr,0,0,dpr,0,0);drawOvl();}
new ResizeObserver(()=>{sizeChart();sizeOvl();}).observe(wrap);
addEventListener('load',sizeChart);   // 保險：字型/文字完全落定後再對一次
const countBars=(t1,t2)=>{const lo=Math.min(t1,t2),hi=Math.max(t1,t2);return curCandles.filter(c=>c.time>=lo&&c.time<=hi).length;};
function drawOvl(){
  octx.clearRect(0,0,ovl.width,ovl.height);
  const a=mp1,b=mp2||mprev;if(!a||!b)return;
  const ts=chart.timeScale();
  const x1=ts.timeToCoordinate(a.time),x2=ts.timeToCoordinate(b.time),y1=candle.priceToCoordinate(a.price),y2=candle.priceToCoordinate(b.price);
  if(x1==null||x2==null||y1==null||y2==null)return;
  const xL=Math.min(x1,x2),xR=Math.max(x1,x2),yT=Math.min(y1,y2),yB=Math.max(y1,y2),up=b.price>=a.price;
  octx.fillStyle=up?'rgba(38,166,154,.18)':'rgba(239,83,80,.18)';octx.fillRect(xL,yT,xR-xL,yB-yT);
  octx.strokeStyle=up?'rgba(38,166,154,.9)':'rgba(239,83,80,.9)';octx.lineWidth=1;
  octx.setLineDash(mp2?[]:[6,4]);octx.strokeRect(xL,yT,xR-xL,yB-yT);
  octx.beginPath();octx.moveTo(x1,y1);octx.lineTo(x2,y2);octx.stroke();octx.setLineDash([]);
  const dP=b.price-a.price,sg=dP>=0?'+':'',pct=a.price?dP/a.price*100:0,bars=countBars(a.time,b.time);
  const lines=[`${sg}${fmt(dP)}  ${sg}${pct.toFixed(2)}%`,`${bars} 根`];
  octx.font='12px ui-monospace,Consolas,monospace';octx.textAlign='center';octx.textBaseline='middle';
  let mw=0;for(const l of lines)mw=Math.max(mw,octx.measureText(l).width);
  const lh=15,th=lines.length*lh,cx=(xL+xR)/2,cy=(yT+yB)/2;
  octx.fillStyle='rgba(0,0,0,.8)';octx.fillRect(cx-mw/2-5,cy-th/2-5,mw+10,th+10);
  octx.fillStyle=up?'#86efac':'#fca5a5';let ty=cy-th/2+lh/2;for(const l of lines){octx.fillText(l,cx,ty);ty+=lh;}
}
document.getElementById('measBtn').onclick=function(){measureOn=!measureOn;this.classList.toggle('on',measureOn);
  if(!measureOn){mp1=mp2=mprev=null;drawOvl();}};
window.addEventListener('keydown',e=>{if(e.key==='Escape'){mp1=mp2=mprev=null;drawOvl();}});
function show(r){
  cur=r;document.querySelectorAll('.row').forEach(x=>x.classList.toggle('sel',x.dataset.id===r._id));
  chart.priceScale('right').applyOptions({autoScale:true});  // 換標的恢復自動縮放（使用者縮放過會被鎖死→新標的空白）
  const ch=r.chart;
  const candles=ch.candles;
  applyPrec(r.level||(candles.length?candles[candles.length-1].close:1));   // 依此標的價格量級設小數位
  candle.setData(candles);curCandles=ch.candles;mp1=mp2=mprev=null;
  clearPLines();   // 交易計畫線：停損(持倉中=當前結構移停水位) + TP(下一個流動性聚集區)
  const stopPx=r.cur_stop!=null?r.cur_stop:r.stop;
  if(stopPx!=null)pLines.push(candle.createPriceLine({price:stopPx,color:'#ef5350',lineWidth:1,lineStyle:2,axisLabelVisible:true,title:r.armed?'停損(結構移停中)':'停損(初始)'}));
  if(r.tgt!=null)pLines.push(candle.createPriceLine({price:r.tgt,color:'#26a69a',lineWidth:1,lineStyle:2,axisLabelVisible:true,title:'TP·下一個流動性聚集區'}));
  volSeries.setData(ch.candles.map(c=>({time:c.time,value:c.volume||0,
    color:c.close>=c.open?'rgba(38,166,154,.45)':'rgba(239,83,80,.45)'})));
  if(lvlSeries)chart.removeSeries(lvlSeries);
  lvlSeries=chart.addLineSeries({color:r.side==='high'?'rgba(239,83,80,.8)':'rgba(66,165,245,.8)',lineWidth:2,lastValueVisible:false,priceLineVisible:false});
  lvlSeries.setData([{time:ch.candles[0].time,value:ch.level},{time:ch.candles[ch.candles.length-1].time,value:ch.level}]);
  const mk=[];
  for(const tt of ch.touch_times){const cd=ch.candles.find(c=>c.time===tt);if(cd)mk.push({time:tt,position:r.side==='high'?'aboveBar':'belowBar',color:'#8b949e',shape:'circle',text:'測'});}
  const tkMark=r.taker_z==null?'':` ${r.taker_ok?'⚡':''}${r.taker_z>=0?'+':''}${r.taker_z.toFixed(2)}`;
  mk.push({time:ch.sweep_time,position:r.side==='high'?'aboveBar':'belowBar',color:r.side==='high'?'#ef5350':'#26a69a',shape:r.side==='high'?'arrowDown':'arrowUp',text:'獵取'+tkMark});
  if(r.exit_time){   // 已出場：標出場點（止盈在獲利側、停損/移停在風險側）
    const lbl={tp:'✓止盈@聚集區',trail:'移停出場',sl:'✕停損'}[r.exit_kind]||'出場';
    const col={tp:'#26a69a',trail:'#ff9800',sl:'#ef5350'}[r.exit_kind]||'#8b949e';
    const above=(r.exit_kind==='tp')===(r.side==='low');
    mk.push({time:r.exit_time,position:above?'aboveBar':'belowBar',color:col,shape:'square',text:lbl});
  }
  const seen=new Set();candle.setMarkers(mk.sort((a,b)=>a.time-b.time).filter(m=>seen.has(m.time)?false:(seen.add(m.time),true)));
  chart.priceScale('right').applyOptions({mode:document.getElementById('log').checked?1:0});
  // 預設視野：最後 view_bars 根；觸碰點更早時延伸到第一個觸碰前 10 根；出場參考在未來時涵蓋它
  const N=ch.candles.length;
  let from=N-D.view_bars;
  if(ch.touch_times.length){
    const ti=ch.candles.findIndex(c=>c.time===ch.touch_times[0]);
    if(ti>=0)from=Math.min(from,ti-10);
  }
  chart.timeScale().setVisibleLogicalRange({from:Math.max(0,from),to:N+3});
  drawOvl();
}
function render(){
  let rows=D.results.filter(r=>(mF==='all'||r.market===mF)&&(dF==='all'||r.side===dF)&&(tF==='all'||r.taker_ok===true)
    &&(sF==='all'||(sF==='exit'?(r.status==='sl'||r.status==='trail'):r.status===sF))
    &&(qF===''||r.symbol.toUpperCase().includes(qF)));
  const nNew=rows.filter(r=>r.is_new).length;
  const nW=rows.filter(r=>r.status==='watching').length,nH=rows.filter(r=>r.status==='holding').length,nTp=rows.filter(r=>r.status==='tp').length,nEx=rows.filter(r=>r.status==='sl'||r.status==='trail').length;
  document.getElementById('meta').textContent=`掃描 ${fmtD(D.now)} · 共 ${rows.length}（觀察 ${nW} / 持倉 ${nH} / 停利 ${nTp} / 停損·移停 ${nEx} · 新 ${nNew}）`;
  const s=document.getElementById('side');s.innerHTML='';
  rows.forEach((r,i)=>{r._id='r'+i;const div=document.createElement('div');div.className='row';div.dataset.id=r._id;
    const cls=r.side==='high'?'high':'low';
    const st=ST[r.status]||ST.holding;
    const stTag=`<span class="tag" style="background:${st[1]}33;color:${st[1]}">${st[0]}</span>`;
    const newTag=r.is_new?'<span class="tag" style="background:#1f6feb;color:#fff">新</span>':'';
    const heldTxt=r.status==='watching'?' · 觀察中(待站回確認/破針尖放棄)':
      r.status==='holding'?(r.held_days!=null?` · 持倉${r.held_days}日`:''):
      (r.exit_time!=null?` · 持${r.held_days}日→${fmtD(r.exit_time)}出場`:'');
    const armed=(r.status==='holding'&&r.armed)?' · <span style="color:#ff9800">結構移停中</span>':'';
    div.innerHTML=`<div class="top"><span class="tag">${r.market==='crypto'?'加密':'美股'}</span><b>${r.symbol}</b><span class="${cls}">${r.dir}</span>${stTag}${newTag}<span class="d">${fmtD(r.sweep_time)}${heldTxt}</span></div>`+
      `<div class="d">S/R ${r.level} · 插破至 ${r.extreme} · 觸碰${r.touches} · ${tkTxt(r)} · 現價 ${r.last_close}${armed}</div>`;
    div.onclick=()=>show(r);s.appendChild(div);});
  if(rows.length)show(rows[0]);
}
document.querySelectorAll('.filt').forEach(b=>b.onclick=()=>{mF=b.dataset.m;document.querySelectorAll('.filt').forEach(x=>x.classList.toggle('on',x===b));render();});
document.querySelectorAll('.filt2').forEach(b=>b.onclick=()=>{dF=b.dataset.d;document.querySelectorAll('.filt2').forEach(x=>x.classList.toggle('on',x===b));render();});
document.querySelectorAll('.filt3').forEach(b=>b.onclick=()=>{tF=b.dataset.t;document.querySelectorAll('.filt3').forEach(x=>x.classList.toggle('on',x===b));render();});
document.querySelectorAll('.filt4').forEach(b=>b.onclick=()=>{sF=b.dataset.s;document.querySelectorAll('.filt4').forEach(x=>x.classList.toggle('on',x===b));render();});
document.getElementById('q').oninput=e=>{qF=e.target.value.trim().toUpperCase();render();};
document.getElementById('log').onchange=()=>{if(cur)chart.priceScale('right').applyOptions({mode:document.getElementById('log').checked?1:0});};
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="scanner.html")
    ap.add_argument("--skip-crypto", action="store_true")
    ap.add_argument("--skip-stock", action="store_true")
    ap.add_argument("--notify", action="store_true", help="跟上次比對，推播新增命中")
    ap.add_argument("--min-qvol", type=float, default=0.0,
                    help="crypto 24h 成交額門檻(USDT)；0=全掃527檔，5e6=只掃流動性≥5M(207檔)")
    ap.add_argument("--cached", nargs="?", const="", default=None, metavar="DATE",
                    help="用 scan_history 快照重算(不連網,秒級)；可帶日期 YYYY-MM-DD，省略=最新一份。改程式驗證用")
    args = ap.parse_args()
    out = Path(__file__).parent / args.out
    if args.cached is not None:   # 快取重算模式：不連網、不存檔、不推播
        t0 = time.time()
        results, now = scan_cached(args.cached or None, args.days)
        build_html(results, now, str(out))
        print(f"\n[cached] 重算完成 B 型 {len(results)} 筆，耗時 {time.time()-t0:.1f}s；前端：{out}")
        return
    if args.notify and (args.limit or args.skip_crypto or args.skip_stock):
        print("部分掃描(--limit/--skip-*)會汙染推播基準，已自動停用 --notify")
        args.notify = False
    t0 = time.time()
    results, now, klines = scan(args.days, args.limit, args.skip_crypto, args.skip_stock, args.min_qvol)
    build_html(results, now, str(out))
    if not (args.limit or args.skip_crypto or args.skip_stock):   # 僅完整掃描才存檔（部分掃描不污染歷史）
        snap = _archive(now, args.days, results, klines)
        print(f"已存檔：{snap}")
    if args.notify:
        notify_new(results, now)
    n_c = sum(1 for r in results if r["market"] == "crypto")
    n_s = sum(1 for r in results if r["market"] == "stock")
    n_new = sum(1 for r in results if r.get("is_new"))
    n_watch = sum(1 for r in results if r.get("status") == "watching")
    n_hold = sum(1 for r in results if r.get("status") == "holding")
    n_tp = sum(1 for r in results if r.get("status") == "tp")
    n_exit = sum(1 for r in results if r.get("status") in ("sl", "trail"))
    print(f"\n完成：B 型 {len(results)} 筆（加密 {n_c} / 美股 {n_s}；觀察 {n_watch} / 持倉 {n_hold} / 近{args.days}天停利 {n_tp} / 停損·移停 {n_exit}；新命中 {n_new}），耗時 {time.time()-t0:.0f}s")
    print(f"前端清單：{out}")


if __name__ == "__main__":
    main()
