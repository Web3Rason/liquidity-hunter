"""
B 型流動性獵取訊號回測（訊號純度驗證版）：驗證訊號本身與 taker/放量 ⚡ 過濾的真實價值。

設計（經第二輪架構審查修正後）：
- 樣本：Binance USDT 永續（現存、24h>=5M）完整日線 + 美股 S&P500+NASDAQ100 五年日線
  → 倖存者偏差：已下架/被踢出指數者不在樣本，多單偏樂觀，解讀須記住
- 進場：收復根的「次根開盤」（排程在收盤後才看到訊號，用收盤進場是作弊）；
  次根開盤偏離收復收盤 > 1 ATR → 跳空吃不到，剔除（計數）
- 出場：時間出場 N∈{3,5,10,20} 根後收盤——只驗訊號漂移，不摻停損設計變數
- 報酬：ATR(14) 標準化（賺幾倍 ATR），防高波動山寨幣綁架平均；成本換算進 ATR 單位
- MFE/MAE：進場後 10 根內最大有利/不利波動（ATR 單位），衡量訊號爆發力
- 防自相關：同標的同方向訊號 20 根冷卻期，期內新訊號丟棄
- 分組：⚡ vs 非⚡（核心）、吸收/反轉/放量、touches、多空、市場、MA200 順逆勢、時代(≤2022/2023+)
- 統計：n/勝率/平均/中位/Sharpe(訊號)/PF + ⚡vs非⚡ bootstrap 95% CI
- 注意：taker 門檻(ALPHA=0.12)的校準用過 2023~2026 BTC/ETH/SOL，「2023+」組有樣本內成分，
  「≤2022」組才是乾淨的樣本外檢驗
- 執行：單一程序序列；K 線快取 .cache_klines/（24h 內重跑不重抓）

用法：
    python backtest.py                    # 全樣本
    python backtest.py --limit 30         # 每市場前 N 檔（測試）
    python backtest.py --skip-stock
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import fetch_binance_klines, fetch_stocks_batch
from detector import _swings, detect_sweeps, get_config
from taker import attach_taker
from taker_1m import m1_confirm, m1_window_confirm
from universe import get_crypto_perps, get_stock_universe

CACHE = Path(__file__).parent / ".cache_klines"
CRYPTO_BARS_FULL = 3000          # ~8 年（幣安 USDT 永續最早 2019-09）
COST_RT = {"crypto": 0.001, "stock": 0.0005}   # 來回成本（佔進場價比例）
TIME_EXITS = [3, 5, 10, 20]
MFE_WINDOW = 10
COOLDOWN = 20                    # 同標的同方向冷卻期（根）
_RECLAIM_OVERRIDE = None         # 掃描用：覆寫 reclaim_enter（None=用 cfg 預設）
MIN_TOUCHES = 2                  # 與每日監控一致
_TRAIL_MODE = "localhigh"        # 移停模式(預設正式版)：localhigh=swing-low 結構移停 / retest=回踩收復線(舊v5,對照用)
                                 #   localhigh(使用者準則)：沒 lower low 就一路跟，higher-low 出現→SL 上移到前一個低點


def get_crypto_df(sym):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{sym}_1d.pkl"
    if f.exists() and time.time() - f.stat().st_mtime < 86400:
        return pd.read_pickle(f)
    df = fetch_binance_klines(sym, "1d", CRYPTO_BARS_FULL)
    df.to_pickle(f)
    time.sleep(0.3)              # 節流
    return df


def get_stock_dfs(symbols):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "stocks_5y.pkl"
    if f.exists() and time.time() - f.stat().st_mtime < 86400:
        return pd.read_pickle(f)
    out = {}
    CHUNK = 50
    for c in range(0, len(symbols), CHUNK):
        out.update(fetch_stocks_batch(symbols[c:c + CHUNK], period="5y"))
        time.sleep(0.3)
    pd.to_pickle(out, f)
    return out


def _atr(df, n=14):
    h, lo, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


def run_symbol(df, sym, market):
    cfg = get_config("1d", market)
    if _RECLAIM_OVERRIDE is not None:
        cfg = {**cfg, "reclaim_enter": _RECLAIM_OVERRIDE}
    sigs = attach_taker(df, detect_sweeps(df, cfg))
    sigs = [s for s in sigs if s["type"] == "B" and s["touches"] >= MIN_TOUCHES]
    if not sigs:
        return [], {"gap": 0, "cooldown": 0, "no_atr": 0}
    atr = _atr(df)
    ma200 = df["close"].rolling(200).mean().values
    o = df["open"].values
    h = df["high"].values
    lo = df["low"].values
    c = df["close"].values
    n = len(df)
    lr = cfg["pivot_lr"]
    sw_h, sw_l = _swings(h, lo, lr)   # 結構移停的 swing high/low 階梯
    reclaim_enter = cfg.get("reclaim_enter", 0.0)   # 強收復門檻(站回幾倍ATR)；0=關閉分流(全部立即進場)
    watch_bars = cfg.get("watch_bars", 3)           # 弱收復進觀察，最多看幾根等站回達門檻
    rows = []
    skips = {"gap": 0, "cooldown": 0, "no_atr": 0, "unconfirmed": 0}
    last_idx = {}                              # side -> 上次成交訊號 idx（冷卻期用）
    for s in sorted(sigs, key=lambda x: x["sweep_idx"]):
        i = s["sweep_idx"]
        side = s["side"]
        if i + 1 >= n:
            continue                           # 還沒有次根，無法進場（最新訊號）
        a_sig = atr[i]
        if not np.isfinite(a_sig) or a_sig <= 0:
            skips["no_atr"] += 1
            continue
        lvl = s["level_price"]
        tip = float(s["extreme"])
        # 進場分流（依收復站回流動線的力道，不洩漏：每根都用「該根收盤」決定「次根」進場）：
        #   強收復(站回 ≥ reclaim_enter ATR)＝當下確認獵取 → 次根進場；
        #   弱收復＝進觀察，往後最多 watch_bars 根：某根收盤站回達門檻 → 該根次根進場；
        #            收盤跌破針尖 → 獵取假設證偽，放棄。
        def _reclaim(k, ak):
            return (c[k] - lvl) / ak if side == "low" else (lvl - c[k]) / ak
        if _reclaim(i, a_sig) >= reclaim_enter:
            ie = i                              # 強收復，立即(次根)進場
        else:
            ie = None
            for k in range(i + 1, min(n, i + 1 + watch_bars)):
                if (c[k] < tip) if side == "low" else (c[k] > tip):
                    break                       # 收盤破針尖＝證偽，放棄
                ak = atr[k]
                if np.isfinite(ak) and ak > 0 and _reclaim(k, ak) >= reclaim_enter:
                    ie = k                      # 觀察期內站回達門檻＝確認獵取
                    break
            if ie is None:
                skips["unconfirmed"] += 1
                continue
        if ie + 1 >= n:
            continue                            # 進場次根不足（觀察到資料末，待確認）
        if side in last_idx and ie - last_idx[side] < COOLDOWN:
            skips["cooldown"] += 1
            continue
        a = atr[ie]                             # 報酬/停損標準化用進場前一根 ATR(=ie)，不洩漏
        if not np.isfinite(a) or a <= 0:
            skips["no_atr"] += 1
            continue
        entry = o[ie + 1]
        if abs(entry - c[ie]) > a:              # 跳空 > 1 ATR，實務吃不到
            skips["gap"] += 1
            continue
        last_idx[side] = ie
        sgn = 1.0 if side == "low" else -1.0
        cost_atr = COST_RT[market] * entry / a
        if market == "crypto":   # crypto ⚡ 改由 1m 判定（快取齊備後離線；首跑需網路）
            j0 = s["dev_idx"]
            if int(s.get("rearm_level", 1)) == 2:
                # L2 放寬閘（使用者裁決）：掃插破→收復整段窗口，非僅極值/收復兩節點
                r1m = m1_window_confirm(sym, side,
                                        int(df["time"].iloc[j0]) // 86400 * 86400,
                                        int(s["sweep_time"]) // 86400 * 86400)
            else:
                ed = j0 + (int(np.argmin(lo[j0:i + 1])) if side == "low"
                           else int(np.argmax(h[j0:i + 1])))
                r1m = m1_confirm(sym, side, s["level_price"],
                                 int(df["time"].iloc[ed]) // 86400 * 86400,
                                 int(s["sweep_time"]) // 86400 * 86400)
            if r1m["ok"] is not None:
                s["taker_ok"] = bool(r1m["ok"])
                s["taker_kind"] = f"1m{r1m['kind']}" if r1m["ok"] else s["taker_kind"]
        r = {"market": market, "symbol": sym, "side": side,
             "touches": s["touches"], "taker_ok": s["taker_ok"],
             "rearm_level": int(s.get("rearm_level", 1)),   # 1=第一級；2=線復活的更深第二級
             "taker_kind": s["taker_kind"], "taker_z": s["taker_z"],
             "year": pd.Timestamp(int(s["sweep_time"]), unit="s").year,
             "sweep_time": int(s["sweep_time"]),
             "birth_tan": int(s.get("birth_tan", 0)),
             "with_trend": bool(np.isfinite(ma200[ie]) and
                                ((c[ie] > ma200[ie]) == (side == "low"))),
             # 案例圖表用（不進 CSV）
             "sweep_idx": i, "entry_idx": ie, "level": s["level_price"], "touch_idxs": list(s["touch_idxs"])}
        for N in TIME_EXITS:
            j = ie + N                         # 進場根=ie+1，持有 N 根 → 出在第 ie+N 根收盤
            r[f"t{N}"] = sgn * (c[j] - entry) / a - cost_atr if j < n else None
        # v5 retest 移停（使用者交易計畫，依作者「取 Stop Hunt 發生位置」）：
        # 初始 SL=插破極值(跌破/突破流動線後的最低/最高點=作者「針尖高/低點」，非收復根低點)；
        # TP=下一個流動性聚集區(opp_level)；
        # 途中只有「回踩到收復線(level)附近的 higher swing low(多)/lower swing high(空)」= retest
        # 才把 SL 上移到該結構點（過濾盤整雜訊小回踩），抱向 TP。
        # 出場：碰 TP=tp、SL 掃出且移過=trail(retest 後鎖利潤)、初始 SL 掃出=sl。
        # 同根先檢查 SL(保守)；跳空按開盤價；swing 確認延遲 lr 根，無前視。
        stop0 = tip                                # =s["extreme"] 針尖，已於進場分流段取得
        tgt = s.get("opp_level")
        near = lvl * cfg["sep_pct"]                # retest 須回踩到收復線 ±sep_pct 內
        r["stop_px"] = round(stop0, 6)
        r["tgt_px"] = tgt
        r["ts_pnl"] = None
        r["ts_bars"] = None
        r["ts_kind"] = None
        if tgt is None:
            r["ts_kind"] = "no_target"             # 無聚集區目標=無出場計畫，不混入統計
        elif sgn * (tgt - entry) <= 0:
            r["ts_kind"] = "invalid"               # 目標已在進場價內側，計畫不成立
        else:
            stop_cur = stop0
            prev_sw = None   # localhigh：上一個已確認 swing low(多)/high(空)；higher-low 出現才上移 SL 到它
            for j in range(ie + 1, n):
                if (lo[j] <= stop_cur) if side == "low" else (h[j] >= stop_cur):
                    exit_px = min(o[j], stop_cur) if side == "low" else max(o[j], stop_cur)
                    r["ts_pnl"] = sgn * (exit_px - entry) / a - cost_atr
                    r["ts_bars"] = j - ie
                    r["ts_kind"] = "sl" if stop_cur == stop0 else "trail"
                    break
                if (h[j] >= tgt) if side == "low" else (lo[j] <= tgt):
                    exit_px = max(o[j], tgt) if side == "low" else min(o[j], tgt)
                    r["ts_pnl"] = sgn * (exit_px - entry) / a - cost_atr
                    r["ts_bars"] = j - ie
                    r["ts_kind"] = "tp"
                    break
                if _TRAIL_MODE == "localhigh":
                    # 使用者準則：兩個 low 之間最高點=local high；只要沒 lower low(維持高低點墊高)就一路跟。
                    # 每出現一個更高的 swing low → SL 上移到「前一個 swing low」(=該 local high 前的低點)；
                    # 出現 lower low → 只更新基準不上移(結構重設)。無前視：swing 由 pi=j-lr 已確認的點取。
                    pi = j - lr
                    if pi > ie and (sw_l[pi] if side == "low" else sw_h[pi]):
                        cur = lo[pi] if side == "low" else h[pi]
                        if prev_sw is not None:
                            if side == "low" and cur > prev_sw and prev_sw > stop_cur:
                                stop_cur = float(prev_sw)         # higher-low → 鎖到前一個低點
                            elif side == "high" and cur < prev_sw and prev_sw < stop_cur:
                                stop_cur = float(prev_sw)         # lower-high → 鎖到前一個高點
                        prev_sw = cur
                else:
                    pi = j - lr   # retest：回踩到收復線附近的 higher swing low / lower swing high → SL 上移
                    if pi > ie:
                        if side == "low" and sw_l[pi] and lo[pi] > stop_cur and abs(lo[pi] - lvl) <= near:
                            stop_cur = float(lo[pi])
                        elif side == "high" and sw_h[pi] and h[pi] < stop_cur and abs(h[pi] - lvl) <= near:
                            stop_cur = float(h[pi])
        j2 = min(n, ie + 1 + MFE_WINDOW)       # MFE/MAE 窗口 = 進場根起 10 根（含進場根）
        hh = h[ie + 1:j2]
        ll = lo[ie + 1:j2]
        if len(hh):
            r["mfe"] = (hh.max() - entry) / a if sgn > 0 else (entry - ll.min()) / a
            r["mae"] = (entry - ll.min()) / a if sgn > 0 else (hh.max() - entry) / a
        rows.append(r)
    return rows, skips


def collect(args):
    recs = []
    tot_skips = {"gap": 0, "cooldown": 0, "no_atr": 0}

    def acc(sk):
        for k in tot_skips:
            tot_skips[k] += sk[k]

    if not args.skip_crypto:
        syms = get_crypto_perps()
        if args.limit:
            syms = syms[:args.limit]
        print(f"[crypto] {len(syms)} 檔 ...")
        for k, sym in enumerate(syms, 1):
            try:
                df = get_crypto_df(sym)
            except Exception as e:
                print(f"  {sym} 抓取失敗：{e}")
                continue
            if len(df) < 100:
                continue
            rows, sk = run_symbol(df, sym, "crypto")
            recs += rows
            acc(sk)
            if k % 25 == 0:
                print(f"  crypto {k}/{len(syms)} ... 累積 {len(recs)} 筆")
    if not args.skip_stock:
        syms = get_stock_universe()
        if args.limit:
            syms = syms[:args.limit]
        print(f"[stock] {len(syms)} 檔 ...")
        data = get_stock_dfs(syms)
        for sym, df in data.items():
            if len(df) < 100:
                continue
            rows, sk = run_symbol(df, sym, "stock")
            recs += rows
            acc(sk)
        print(f"  stock 完成 ... 累積 {len(recs)} 筆")
    return pd.DataFrame(recs), tot_skips


def stat_line(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    wins, losses = a[a > 0], a[a <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    shp = a.mean() / a.std() if a.std() > 0 else float("nan")
    return {"n": len(a), "win": (a > 0).mean(), "avg": a.mean(),
            "med": float(np.median(a)), "shp": shp, "pf": pf}


def fmt(st):
    if st is None:
        return "n=0"
    return (f"n={st['n']:>5}  勝率{st['win']*100:5.1f}%  平均{st['avg']:+.3f}ATR  "
            f"中位{st['med']:+.3f}  Sharpe{st['shp']:+.3f}  PF{st['pf']:5.2f}")


def bootstrap_diff(a, b, iters=5000, seed=35):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
             for _ in range(iters)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def report(df, skips, out_lines):
    def w(line=""):
        print(line)
        out_lines.append(line)

    rules = [f"t{N}" for N in TIME_EXITS]
    w(f"# 5035 B 型訊號回測報告（{pd.Timestamp.now():%Y-%m-%d %H:%M}）")
    w()
    w(f"成交訊號 {len(df)}（crypto {(df.market=='crypto').sum()} / stock {(df.market=='stock').sum()}）"
      f"；剔除：跳空>{1}ATR {skips['gap']}、冷卻期 {skips['cooldown']}、ATR未成形 {skips['no_atr']}")
    w(f"⚡ {(df.taker_ok == True).sum()} / 非⚡ {(df.taker_ok == False).sum()} / taker無值 {df.taker_ok.isna().sum()}")
    w("報酬單位 = ATR(14) 倍數（已扣成本：crypto 0.1%、美股 0.05% 來回；"
      "⚠ crypto 未計 funding，20 日持有約再 -0.6%，絕對 PF 偏樂觀）；進場 = 收復根次根開盤。")
    w("倖存者偏差：樣本為現存標的，多單偏樂觀。")
    w("⚠ 樣本內校準聲明：taker ALPHA 用過 2023~2026 BTC/ETH/SOL；幾何參數"
      "（max_acc_dev/reclaim_atr/max_birth_tan）亦以全樣本+標注案例校準——headline 數字非樣本外。")
    w("⚠ 統計註記：同日多標的訊號（宏觀事件聚集，單日最多 10 筆）非獨立樣本，"
      "bootstrap CI 偏窄；肥尾明顯（|t20|>10ATR 共數十筆），平均數結論看中位數佐證。")
    w("⚠ crypto 1m⚡ 的回測增量「尚無法統計證明」（n 小、CI 跨零、尾部驅動）——"
      "採用 1m 判定的依據是方法論定義（作者實際做法），非回測績效。")
    w()

    w("## 時間出場（訊號漂移參考·非交易績效——固定持有 N 根看純價格漂移；下方各分組已改用 v5 結構移停）")
    for rule in rules:
        w(f"  {rule:>4} | {fmt(stat_line(df[rule]))}")
    if "mfe" in df:
        w(f"  MFE/MAE(10根) | 平均 {df.mfe.mean():+.3f} / {df.mae.mean():+.3f} ATR"
          f"（比值 {df.mfe.mean()/max(df.mae.mean(),1e-9):.2f}，>1 = 有利波動較大）")
    w()

    w("## ★ v5 結構移停框架（主框架=實際交易計畫，勝率/PF 以此為準）")
    w("##   初始 SL=插破極值(針尖高/低點)；TP=下一個流動性聚集區；途中回踩收復線±near 的 swing low/high(retest) 才把 SL 上移，抱向 TP")
    st_groups = [
        ("做多 全部", (df.side == "low")),
        ("做多 crypto·1m⚡", (df.side == "low") & (df.market == "crypto") & (df.taker_ok == True)),
        ("做多 crypto·✗", (df.side == "low") & (df.market == "crypto") & (df.taker_ok == False)),
        ("做多 stock·放量⚡", (df.side == "low") & (df.market == "stock") & (df.taker_ok == True)),
        ("做多 stock·✗", (df.side == "low") & (df.market == "stock") & (df.taker_ok == False)),
        ("做空 全部", (df.side == "high")),
        # 線復活第二級（驗收：閘3 強制 taker 後才可交易，故 L2·⚡ 才是真正採計組）
        ("L2 做多 全部", (df.side == "low") & (df.rearm_level == 2)),
        ("L2 做多·⚡(可交易)", (df.side == "low") & (df.rearm_level == 2) & (df.taker_ok == True)),
        ("L2 做空 全部", (df.side == "high") & (df.rearm_level == 2)),
        ("L2 做空·⚡(可交易)", (df.side == "high") & (df.rearm_level == 2) & (df.taker_ok == True)),
    ]
    for label, mask in st_groups:
        sub = df[mask]
        op = sub["ts_pnl"].isna().sum()
        bars = sub["ts_bars"].dropna().mean()
        kc = sub["ts_kind"].value_counts()
        tp, tr, sl = int(kc.get("tp", 0)), int(kc.get("trail", 0)), int(kc.get("sl", 0))
        nt = int(kc.get("no_target", 0)) + int(kc.get("invalid", 0))
        done = tp + tr + sl
        reach = tp / done if done else float("nan")
        w(f"    {label:<16} {fmt(stat_line(sub['ts_pnl']))}"
          f"  到聚集區{reach:.0%}(tp{tp}/移停{tr}/初損{sl})  無目標{nt}  均持有{bars:.1f}根  未完成{op}")
    w("  ⚠ tp=漲到聚集區止盈 / trail=結構移停掃出(SL已上移、鎖部分利潤或減損) / sl=初始停損(沒漲就跌)；")
    w("  ⚠ 未完成剔除偏砍「抱到資料尾的大贏家」，數字偏保守；同根 SL/TP 同觸算 SL；跳空按開盤價。")
    w()

    era = np.where(df.year <= 2022, "≤2022(樣本外)", "2023+(含校準期)")
    # ⚡ 在 crypto(1m判定) 與 stock(放量) 語意不同，混合呈現會誤導 → 拆市場
    conf_split = np.where(df.market == "crypto",
                          np.where(df.taker_ok == True, "crypto·1m⚡",
                                   np.where(df.taker_ok == False, "crypto·✗", "crypto·—")),
                          np.where(df.taker_ok == True, "stock·放量⚡",
                                   np.where(df.taker_ok == False, "stock·✗", "stock·—")))
    kind_conf = np.where(df.taker_ok == True, df.taker_kind.fillna(""), "")
    groupings = [
        ("確認 × 市場（crypto=1m 精準判定 / stock=日線放量，語意不同不可混讀）", conf_split),
        ("確認類型（僅 ⚡ 列；✗ 列的 kind 是被否決前的日線標籤，不列）",
         np.where(kind_conf != "", kind_conf, "（非⚡）")),
        ("觸碰數", np.where(df.touches >= 3, "touches>=3", "touches=2")),
        ("方向", df.side.map({"low": "做多(B低)", "high": "做空(B高)"})),
        ("線復活級別（L2=殭屍線被更深獵取一次後的第二級）", df.rearm_level.map({1: "L1 第一級", 2: "L2 復活第二級"})),
        ("市場", df.market),
        ("MA200 順勢", df.with_trend.map({True: "順勢", False: "逆勢"})),
        ("時代", era),
    ]
    w("（以下各分組指標 = v5 結構移停淨損益 ts，即實際交易計畫；勝率=淨損益>0 比例）")
    for title, key in groupings:
        w(f"## {title}")
        for g, sub in df.groupby(key):
            w(f"    {str(g):<12} {fmt(stat_line(sub['ts_pnl']))}")
        w()

    w("## ⚡ vs 非⚡ 結構移停淨損益差（v5 ts，ATR，bootstrap 95% CI，正=⚡較好）")
    for era_name in ["全部", "≤2022(樣本外)", "2023+(含校準期)"]:
        sub = df if era_name == "全部" else df[era == era_name]
        a = sub[sub.taker_ok == True]["ts_pnl"].dropna().tolist()
        b = sub[sub.taker_ok == False]["ts_pnl"].dropna().tolist()
        if len(a) >= 30 and len(b) >= 30:
            lo_ci, hi_ci = bootstrap_diff(a, b)
            sig = "✓顯著" if lo_ci > 0 else ("✗顯著為負" if hi_ci < 0 else "不顯著")
            w(f"  [{era_name}] 差 {np.mean(a)-np.mean(b):+.3f}ATR "
              f"CI[{lo_ci:+.3f}, {hi_ci:+.3f}] {sig} (n={len(a)}/{len(b)})")
        else:
            w(f"  [{era_name}] 樣本不足 (n={len(a)}/{len(b)})")


# ---------- 互動前端（backtest.html）：統計 + 抽樣案例 ----------

def _clean(st):
    """stat dict → JSON 安全（nan→None、取整）。"""
    if st is None:
        return None
    out = {}
    for k, v in st.items():
        if isinstance(v, float):
            out[k] = None if not np.isfinite(v) else round(v, 4)
        else:
            out[k] = int(v) if k == "n" else v
    return out


def build_stats(df):
    rules = [f"t{N}" for N in TIME_EXITS]
    overall = [{"rule": r, **(_clean(stat_line(df[r])) or {})} for r in rules]

    def grp(title, key, rule="ts_pnl"):
        rows = []
        for g, sub in df.groupby(key):
            st = _clean(stat_line(sub[rule]))
            if st:
                rows.append({"name": str(g), **st})
        return {"title": f"{title}（v5移停）", "rows": rows}

    L = df[df.side == "low"]
    pockets = []

    def pocket(name, sub):
        st = _clean(stat_line(sub["ts_pnl"]))   # v5 結構移停淨損益（實際交易計畫）
        if st:
            pockets.append({"name": name, **st})

    pocket("做多 全部", L)
    pocket("做多 stock·放量⚡", L[(L.taker_ok == True) & (L.market == "stock")])
    pocket("做多 crypto·1m⚡ (n小,增量未證)", L[(L.taker_ok == True) & (L.market == "crypto")])
    pocket("做多 crypto·1m✗", L[(L.taker_ok == False) & (L.market == "crypto")])
    pocket("做多 ⚡+touches≥3", L[(L.taker_ok == True) & (L.touches >= 3)])
    pocket("做多 ≤2022樣本外 ⚡", L[(L.year <= 2022) & (L.taker_ok == True)])
    pocket("做空 crypto", df[(df.side == "high") & (df.market == "crypto")])
    pocket("做空 stock", df[(df.side == "high") & (df.market == "stock")])

    conf_split = np.where(df.market == "crypto",
                          np.where(df.taker_ok == True, "crypto·1m⚡",
                                   np.where(df.taker_ok == False, "crypto·✗", "crypto·—")),
                          np.where(df.taker_ok == True, "stock·放量⚡",
                                   np.where(df.taker_ok == False, "stock·✗", "stock·—")))
    groups = [
        grp("確認×市場（crypto=1m / stock=放量，語意不同）", conf_split),
        grp("方向", df.side.map({"low": "做多(B低)", "high": "做空(B高)"})),
        grp("觸碰數", np.where(df.touches >= 3, "touches≥3", "touches=2")),
        grp("市場", df.market),
        grp("時代", np.where(df.year <= 2022, "≤2022(樣本外)", "2023+(含校準期)")),
    ]

    # 核心顯著性：做多 ⚡vs非⚡ v5 結構移停勝率差（淨損益>0 比例）
    rng = np.random.default_rng(35)
    a = (L[L.taker_ok == True].ts_pnl.dropna() > 0).values
    b = (L[L.taker_ok == False].ts_pnl.dropna() > 0).values
    sig = []
    if len(a) >= 30 and len(b) >= 30:
        d = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
             for _ in range(5000)]
        lo_ci, hi_ci = np.percentile(d, 2.5), np.percentile(d, 97.5)
        tag = "✓ 顯著" if lo_ci > 0 else "不顯著"
        sig.append(f"做多 ⚡ vs 非⚡（v5 結構移停）勝率差 {a.mean()-b.mean():+.1%}，"
                   f"bootstrap 95% CI [{lo_ci:+.1%}, {hi_ci:+.1%}] → {tag}")
    sig.append("做空(B高)全面虧損（crypto 空單有 -20ATR 級噴幣肥尾），僅供參考、不當進場依據")
    sig.append("倖存者偏差：樣本為現存標的，絕對績效偏樂觀；⚡ vs 非⚡ 為同樣本相對比較，較穩")
    sig.append("⚠ ⚡ 在 crypto(1m) 與 stock(放量) 語意不同；上行勝率差由美股主導，勿混讀")
    sig.append("⚠ crypto 未計 funding(20日約-0.6%)；同日宏觀聚集使 CI 偏窄；"
               "幾何參數為樣本內校準，headline 非樣本外")
    sig.append("⚠ crypto 1m⚡ 增量尚無法統計證明(n=38、CI跨零、尾部驅動)——"
               "採用依據是方法論定義，非回測績效")
    # 目標觸發移停框架（初始停損不動；碰到前高/前低後啟動 k×ATR 移停）
    st_pockets = []
    for label, mask in [("做多 全部", (df.side == "low")),
                        ("做多 crypto·1m⚡", (df.side == "low") & (df.market == "crypto") & (df.taker_ok == True)),
                        ("做多 stock·放量⚡", (df.side == "low") & (df.market == "stock") & (df.taker_ok == True)),
                        ("做空 全部", (df.side == "high"))]:
        st = _clean(stat_line(df[mask]["ts_pnl"]))
        if st:
            st_pockets.append({"name": label, **st})
    return {"overall": overall, "groups": groups, "pockets": pockets,
            "st_pockets": st_pockets, "sig": sig}


BT_DATA = Path(__file__).parent / "bt_data"


def _f(v, nd=3):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v) \
        else round(float(v), nd)


def write_symbol_data(df):
    """每標的一個 JSON：全部 K 線 + 該標的所有成交案例（測點/獵取/進出場），
    前端點明細列時才延遲載入（避免把幾百 MB 塞進單一 HTML）。"""
    BT_DATA.mkdir(exist_ok=True)
    stock_cache = None
    files = {}
    for (market, sym), grp_ in df.groupby(["market", "symbol"]):
        try:
            if market == "crypto":
                d = pd.read_pickle(CACHE / f"{sym}_1d.pkl")
            else:
                if stock_cache is None:
                    stock_cache = pd.read_pickle(CACHE / "stocks_5y.pkl")
                d = stock_cache.get(sym)
        except Exception:
            continue
        if d is None or d.empty:
            continue
        t = d["time"].values
        n = len(d)
        vols = d["volume"].fillna(0.0) if "volume" in d.columns else pd.Series([0.0] * n)
        candles = [{"time": int(tt), "open": round(o, 6), "high": round(h, 6),
                    "low": round(lo_, 6), "close": round(c, 6), "volume": round(v, 2)}
                   for tt, o, h, lo_, c, v in zip(d["time"], d["open"], d["high"],
                                                  d["low"], d["close"], vols)]
        sep_pct = get_config("1d", market)["sep_pct"]   # retest 容忍帶（前端軌跡用）
        sigs = []
        for _, r in grp_.iterrows():
            i = int(r.sweep_idx)
            ie = int(r.entry_idx)                # 進場根（進場分流決定；ts_bars 以此為基準）
            tis = [int(x) for x in r.touch_idxs if 0 <= int(x) < n]
            ts_bars = r.get("ts_bars")
            exit_i = ie + int(ts_bars) if pd.notna(ts_bars) else None   # v5 出場根（ts_bars=j-ie）
            sigs.append({
                "side": r.side, "touches": int(r.touches), "level": _f(r.level, 8),
                "rearm_level": int(r.get("rearm_level", 1)),   # 2=線復活第二級（前端標徽章）
                "near": _f(r.level * sep_pct, 8),   # retest：回踩到 level ±near 內才上移 SL
                "taker_ok": None if pd.isna(r.taker_ok) else bool(r.taker_ok),
                "taker_kind": None if pd.isna(r.taker_kind) else r.taker_kind,
                "taker_z": _f(r.taker_z, 2), "t20": _f(r.t20),
                "ts": _f(r.get("ts_pnl")),                            # v5 結構移停淨損益
                "ts_kind": (None if pd.isna(r.get("ts_kind")) else r.get("ts_kind")),
                "stop": _f(r.get("stop_px"), 6), "tgt": _f(r.get("tgt_px"), 8),
                "touch_times": [int(t[x]) for x in tis],
                "sweep_time": int(t[i]),
                "entry_time": int(t[ie + 1]) if ie + 1 < n else None,
                "exit_time": int(t[exit_i]) if exit_i is not None and exit_i < n else None,
            })
        fname = f"{market}_{sym}.json"
        (BT_DATA / fname).write_text(
            json.dumps({"candles": candles, "signals": sigs}, separators=(",", ":"),
                       ensure_ascii=False), encoding="utf-8")
        files[(market, sym)] = fname
    return files


def trades_payload(df, files):
    """明細表資料（不含 K 線，全量 4900+ 筆約 1MB）。"""
    out = []
    for _, r in df.iterrows():
        f = files.get((r.market, r.symbol))
        if not f:
            continue
        out.append({"market": r.market, "symbol": r.symbol, "side": r.side,
                    "touches": int(r.touches), "sweep_time": int(r.sweep_time),
                    "rearm_level": int(r.get("rearm_level", 1)),   # 2=線復活第二級
                    "taker_ok": None if pd.isna(r.taker_ok) else bool(r.taker_ok),
                    "taker_kind": None if pd.isna(r.taker_kind) else r.taker_kind,
                    "taker_z": _f(r.taker_z, 2),
                    "t3": _f(r.t3), "t5": _f(r.t5), "t10": _f(r.t10), "t20": _f(r.t20),
                    "ts": _f(r.get("ts_pnl")), "ts_bars": _f(r.get("ts_bars"), 0),
                    "mfe": _f(r.get("mfe"), 2), "mae": _f(r.get("mae"), 2), "file": f})
    out.sort(key=lambda x: -x["sweep_time"])
    return out


def build_html(stats, trades, meta, out_path):
    payload = json.dumps({"stats": stats, "trades": trades, "meta": meta},
                         separators=(",", ":"), ensure_ascii=False)
    lib_path = Path(__file__).parent / "vendor" / "lightweight-charts.standalone.production.js"
    lib = lib_path.read_text(encoding="utf-8").replace("</script>", "<\\/script>") if lib_path.exists() else ""
    html = _BT_HTML.replace("__LIB__", lib).replace("__DATA__", payload)
    Path(out_path).write_text(html, encoding="utf-8")


_BT_HTML = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>5035 流動性獵取 — 回測</title>
<script>__LIB__</script>
<style>
:root{color-scheme:dark;}*{box-sizing:border-box;}
body{margin:0;font-family:"Segoe UI",system-ui,sans-serif;background:#0e1117;color:#d1d4dc;height:100vh;display:flex;flex-direction:column;}
header{padding:10px 16px;border-bottom:1px solid #222;display:flex;gap:10px;align-items:center;flex-wrap:wrap;}
h1{font-size:15px;margin:0;}.sub{font-size:12px;color:#8b949e;}
a.btn,button{background:#161b22;color:#d1d4dc;border:1px solid #30363d;border-radius:6px;padding:5px 11px;cursor:pointer;font-size:13px;text-decoration:none;}
button.on{background:#1f6feb;border-color:#1f6feb;color:#fff;}
#main{flex:1;display:flex;min-height:0;}
#casewrap{flex:1;display:none;flex-direction:column;min-height:0;}
#chartwrap{flex:0 0 56%;position:relative;min-height:0;border-bottom:1px solid #222;}
#chart{position:absolute;inset:0;}
#chartTitle{position:absolute;top:6px;left:10px;z-index:5;font:12px ui-monospace,Consolas,monospace;color:#a8acb3;text-shadow:0 0 4px #000;pointer-events:none;}
#statwrap{flex:1;overflow-y:auto;padding:14px 20px;display:none;}
#tablewrap{flex:1;overflow:auto;}
.high{color:#ef5350;}.low{color:#26a69a;}
.d{color:#9aa4b2;}
.pos{color:#26a69a;}.neg{color:#ef5350;}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0 18px;}
th,td{border:1px solid #2a2f3a;padding:4px 10px;text-align:right;}
th:first-child,td:first-child{text-align:left;}
th{background:#161b22;color:#a8b3c4;font-weight:600;}
h2{font-size:14px;margin:14px 0 4px;color:#e6edf3;}
.note{font-size:12px;color:#8b949e;margin:3px 0;}
#dtable{width:100%;margin:0;font-size:12px;}
#dtable th{position:sticky;top:0;z-index:2;cursor:default;}
#dtable td{padding:3px 8px;border:none;border-bottom:1px solid #1b1f27;white-space:nowrap;}
#dtable tr{cursor:pointer;}
#dtable tbody tr:hover,#dtable tr.sel{background:#161b22;}
#filtbar{padding:6px 10px;border-bottom:1px solid #222;display:flex;gap:6px;align-items:center;font-size:12px;color:#8b949e;flex-wrap:wrap;}
#casebar{padding:4px 10px;border-bottom:1px solid #222;display:flex;gap:5px;align-items:center;font-size:11.5px;color:#8b949e;overflow-x:auto;white-space:nowrap;}
#casebar:empty{display:none;}
.cchip{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:2px 9px;cursor:pointer;flex:0 0 auto;}
.cchip:hover{border-color:#58a6ff;}
.cchip.on{background:#1f6feb;border-color:#1f6feb;color:#fff;}
.cchip .pos,.cchip.on .pos{color:#7ee2b8;}.cchip .neg,.cchip.on .neg{color:#ffa198;}
</style></head><body>
<header>
  <h1>📊 5035 流動性獵取 · 回測</h1><span class="sub" id="meta"></span>
  <span style="flex:1"></span>
  <button id="tabStat" class="on">統計數據</button>
  <button id="tabCase">案例明細</button>
  &nbsp;<a class="btn" href="scanner.html">← 掃描清單</a>
</header>
<div id="main">
  <div id="statwrap"></div>
  <div id="casewrap">
    <div id="chartwrap"><div id="chart"></div><div id="chartTitle"></div></div>
    <div id="casebar"></div>
    <div id="filtbar">篩選:
      <button class="f on" data-k="m" data-v="all">全部</button><button class="f" data-k="m" data-v="crypto">加密</button><button class="f" data-k="m" data-v="stock">美股</button>
      <span>|</span>
      <button class="f on" data-k="d" data-v="all">多空</button><button class="f" data-k="d" data-v="low">B低(多)</button><button class="f" data-k="d" data-v="high">B高(空)</button>
      <span>|</span>
      <button class="f on" data-k="t" data-v="all">全部</button><button class="f" data-k="t" data-v="ok">⚡確認</button>
      <span>|</span>
      <button class="f on" data-k="r" data-v="all">全級別</button><button class="f" data-k="r" data-v="2">L2復活</button>
      <span>|</span>
      <input id="q" placeholder="🔍 標的" style="background:#161b22;color:#d1d4dc;border:1px solid #30363d;border-radius:6px;padding:3px 8px;font-size:12px;width:84px">
      <span id="cnt"></span>
    </div>
    <div id="tablewrap"></div>
  </div>
</div>
<script>
const D=__DATA__;
const pct=v=>v==null?'—':(v*100).toFixed(1)+'%';
const f3=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(3);
const f2=v=>v==null?'—':v.toFixed(2);
// ---- 統計頁 ----
function statTable(title,rows,withRule){
  let h=`<h2>${title}</h2><table><tr><th></th><th>n</th><th>勝率</th><th>平均(ATR)</th><th>中位</th><th>Sharpe</th><th>PF</th></tr>`;
  for(const r of rows){
    const nm=r.rule||r.name;
    h+=`<tr><td>${nm}</td><td>${r.n??'—'}</td><td>${pct(r.win)}</td><td class="${(r.avg||0)>=0?'pos':'neg'}">${f3(r.avg)}</td><td>${f3(r.med)}</td><td>${f3(r.shp)}</td><td>${f2(r.pf)}</td></tr>`;
  }
  return h+'</table>';
}
function renderStats(){
  const s=D.stats;let h='';
  h+=statTable('🎯 口袋分析（v5 結構移停＝實際交易計畫，勝率=淨損益>0 比例）',s.pockets);
  for(const g of s.groups)h+=statTable(g.title,g.rows);
  if(s.st_pockets&&s.st_pockets.length)h+=statTable('🛑 v5 結構移停口袋（初始 SL=插破極值/針尖；retest 才上移 SL；TP=下一個流動性聚集區）',s.st_pockets);
  h+=statTable('時間出場（訊號漂移參考·非交易績效，固定持有 N 根看純價格漂移）',s.overall);
  h+='<h2>顯著性與注意事項</h2>';
  for(const t of s.sig)h+=`<div class="note">• ${t}</div>`;
  h+=`<div class="note" style="margin-top:10px">完整報告見 BACKTEST.md；明細 backtest_trades.csv。案例明細分頁共 ${D.trades.length} 筆，點任一列載入該標的全部 K 線與所有案例。</div>`;
  document.getElementById('statwrap').innerHTML=h;
}
// ---- 案例頁：上=該標的全部K線+所有案例標記，下=明細表(點列跳轉，K線延遲載入) ----
const chart=LightweightCharts.createChart(document.getElementById('chart'),{
  layout:{background:{color:'#0e1117'},textColor:'#d1d4dc'},grid:{vertLines:{color:'#1b1f27'},horzLines:{color:'#1b1f27'}},
  timeScale:{timeVisible:false,borderColor:'#30363d'},rightPriceScale:{borderColor:'#30363d'},crosshair:{mode:0}});
const candle=chart.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'});
// 價格精度自適應：低價幣(0.0026/0.04)用更多小數，否則顯示 0.00 看不到價
const precFor=p=>{if(!(p>0))return 2;const d=Math.ceil(-Math.log10(p));return Math.min(8,Math.max(2,d+3));};
chart.priceScale('right').applyOptions({scaleMargins:{top:0.05,bottom:0.22}});
const volSeries=chart.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'vol',lastValueVisible:false,priceLineVisible:false});
chart.priceScale('vol').applyOptions({scaleMargins:{top:0.82,bottom:0},visible:false});
let lvlSeries=[];
let planSeries=[],planLines=[];
function clearPlan(){
  for(const s of planSeries)chart.removeSeries(s);planSeries=[];
  for(const pl of planLines)candle.removePriceLine(pl);planLines=[];
}
const LR=5;   // = detector 1d pivot_lr，結構移停的 swing 窗口
let TRAIL_MODE='retest';   // 由 D.meta.trail 設定，決定 trailPath 用哪套移停（與回測產生此報告的模式一致）
function calcSwings(cs){   // 鏡像 detector._swings：窗口 ±LR 的局部極值
  const n=cs.length,sh=new Array(n).fill(false),sl=new Array(n).fill(false);
  for(let i=LR;i<n-LR;i++){
    let isH=true,isL=true;
    for(let k=i-LR;k<=i+LR;k++){
      if(cs[k].high>cs[i].high)isH=false;
      if(cs[k].low<cs[i].low)isL=false;
    }
    sh[i]=isH;sl[i]=isL;
  }
  return {sh,sl};
}
// 結構移停軌跡（鏡像回測，依 TRAIL_MODE）：初始 SL=插破極值(stop0=針尖)；碰 tgt 停。
// retest：只有回踩到收復線(lvl)±near 內的 higher swing low(多)/lower swing high(空) 才上移 SL；
// localhigh：每出現一個更高的 swing low(多)→SL 上移到前一個 swing low；lower low 只更新基準不上移。
function trailPath(cs,si,side,tgt,lvl,near,stop0){
  const n=cs.length,{sh,sl}=calcSwings(cs);
  let stop=stop0!=null?stop0:(side==='low'?cs[si].low:cs[si].high);
  const pts=[{time:cs[si].time,value:stop}];   // 錨在獵取根，1根出場也看得到線
  let prev=null;   // localhigh：上一個已確認 swing low(多)/high(空)
  for(let j=si+1;j<n;j++){
    pts.push({time:cs[j].time,value:stop});           // 本根生效的停損水位
    if(side==='low'?cs[j].low<=stop:cs[j].high>=stop)return pts;   // SL 掃出
    if(tgt!=null&&(side==='low'?cs[j].high>=tgt:cs[j].low<=tgt))return pts;   // 碰 TP
    const pi=j-LR;
    if(TRAIL_MODE==='localhigh'){
      if(pi>si&&(side==='low'?sl[pi]:sh[pi])){
        const cur=side==='low'?cs[pi].low:cs[pi].high;
        if(prev!=null){
          if(side==='low'&&cur>prev&&prev>stop)stop=prev;        // higher-low → 鎖前一個低點
          else if(side==='high'&&cur<prev&&prev<stop)stop=prev;  // lower-high → 鎖前一個高點
        }
        prev=cur;
      }
    }else if(pi>si&&lvl!=null&&near!=null){   // retest：回踩到收復線附近的 swing 才上移
      if(side==='low'&&sl[pi]&&cs[pi].low>stop&&Math.abs(cs[pi].low-lvl)<=near)stop=cs[pi].low;
      else if(side==='high'&&sh[pi]&&cs[pi].high<stop&&Math.abs(cs[pi].high-lvl)<=near)stop=cs[pi].high;
    }
  }
  return pts;
}
const symCache={};
async function loadSym(file){
  if(symCache[file])return symCache[file];
  const res=await fetch('bt_data/'+file);
  if(!res.ok)throw new Error('載入失敗 '+file);
  return symCache[file]=await res.json();
}
function showSym(data,tr){
  chart.priceScale('right').applyOptions({autoScale:true});  // 使用者縮放過價格軸會關掉自動縮放，換案例必須恢復，否則新標的價格在視窗外=整張空白
  const pr=precFor(tr.level||(data.candles.length?data.candles[data.candles.length-1].close:1));
  candle.applyOptions({priceFormat:{type:'price',precision:pr,minMove:Math.pow(10,-pr)}});
  candle.setData(data.candles);
  volSeries.setData(data.candles.map(c=>({time:c.time,value:c.volume||0,color:c.close>=c.open?'rgba(38,166,154,.45)':'rgba(239,83,80,.45)'})));
  for(const s of lvlSeries)chart.removeSeries(s);lvlSeries=[];
  const mk={};const put=(m,pri)=>{const k=m.time;if(!(k in mk)||mk[k].pri<pri)mk[k]={...m,pri};};
  for(const sg of data.signals){
    const pos=sg.side==='high'?'aboveBar':'belowBar';
    const ls=chart.addLineSeries({color:sg.side==='high'?'rgba(239,83,80,.7)':'rgba(66,165,245,.7)',lineWidth:2,lastValueVisible:false,priceLineVisible:false,
      autoscaleInfoProvider:()=>null});   // 輔助線不參與價格軸縮放，避免他案例的線把 K 線壓扁
    const t0=sg.touch_times.length?sg.touch_times[0]:sg.sweep_time;
    ls.setData([{time:t0,value:sg.level},{time:sg.sweep_time,value:sg.level}]);
    lvlSeries.push(ls);
    for(const tt of sg.touch_times)put({time:tt,position:pos,color:'#8b949e',shape:'circle',text:'測'},1);
    const tk=sg.taker_z==null?'':` ${sg.taker_ok?'⚡':''}${sg.taker_z>=0?'+':''}${sg.taker_z.toFixed(2)}`;
    put({time:sg.sweep_time,position:pos,color:sg.side==='high'?'#ef5350':'#26a69a',shape:sg.side==='high'?'arrowDown':'arrowUp',text:'獵取'+tk},3);
    if(sg.entry_time)put({time:sg.entry_time,position:pos,color:'#ffd54f',shape:'circle',text:'進'},2);
    if(sg.exit_time){
      // 出場K棒標出場類型與價位（tp=漲到聚集區 / trail=結構移停掃出 / sl=初始停損）
      const ei=data.candles.findIndex(c=>c.time===(sg.entry_time||sg.sweep_time));   // 進場根=ie+1
      const si2=ei>0?ei-1:data.candles.findIndex(c=>c.time===sg.sweep_time);          // 軌跡錨在進場前一根 ie（鏡像 backtest range(ie+1,n)）
      const pts2=si2>=0?trailPath(data.candles,si2,sg.side,sg.tgt,sg.level,sg.near,sg.stop):[];
      const px=pts2.length?pts2[pts2.length-1].value:null;
      const pdec=px==null?2:(Math.abs(px)>=100?2:Math.abs(px)>=1?3:6);
      const lab=sg.ts_kind==='tp'?'止盈@聚集區':(sg.ts_kind==='trail'?'移停出場':'停損出場');
      put({time:sg.exit_time,position:pos,color:sg.ts==null?'#9aa4b2':(sg.ts>0?'#26a69a':'#ef5350'),shape:'square',
           text:lab+(px!=null?'@'+px.toFixed(pdec):'')},2);
    }
  }
  candle.setMarkers(Object.values(mk).sort((a,b)=>a.time-b.time).map(({pri,...m})=>m));
  document.getElementById('chartTitle').textContent=`${tr.symbol}（${tr.market==='crypto'?'加密':'美股'}）· ${data.candles.length} 根 · 本標的案例 ${data.signals.length} 筆`;
  buildCaseBar(data,tr);
  const sg=data.signals.find(s=>s.sweep_time===tr.sweep_time)||data.signals[0];
  focusCase(data,sg);
}
// 聚焦某案例：第一個觸碰前 10 根 ~ 出場後；右側留白隨視野等比放大，獵取區不貼邊
function focusCase(data,sg){
  chart.priceScale('right').applyOptions({autoScale:true});  // 使用者縮放過價格軸會鎖死，切案例必恢復（chip 與表列兩路徑都經過這裡）
  const idx=t=>data.candles.findIndex(c=>c.time===t);
  const si=idx(sg.sweep_time);
  const fi=sg.touch_times.length?idx(sg.touch_times[0]):si-40;
  const from=Math.max(0,(fi>=0?fi:si-40)-10);
  const pad=Math.max(26,Math.round((si-from)*0.3));
  chart.timeScale().setVisibleLogicalRange({from:from,to:si+pad});
  document.querySelectorAll('#casebar .cchip').forEach(x=>x.classList.toggle('on',+x.dataset.t===sg.sweep_time));
  // 交易計畫視覺化（聚焦案例專屬）：移動停損軌跡 + 參考目標線
  // （初始停損線已移除——停損資訊改標在出場K棒上，依使用者要求）
  clearPlan();
  if(sg.tgt!=null)planLines.push(candle.createPriceLine({price:sg.tgt,color:'#26a69a',lineWidth:1,lineStyle:2,axisLabelVisible:true,title:'TP·下一個流動性聚集區'}));
  if(si>=0&&si+1<data.candles.length){
    const pts=trailPath(data.candles,si,sg.side,sg.tgt,sg.level,sg.near);
    if(pts.length>1){
      const tline=chart.addLineSeries({color:'#ff9800',lineWidth:2,lineStyle:0,lineType:1,lastValueVisible:false,priceLineVisible:false,title:'結構移停',
        autoscaleInfoProvider:()=>null});   // 不參與價格軸縮放
      tline.setData(pts);planSeries.push(tline);
    }
  }
}
// 同標的多案例快速切換列
function buildCaseBar(data,tr){
  const bar=document.getElementById('casebar');bar.innerHTML='';
  if(data.signals.length<2)return;
  const lab=document.createElement('span');lab.textContent=`本標的 ${data.signals.length} 筆案例:`;bar.appendChild(lab);
  [...data.signals].sort((a,b)=>a.sweep_time-b.sweep_time).forEach(sg=>{
    const d=new Date(sg.sweep_time*1000).toISOString().slice(2,10);
    const r=sg.ts==null?'<span>未完成</span>':`<span class="${sg.ts>0?'pos':'neg'}">${sg.ts>=0?'+':''}${sg.ts.toFixed(1)}</span>`;
    const chip=document.createElement('span');chip.className='cchip';chip.dataset.t=sg.sweep_time;
    chip.innerHTML=`${d} ${sg.side==='low'?'多':'空'}${sg.taker_ok?'⚡':''} ${r}`;
    chip.onclick=()=>{
      focusCase(data,sg);
      // 同步明細表選取（若該案例在目前篩選的列中）
      const tr2=[...document.querySelectorAll('#dtable tbody tr')].find(el=>{
        const c=curRows[+el.dataset.i];
        return c&&c.symbol===tr.symbol&&c.sweep_time===sg.sweep_time;
      });
      if(tr2){if(selRow)selRow.classList.remove('sel');selRow=tr2;tr2.classList.add('sel');tr2.scrollIntoView({block:'center'});}
    };
    bar.appendChild(chip);
  });
}
let mF='all',dF='all',tF='all',rF='all',qF='',selRow=null,curRows=[];
const fA=v=>v==null?'—':`<span class="${v>=0?'pos':'neg'}">${v>=0?'+':''}${v.toFixed(2)}</span>`;
function renderTable(){
  const rows=D.trades.filter(c=>(mF==='all'||c.market===mF)&&(dF==='all'||c.side===dF)&&(tF==='all'||c.taker_ok===true)&&(rF==='all'||c.rearm_level===2)&&(qF===''||c.symbol.toUpperCase().includes(qF)));
  curRows=rows;
  document.getElementById('cnt').textContent=`${rows.length} 筆`;
  let h='<table id="dtable"><thead><tr><th>日期</th><th>市場</th><th>標的</th><th>方向</th><th>觸碰</th><th>taker/量</th><th>v5移停</th><th>t3</th><th>t5</th><th>t10</th><th>t20</th><th>MFE</th><th>MAE</th></tr></thead><tbody>';
  rows.forEach((c,i)=>{
    const dt=new Date(c.sweep_time*1000).toISOString().slice(0,10);
    const tk=c.taker_z==null?'—':`${c.taker_z>=0?'+':''}${c.taker_z.toFixed(2)} ${c.taker_kind||''}${c.taker_ok?' ⚡':''}`;
    const l2=c.rearm_level===2?' <span style="background:#8957e5;color:#fff;border-radius:3px;padding:0 4px;font-size:10px">L2復活</span>':'';
    h+=`<tr data-i="${i}"><td>${dt}</td><td>${c.market==='crypto'?'加密':'美股'}</td><td><b>${c.symbol}</b>${l2}</td>`+
      `<td class="${c.side==='high'?'high':'low'}">${c.side==='low'?'B低(多)':'B高(空)'}</td><td>${c.touches}</td><td>${tk}</td>`+
      `<td>${c.ts==null?'—':fA(c.ts)+(c.ts_bars!=null?'<span style="color:#8b949e">/'+c.ts_bars+'根</span>':'')}</td>`+
      `<td>${fA(c.t3)}</td><td>${fA(c.t5)}</td><td>${fA(c.t10)}</td><td>${fA(c.t20)}</td><td>${c.mfe??'—'}</td><td>${c.mae??'—'}</td></tr>`;
  });
  document.getElementById('tablewrap').innerHTML=h+'</tbody></table>';
  document.querySelectorAll('#dtable tbody tr').forEach(el=>el.onclick=async()=>{
    if(selRow)selRow.classList.remove('sel');
    selRow=el;el.classList.add('sel');
    const c=rows[+el.dataset.i];
    try{showSym(await loadSym(c.file),c);}catch(e){document.getElementById('chartTitle').textContent='載入失敗: '+e.message+'（請經 http://localhost:5035 開啟，file:// 無法 fetch）';}
  });
  const first=document.querySelector('#dtable tbody tr');
  if(first)first.click();
}
document.querySelectorAll('#filtbar .f').forEach(b=>b.onclick=()=>{
  const k=b.dataset.k;
  document.querySelectorAll(`#filtbar .f[data-k="${k}"]`).forEach(x=>x.classList.toggle('on',x===b));
  if(k==='m')mF=b.dataset.v;else if(k==='d')dF=b.dataset.v;else if(k==='t')tF=b.dataset.v;else rF=b.dataset.v;
  renderTable();
});
document.getElementById('q').oninput=e=>{qF=e.target.value.trim().toUpperCase();renderTable();};
// ---- 分頁切換 ----
function tab(which){
  document.getElementById('tabStat').classList.toggle('on',which==='stat');
  document.getElementById('tabCase').classList.toggle('on',which==='case');
  document.getElementById('statwrap').style.display=which==='stat'?'block':'none';
  document.getElementById('casewrap').style.display=which==='case'?'flex':'none';
}
let caseInit=false;
document.getElementById('tabStat').onclick=()=>tab('stat');
document.getElementById('tabCase').onclick=()=>{tab('case');sizeChart();if(!caseInit){caseInit=true;renderTable();}};
function sizeChart(){
  const cw=document.getElementById('chartwrap');
  if(cw.clientWidth&&cw.clientHeight)chart.applyOptions({width:cw.clientWidth,height:cw.clientHeight});
}
new ResizeObserver(sizeChart).observe(document.getElementById('chartwrap'));
TRAIL_MODE=D.meta.trail||'retest';
document.getElementById('meta').textContent=D.meta.text;
renderStats();
tab('stat');
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-crypto", action="store_true")
    ap.add_argument("--skip-stock", action="store_true")
    ap.add_argument("--out", default="BACKTEST.md")
    ap.add_argument("--reclaim-enter", type=float, default=None, help="覆寫進場分流門檻(掃描用)")
    ap.add_argument("--trail", choices=["retest", "localhigh"], default=None,
                    help="移停模式：retest(原v5回踩線) / localhigh(swing-low結構移停)")
    args = ap.parse_args()
    if args.reclaim_enter is not None:
        global _RECLAIM_OVERRIDE
        _RECLAIM_OVERRIDE = args.reclaim_enter
    if args.trail is not None:
        global _TRAIL_MODE
        _TRAIL_MODE = args.trail
    t0 = time.time()
    df, skips = collect(args)
    if df.empty:
        print("沒有任何訊號")
        return
    helper_cols = ["sweep_idx", "level", "touch_idxs"]
    df.drop(columns=helper_cols).to_csv(Path(__file__).parent / "backtest_trades.csv",
                                        index=False, encoding="utf-8-sig")
    lines = []
    report(df.drop(columns=helper_cols), skips, lines)
    Path(__file__).parent.joinpath(args.out).write_text("\n".join(lines), encoding="utf-8")
    # 互動前端：每標的 JSON（全 K 線+所有案例，前端延遲載入）+ 明細表
    stats = build_stats(df)
    files = write_symbol_data(df)
    trades = trades_payload(df, files)
    meta = {"text": (f"訊號 {len(df)} 筆（crypto {(df.market=='crypto').sum()} / "
                     f"stock {(df.market=='stock').sum()}）· 移停={_TRAIL_MODE} · {pd.Timestamp.now():%Y-%m-%d}"),
            "trail": _TRAIL_MODE}
    build_html(stats, trades, meta, Path(__file__).parent / "backtest.html")
    print(f"\n耗時 {time.time()-t0:.0f}s；明細 backtest_trades.csv，報告 {args.out}，"
          f"前端 backtest.html（明細 {len(trades)} 筆 / 標的資料 {len(files)} 檔）")


if __name__ == "__main__":
    main()
