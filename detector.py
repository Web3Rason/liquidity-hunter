"""
流動性獵取（Stop Hunt / Liquidity Sweep）偵測器 —— 依「貝格先生」方法。

【流動性聚集區（獵取目標）】兩種、兩側：
  - 等高 / 等低（共識區）：價格「多次進出同一價帶並被拒絕」→ 越多次觸碰，止損越密
      觸碰數 = 價格進出該價帶的次數（含 swing 點合併；兩次觸碰之間須離開價帶 sep_pct）
      需 >= min_touches（預設 2）才算有效共識區
  - 前高 / 前低：單一「足夠顯著」的極值（過去 prom_window 內的最高/最低）

【獵取型態】兩型、兩側：
  A 影線插破收回：影線插破聚集區極值、收盤收回內側（當根拒絕）
  B 假突破收復(MSB)：收盤假突破極值、停留數根後又收盤站回內側；
      期間極值須超出 touch_tol 容忍帶（沒掃到止損帶的微量假突破不算獵取）；
      acceptance 積分 Σ max(0,收盤超出距離)/ATR(插破前鎖定) > max_acc_dev 即視為
      真突破作廢——插越深容許停留越短（視覺QA：深插久留全是趨勢反轉假陽性）；
      收復收盤須站回線內側 ≥ reclaim_atr×ATR（貼線的猶豫收復不算）

【確認時點】訊號在 pattern 完成的那根成立（A=插破收回當根、B=收復當根），
  刻意不用「之後 N 根的走勢」過濾——那是前視偏誤，即時監控也做不到；
  真假獵取的進一步鑑別由 taker/CVD 處理（見 taker.py 的 attach_taker）。

side='high' → 獵取上方流動性（看空）；side='low' → 獵取下方流動性（看多）。
為避免前視：價帶由「已確認(右側 lr 根後)」的 swing 點建立；觸碰/獵取只用當下及更早的 K 棒。
"""
import pandas as pd

DEFAULTS = {
    "1h": {
        "pivot_lr": 8,          # 建立價帶的 swing 左右窗口
        "eq_tol": 0.004,        # 價帶寬度 / 聚類容忍度（0.4%）
        "min_touches": 2,       # 最少觸碰數（2=等高低EQH/EQL，3+=強共識；報告可再拉高過濾）
        "min_hold": 5,          # 站穩期：突破須距「最後一次建立觸碰」至少這麼多根（否則=只測1次就破，不算有效S/R）
        "touch_tol": 0.005,     # 容忍帶：影線在此範圍內插破並收回 = 觸碰(強化S/R)，不算突破
        "sep_pct": 0.012,       # 兩次觸碰之間，價格必須先離開價帶這麼遠才算「另一個點」
        "prom_window": 120,     # 前高/前低：需為過去這麼多根內的極值（足夠顯著）
        "max_pierce": 0.10,     # 插破深度上限（僅作極端 sanity 界線；真正判準是收復+反轉，深插不排除）
        "min_below": 1,         # B 型：假突破後至少停留幾根
        "max_below": 6,         # B 型：最多幾根內要收復
        "rej_wick": 0.30,       # A 型：拒絕影線佔整根全幅比例下限
        "expiry": 360,          # 價帶過期（根）
        "max_acc_dev": 4.0,     # B 型 acceptance 積分上限：Σ max(0,收盤超出線距離)/ATR(插破前)
                                #   插越深容許停留越短；含插破當根、單根深收盤即可爆表（有意：
                                #   收盤一根站到 4 ATR 外=極端接受度，深「影線」歸 A 型管）
                                #   ⚠ K=4.0 由 1d 標注案例實測校準，1h 沿用未獨立驗證
        "reclaim_atr": 0.15,    # B 型收復邊距：收盤須站回線內側 ≥ 此倍數 ATR（貼線不算收復）
        "reclaim_enter": 0.5,   # 進場分流(交易層)：收復站回≥此ATR倍=強收復→次根立即進場；弱收復→進觀察
                                #   0.5=fable 審定(排貼線噪音、非擬合峰值0.75；對齊作者「站回之上+迅速」語意)
        "watch_bars": 3,        # 弱收復觀察最多幾根，等收盤站回達門檻才進場(破針尖則放棄)
        "max_birth_tan": 1,     # 線誕生前 20 根切線數上限：被切 2 次即不行（使用者裁決）——
                                #   散戶止損不會一致堆在它後面 → 不符方法論前提，不建帶
                                #   （教科書案例實測 0~2；註：左濁做多回測 PF 反而高，但那不是本 pattern 的錢）
        "rearm_enable": False,  # 線復活（多級獵取）：1h 暫不開（zombie_ttl 語意=20 小時，待 v2 校準）
        "rearm_gap": 0.5,       # 更深新針尖須比上級針尖深 >= 此倍 ATR 才算「新一級」
        "zombie_ttl": 20,       # 第一級 emit 後線進 ZOMBIE 潛伏，逾此根數未被更深獵取 → 永久死
        "max_level": 2,         # 級數封頂（工程保守值；兩級以上的樣本僅 CRV 一例）
        "spring_enable": False,            # 大級別假跌破收復(macro spring)：1h 暫不開(ttl 語意=小時，待 v2)
        "spring_keepalive_touches": 3,     # 強帶(touches>=此值)未站穩被影線插破不報廢，保留 NORMAL
        "spring_min_touches": 4,           # DEVIATED 超時/積分爆表後，touches>=此值才降級 SPRING_WAIT 潛伏（4=強共識；QTUM=5 安全，防雜訊強帶爆量）
        "spring_ttl": 25,                  # SPRING_WAIT 潛伏上限根數，逾此未收復 → 死
        "spring_reclaim_atr": 0.5,         # SPRING_WAIT 收復邊距：收盤須站回線內側 >= 此倍 ATR（比 L1 的 reclaim_atr 嚴）
        "spring_max_pierce": 0.30,         # SPRING_WAIT 期間收盤崩破此比例 = 真跌破 → 死
        "spring_min_depth": 0.10,          # 深掃幅度閘（1h spring 未開，參數備用，語意同 1d）
        "spring_min_bars": 7,              # 停留時長閘（1h spring 未開，參數備用，語意同 1d）
    },
    "1d": {
        "pivot_lr": 5,
        "eq_tol": 0.012,
        "min_touches": 2,
        "min_hold": 3,
        "touch_tol": 0.012,
        "sep_pct": 0.035,
        "prom_window": 60,
        "max_pierce": 0.15,
        "min_below": 1,
        "max_below": 6,
        "rej_wick": 0.30,
        "expiry": 220,
        "max_acc_dev": 4.0,
        "reclaim_atr": 0.15,
        "reclaim_enter": 0.5,   # 進場分流(交易層)：收復站回≥此ATR倍=強收復→次根立即進場；弱收復→進觀察
                                #   0.5=fable 審定(排貼線噪音、非擬合峰值0.75；對齊作者「站回之上+迅速」語意)
        "watch_bars": 3,        # 弱收復觀察最多幾根，等收盤站回達門檻才進場(破針尖則放棄)
        "max_birth_tan": 1,
        "rearm_enable": True,   # 線復活（多級獵取）：1d 啟用——第一級 emit 後線進 ZOMBIE 潛伏，
                                #   被「更深新針尖(>=rearm_gap ATR)」再獵取一次→收復舊線+硬性taker→emit 第二級
                                #   （修「一條線 emit 一次就死」的缺陷）
        "rearm_gap": 0.5,       # 更深新針尖須比上級針尖深 >= 此倍 ATR 才算「新一級」（深度增量用 ATR 度量）
        "zombie_ttl": 20,       # ZOMBIE 潛伏上限根數，逾此未被更深獵取 → 永久死
        "max_level": 2,         # 級數封頂（工程保守值，理由同上設計文件）
        "spring_enable": True,             # 大級別假跌破收復(macro spring)：1d 啟用——強帶跌破後住數週、深掃、
                                           #   再強勢站回原線=Wyckoff Spring；補抓「非3~5根V轉」的大週期獵取
        "spring_keepalive_touches": 3,     # 強帶(touches>=此值)未站穩被影線插破不報廢，保留 NORMAL 等真跌破
        "spring_min_touches": 4,           # DEVIATED 超時/積分爆表後，touches>=此值才降級 SPRING_WAIT 潛伏（4=強共識；QTUM=5 安全，防雜訊強帶爆量）
        "spring_ttl": 25,                  # SPRING_WAIT 潛伏上限根數，逾此未收復 → 死（QTUM 需 14，留裕度）
        "spring_reclaim_atr": 0.5,         # SPRING_WAIT 收復邊距：收盤須站回線內側 >= 此倍 ATR（比 L1 的 0.15 嚴）
        "spring_max_pierce": 0.30,         # SPRING_WAIT 期間收盤崩破此比例 = 真跌破 → 死（QTUM 針尖 -18%）
        "spring_min_depth": 0.10,          # 深掃幅度閘：收復 emit 前針尖須比原線深破 >= 此比例（QTUM -18.4% 安全；擋淺破雜訊，macro spring 主濾網）
        "spring_min_bars": 7,              # 停留時長閘：跌破到收復須 >= 此根數才算大級別（> max_below=6；QTUM 19 根安全；擋 acc 爆表單根短命進入）
    },
}


def _swings(high, low, lr):
    n = len(high)
    sh = [False] * n
    sl = [False] * n
    for i in range(lr, n - lr):
        if high[i] == high[i - lr:i + lr + 1].max():
            sh[i] = True
        if low[i] == low[i - lr:i + lr + 1].min():
            sl[i] = True
    return sh, sl


def detect_sweeps(df: pd.DataFrame, cfg: dict) -> list:
    n = len(df)
    o = df["open"].values
    h = df["high"].values
    low = df["low"].values
    c = df["close"].values
    t = df["time"].values
    # ATR(14)：B 型 acceptance 積分與收復邊距的度量衡
    _pc = df["close"].shift(1)
    _tr = pd.concat([df["high"] - df["low"], (df["high"] - _pc).abs(),
                     (df["low"] - _pc).abs()], axis=1).max(axis=1)
    atr = _tr.rolling(14).mean().values

    def dev_atr_at(i):
        """插破當下鎖定的 ATR：用前一根（防黑天鵝當根把基準撐大）；無效則退當根。"""
        a = atr[i - 1] if i >= 1 else float("nan")
        if not (a > 0):
            a = atr[i]
        return a if a > 0 else None

    lr = cfg["pivot_lr"]
    sh, sl = _swings(h, low, lr)
    pw = cfg["prom_window"]
    eq_tol = cfg["eq_tol"]
    max_pierce = cfg["max_pierce"]
    min_touches = cfg["min_touches"]
    min_hold = cfg["min_hold"]
    touch_tol = cfg["touch_tol"]
    sep_pct = cfg["sep_pct"]

    def _valid_downtrend_low(j, i):
        # swing low j 是不是「仍有效的下降趨勢底」(純結構、無前視)：
        #   閘1：j 前最近 swing high < 窗(pw)內最高峰 → 價格從峰下降(非上升趨勢中的小回)
        #   閘2：j 之後到 i 未漲回該峰 → 下降尚未被上升收復(未被取代；漲回/超過峰=底失效)
        hb = [k for k in range(max(0, j - pw), j) if sh[k]]
        if len(hb) < 2:
            return False
        peak = max(h[k] for k in hb)
        if not (h[hb[-1]] < peak * (1 - eq_tol)):
            return False
        if i > j and h[j + 1:i + 1].max() >= peak:
            return False
        return True

    def dominated_low(lv, i):
        # 流動性線錨點規則：下方存在「更低、收盤未跌破、且仍有效的下降趨勢底」D → 抑制此線
        #   (使用者裁決：線該錨在最低、還沒被跌破/還沒被上升取代的下降底，不該畫在上方較高的低)
        lvl = lv["level"]
        for j in range(max(0, i - cfg["expiry"]), i - lr + 1):
            if not sl[j] or low[j] >= lvl * (1 - eq_tol):
                continue
            if c[j:i + 1].min() < low[j] * (1 - touch_tol):   # 收盤跌破過 D → D 失效
                continue
            if _valid_downtrend_low(j, i):
                return True
        return False

    def established(lv, i, side):
        # 已是有效目標(觸碰夠/顯著前高低) 且 距最後一次建立觸碰已站穩 min_hold 根
        if not (lv["touches"] >= min_touches or lv["prominent"]):
            return False
        if (i - lv["touch_idxs"][-1]) < min_hold:
            return False
        if side == "low" and dominated_low(lv, i):   # 下方有更深的有效下降底 → 不算流動性
            return False
        return True

    def _to_spring_or_dead(lv, i, side):
        # DEVIATED 住太久(超 max_below 或 acc 爆表)：符合 macro spring 資格 → 降級 SPRING_WAIT 潛伏等收復，否則 DEAD。
        #   資格(全結構閘 AND)：①強帶 ②已真掃止損(極值超出容忍帶) ③價值未轉移(下方/上方未結成新強帶)
        if not cfg.get("spring_enable"):
            lv["state"] = "DEAD"; return
        lvl = lv["level"]
        strong = lv["touches"] >= cfg["spring_min_touches"]   # macro spring 須真堆疊止損的強等高低帶；單一顯著前高(prominent 但 touches<門檻)被+20%真突破後回落不算(INJ 假陽性)
        if side == "low":
            swept = lv["extreme"] < lvl * (1 - touch_tol)
            migrated = any(l2 is not lv and l2["state"] == "NORMAL" and l2["level"] < lvl
                           and (l2["touches"] >= min_touches or l2["prominent"]) and c[i] < l2["level"]
                           for l2 in lows)
        else:
            swept = lv["extreme"] > lvl * (1 + touch_tol)
            migrated = any(h2 is not lv and h2["state"] == "NORMAL" and h2["level"] > lvl
                           and (h2["touches"] >= min_touches or h2["prominent"]) and c[i] > h2["level"]
                           for h2 in highs)
        if strong and swept and not migrated:
            lv["state"] = "SPRING_WAIT"; lv["spring_since"] = i; lv["spring_dev_idx"] = lv["dev_idx"]
        else:
            lv["state"] = "DEAD"

    highs, lows, signals = [], [], []

    def source_of(lv):
        return "equal" if lv["touches"] >= min_touches else ("prev" if lv["prominent"] else "weak")

    def emit(sig_type, side, lv, idx, extreme):
        # TP 目標（作者原文：「TP 置於下一個 Liquidity 聚集區」）＝對側最近的 local high/low：
        # 取「仍活躍(NORMAL，未過期/未被突破)的價帶」中離收復線最近的一條
        # （B低=上方最低的高帶；B高=下方最高的低帶）。用活躍價帶＝自動濾掉遠古無關價位。
        # 不要求 touches>=2/prominent——任何前高/前低都是流動性（下跌趨勢的 lower-high 也算 TP）。
        # 只用當下活躍價帶，無前視。
        if side == "low":
            cands = [h2["level"] for h2 in highs
                     if h2["state"] == "NORMAL" and h2["level"] > lv["level"]]
            opp = min(cands) if cands else None
        else:
            cands = [l2["level"] for l2 in lows
                     if l2["state"] == "NORMAL" and l2["level"] < lv["level"]]
            opp = max(cands) if cands else None
        lv["last_opp"] = opp        # ZOMBIE 期間「TP 觸及→永久死」閘要用第一級 TP
        signals.append({
            "opp_level": round(float(opp), 8) if opp is not None else None,
            "rearm_level": lv.get("level_no", 1),          # 1=第一級；2=線復活後的更深第二級
            "type": sig_type, "side": side, "source": source_of(lv),
            "level_price": round(lv["level"], 8),
            "zone_top": round(lv["bmax"], 8), "zone_bottom": round(lv["bmin"], 8),
            "level_start_time": int(t[lv["start"]]),
            "sweep_time": int(t[idx]), "sweep_idx": idx,
            "dev_idx": lv["dev_idx"] if lv["dev_idx"] is not None else idx,  # B=收盤插破那根；A=當根
            "extreme": round(extreme, 8), "touches": lv["touches"],
            "touch_idxs": list(lv["touch_idxs"]),
            "acc_dev": round(lv.get("acc_dev", 0.0), 2),   # B 型 acceptance 積分（日後校準/回測分布用）
            "birth_tan": lv.get("birth_tan", 0),           # 線誕生前20根切線數（左側乾淨度，0=最乾淨）
        })

    for i in range(n):
        pi = i - lr

        # ---- 由已確認 swing 建立 / 擴充 價帶 ----
        if pi >= 0 and sh[pi]:
            p = h[pi]
            merged = False
            for lv in highs:
                if lv["state"] != "NORMAL":       # 突破中(DEVIATED)不合併，避免插破尖刺汙染判定線
                    continue
                if abs(p - lv["level"]) / lv["level"] <= eq_tol:
                    # 合併的 swing 點若距上次觸碰已離開過價帶(sep_pct)，記為一次觸碰
                    lt = lv["touch_idxs"][-1]
                    if pi > lt + 1 and h[lt + 1:pi].min() < lv["level"] * (1 - sep_pct):
                        lv["touches"] += 1
                        lv["touch_idxs"].append(pi)
                        lv["armed"] = False       # 計入觸碰即消耗武裝態，防同一次造訪被 armed 路徑重複計數
                    lv["bmax"] = max(lv["bmax"], p)
                    lv["bmin"] = min(lv["bmin"], p)
                    lv["level"] = lv["bmax"]      # 上方：被插破的是價帶頂
                    lv["last"] = max(lv["last"], pi)
                    merged = True
                    break
            if not merged:
                prominent = pi >= pw and p == h[pi - pw:pi + 1].max()
                j0 = max(0, pi - 20)   # 線誕生前 20 根的切線數：左側乾淨度（0=線長在價格行為邊緣）
                btan = int(((low[j0:pi] <= p) & (p <= h[j0:pi])).sum()) if pi > j0 else 0
                if btan <= cfg["max_birth_tan"]:   # 線生在舊密集區=不「明顯」→ 不符方法論前提，不建帶
                    highs.append({"bmin": p, "bmax": p, "level": p, "touches": 1,
                                  "armed": False, "prominent": prominent, "birth_tan": btan,
                                  "start": pi, "last": pi, "state": "NORMAL", "level_no": 1,
                                  "dev_idx": None, "extreme": p, "touch_idxs": [pi]})

        if pi >= 0 and sl[pi]:
            p = low[pi]
            merged = False
            for lv in lows:
                if lv["state"] != "NORMAL":       # 突破中(DEVIATED)不合併，避免插破尖刺汙染判定線
                    continue
                if abs(p - lv["level"]) / lv["level"] <= eq_tol:
                    # 合併的 swing 點若距上次觸碰已離開過價帶(sep_pct)，記為一次觸碰
                    lt = lv["touch_idxs"][-1]
                    if pi > lt + 1 and low[lt + 1:pi].max() > lv["level"] * (1 + sep_pct):
                        lv["touches"] += 1
                        lv["touch_idxs"].append(pi)
                        lv["armed"] = False       # 計入觸碰即消耗武裝態，防同一次造訪被 armed 路徑重複計數
                    lv["bmax"] = max(lv["bmax"], p)
                    lv["bmin"] = min(lv["bmin"], p)
                    lv["level"] = lv["bmin"]      # 下方：被插破的是價帶底
                    lv["last"] = max(lv["last"], pi)
                    merged = True
                    break
            if not merged:
                prominent = pi >= pw and p == low[pi - pw:pi + 1].min()
                j0 = max(0, pi - 20)   # 線誕生前 20 根的切線數：左側乾淨度
                btan = int(((low[j0:pi] <= p) & (p <= h[j0:pi])).sum()) if pi > j0 else 0
                if btan <= cfg["max_birth_tan"]:   # 線生在舊密集區=不「明顯」→ 不符方法論前提，不建帶
                    lows.append({"bmin": p, "bmax": p, "level": p, "touches": 1,
                                 "armed": False, "prominent": prominent, "birth_tan": btan,
                                 "start": pi, "last": pi, "state": "NORMAL", "level_no": 1,
                                 "dev_idx": None, "extreme": p, "touch_idxs": [pi]})

        # DEVIATED/ZOMBIE 不吃 expiry（ZOMBIE 有自己的 zombie_ttl）
        highs = [lv for lv in highs if lv["state"] in ("DEVIATED", "ZOMBIE", "SPRING_WAIT") or (i - lv["last"]) <= cfg["expiry"]]
        lows = [lv for lv in lows if lv["state"] in ("DEVIATED", "ZOMBIE", "SPRING_WAIT") or (i - lv["last"]) <= cfg["expiry"]]

        rng = h[i] - low[i]
        upper_wick = h[i] - max(o[i], c[i])
        lower_wick = min(o[i], c[i]) - low[i]

        # ---- 上方價帶：觸碰計數 + 獵取 ----
        for lv in highs:
            lvl = lv["level"]
            band_lo = min(lv["bmin"], lvl * (1 - eq_tol))
            touch_ceil = lvl * (1 + touch_tol)
            if lv["state"] == "NORMAL":
                if c[i] > lvl:
                    # 收盤站上線外 → B 型假突破(deviation)
                    if established(lv, i, "high"):
                        if (c[i] - lvl) / lvl <= max_pierce:
                            lv["state"] = "DEVIATED"; lv["dev_idx"] = i; lv["extreme"] = h[i]
                            lv["dev_atr"] = dev_atr_at(i)
                            lv["acc_dev"] = (c[i] - lvl) / lv["dev_atr"] if lv["dev_atr"] else 0.0
                        else:
                            lv["state"] = "DEAD"          # 收太上 = 真突破
                    elif (cfg.get("spring_enable") and (c[i] - lvl) / lvl <= max_pierce
                          and lv["touches"] >= cfg["spring_keepalive_touches"]):
                        pass   # 強帶僅因 min_hold 未過(established=False)收盤微破 → 保留 NORMAL 等站穩後真突破(macro spring 鏡像)；不更新 last/touch_idxs
                    elif c[i] > touch_ceil:
                        lv["state"] = "DEAD"              # 未建立/未站穩就明確收破 → 廢棄
                    # else: 微量收破(touch_tol 內)容忍，成形中的等高帶不被雜訊錯殺
                elif h[i] > touch_ceil:
                    # 深影線插破容忍帶、收盤收回 → A 型影線獵取
                    _est = established(lv, i, "high")
                    _pierce_ok = (h[i] - lvl) / lvl <= max_pierce
                    if _est and _pierce_ok:
                        if rng > 0 and upper_wick / rng >= cfg["rej_wick"]:
                            emit("A", "high", lv, i, h[i])
                        lv["state"] = "DEAD"
                    elif (cfg.get("spring_enable") and not _est and _pierce_ok
                          and lv["touches"] >= cfg["spring_keepalive_touches"]):
                        pass   # 強帶未站穩被插破 → 保留 NORMAL 等真突破(macro spring 鏡像)；不更新 last/touch_idxs
                    else:
                        lv["state"] = "DEAD"   # 弱帶/插太深(真突破) → 廢棄
                else:
                    # 其餘(含些微插針 lvl~touch_ceil、或單純觸碰) → 觸碰計數
                    if lv["armed"] and h[i] >= band_lo and c[i] <= lvl:
                        lv["touches"] += 1; lv["armed"] = False; lv["last"] = i; lv["touch_idxs"].append(i)
                    elif (not lv["armed"]) and h[i] < lvl * (1 - sep_pct):
                        lv["armed"] = True
            elif lv["state"] == "DEVIATED":
                lv["extreme"] = max(lv["extreme"], h[i])
                bars = i - lv["dev_idx"]
                a0 = lv["dev_atr"]
                ref = lv.get("ref2", lvl)            # 第二級 acceptance 對新針尖量測；第一級對 lvl
                if a0:
                    lv["acc_dev"] += max(0.0, c[i] - ref) / a0
                margin = cfg["reclaim_atr"] * a0 if a0 else 0.0
                if c[i] > lvl * (1 + max_pierce):
                    lv["state"] = "DEAD"             # 噴出硬上限，spring 不接管
                elif a0 and lv["acc_dev"] > cfg["max_acc_dev"]:
                    _to_spring_or_dead(lv, i, "high")  # acceptance 爆表=住上去=真突破；強帶深掃→SPRING_WAIT 等收復
                elif bars > cfg["max_below"]:
                    _to_spring_or_dead(lv, i, "high")  # 超時未收復；強帶深掃→SPRING_WAIT 等收復
                elif bars >= cfg["min_below"] and c[i] < lvl - margin:   # 收復須站回線內側一個邊距
                    emitted = lv["extreme"] > touch_ceil  # 極值須超出容忍帶=真的掃到止損，微量假突破不算獵取
                    if emitted:
                        emit("B", "high", lv, i, lv["extreme"])
                    # 失敗的獵取(emit 完)進 ZOMBIE 潛伏，等是否被「更高新針尖」再獵取一級
                    if emitted and cfg.get("rearm_enable") and lv["level_no"] < cfg["max_level"]:
                        lv["state"] = "ZOMBIE"; lv["tip1"] = lv["extreme"]; lv["zombie_since"] = i
                    else:
                        lv["state"] = "DEAD"
            elif lv["state"] == "SPRING_WAIT":
                # 大級別假突破(macro spring 鏡像)：強帶噴出後潛伏，等收盤強勢跌回原線才 emit
                lv["extreme"] = max(lv["extreme"], h[i])        # 持續追最高針尖
                a = dev_atr_at(i)
                margin = cfg["spring_reclaim_atr"] * a if a else 0.0
                if c[i] > lvl * (1 + cfg["spring_max_pierce"]):     # 閘(a) 收盤噴破=真突破
                    lv["state"] = "DEAD"
                else:
                    above = [h2["level"] for h2 in highs if h2 is not lv and h2["state"] == "NORMAL"
                             and h2["level"] > lvl and (h2["touches"] >= min_touches or h2["prominent"])]
                    if above and c[i] > min(above):                # 閘(b) 換戰場=價值轉移到上方新帶
                        lv["state"] = "DEAD"
                    elif i - lv["spring_since"] > cfg["spring_ttl"]:  # 閘(c) 逾時未收復
                        lv["state"] = "DEAD"
                    elif c[i] < lvl - margin:   # 收復觸發：收盤強勢跌回原線 → 一律結算（不延遲，避免提早 V 轉變成延遲幽靈訊號）
                        if (lv["extreme"] > touch_ceil                                            # 已真掃止損
                                and (lv["extreme"] - lvl) / lvl >= cfg["spring_min_depth"]        # 深掃閘：針尖夠深才算 macro spring
                                and (i - lv.get("spring_dev_idx", lv["dev_idx"])) >= cfg["spring_min_bars"]):  # 時長閘：住夠久才算大級別
                            lv["dev_idx"] = lv.get("spring_dev_idx", lv["dev_idx"])
                            emit("B", "high", lv, i, lv["extreme"])
                        lv["state"] = "DEAD"                        # 收復後一律結束潛伏：達標 emit、未達標靜默死
            elif lv["state"] == "ZOMBIE":
                a = dev_atr_at(i)
                # 閘1 先判：更高新針尖(>=rearm_gap ATR) → 復活進第二級 DEVIATED（當根不再判換戰場）
                if a and h[i] >= lv["tip1"] + cfg["rearm_gap"] * a:
                    lv["state"] = "DEVIATED"; lv["level_no"] += 1
                    lv["dev_idx"] = i; lv["extreme"] = h[i]; lv["dev_atr"] = a
                    lv["ref2"] = h[i]; lv["acc_dev"] = max(0.0, c[i] - h[i]) / a
                    continue
                # TP 觸及(獵取成功) → 線永久死（作者：清掃完畢退場）
                if lv.get("last_opp") is not None and c[i] <= lv["last_opp"]:
                    lv["state"] = "DEAD"; continue
                # 換戰場：收盤突破上方最近有效聚集區 → 故事換到別條線，不復活舊線
                above = [h2["level"] for h2 in highs if h2 is not lv and h2["state"] == "NORMAL"
                         and h2["level"] > lv["level"] and (h2["touches"] >= min_touches or h2["prominent"])]
                if above and c[i] > min(above):
                    lv["state"] = "DEAD"; continue
                # 逾時
                if i - lv["zombie_since"] > cfg["zombie_ttl"]:
                    lv["state"] = "DEAD"

        # ---- 下方價帶：觸碰計數 + 獵取 ----
        for lv in lows:
            lvl = lv["level"]
            band_hi = max(lv["bmax"], lvl * (1 + eq_tol))
            touch_floor = lvl * (1 - touch_tol)
            if lv["state"] == "NORMAL":
                if c[i] < lvl:
                    # 收盤跌破線下 → B 型假跌破(deviation)
                    if established(lv, i, "low"):
                        if (lvl - c[i]) / lvl <= max_pierce:
                            lv["state"] = "DEVIATED"; lv["dev_idx"] = i; lv["extreme"] = low[i]
                            lv["dev_atr"] = dev_atr_at(i)
                            lv["acc_dev"] = (lvl - c[i]) / lv["dev_atr"] if lv["dev_atr"] else 0.0
                        else:
                            lv["state"] = "DEAD"          # 收太深 = 真跌破
                    elif (cfg.get("spring_enable") and (lvl - c[i]) / lvl <= max_pierce
                          and lv["touches"] >= cfg["spring_keepalive_touches"]):
                        pass   # 強帶僅因 min_hold 未過(established=False)收盤微破 → 保留 NORMAL 等站穩後真跌破(macro spring 鏡像)；不更新 last/touch_idxs
                    elif c[i] < touch_floor:
                        lv["state"] = "DEAD"              # 未建立/未站穩就明確收破 → 廢棄
                    # else: 微量收破(touch_tol 內)容忍，成形中的等低帶不被雜訊錯殺
                elif low[i] < touch_floor:
                    # 深影線插破容忍帶、收盤收回 → A 型影線獵取
                    _estL = established(lv, i, "low")
                    _pokL = (lvl - low[i]) / lvl <= max_pierce
                    if _estL and _pokL:
                        if rng > 0 and lower_wick / rng >= cfg["rej_wick"]:
                            emit("A", "low", lv, i, low[i])
                        lv["state"] = "DEAD"
                    elif (cfg.get("spring_enable") and not _estL and _pokL
                          and lv["touches"] >= cfg["spring_keepalive_touches"]):
                        pass   # 強帶未站穩被插破 → 保留 NORMAL 等真跌破(macro spring)
                    else:
                        lv["state"] = "DEAD"
                else:
                    # 其餘(含些微插針 touch_floor~lvl、或單純觸碰) → 觸碰計數
                    if lv["armed"] and low[i] <= band_hi and c[i] >= lvl:
                        lv["touches"] += 1; lv["armed"] = False; lv["last"] = i; lv["touch_idxs"].append(i)
                    elif (not lv["armed"]) and low[i] > lvl * (1 + sep_pct):
                        lv["armed"] = True
            elif lv["state"] == "DEVIATED":
                lv["extreme"] = min(lv["extreme"], low[i])
                bars = i - lv["dev_idx"]
                a0 = lv["dev_atr"]
                ref = lv.get("ref2", lvl)            # 第二級 acceptance 對新針尖量測；第一級對 lvl
                if a0:
                    lv["acc_dev"] += max(0.0, ref - c[i]) / a0
                margin = cfg["reclaim_atr"] * a0 if a0 else 0.0
                if c[i] < lvl * (1 - max_pierce):
                    lv["state"] = "DEAD"             # 崩盤硬上限，spring 不接管
                elif a0 and lv["acc_dev"] > cfg["max_acc_dev"]:
                    _to_spring_or_dead(lv, i, "low")  # acceptance 爆表=住下來=真跌破；強帶深掃→SPRING_WAIT 等收復
                elif bars > cfg["max_below"]:
                    _to_spring_or_dead(lv, i, "low")  # 超時未收復；強帶深掃→SPRING_WAIT 等收復
                elif bars >= cfg["min_below"] and c[i] > lvl + margin:   # 收復須站回線內側一個邊距
                    emitted = lv["extreme"] < touch_floor  # 極值須超出容忍帶=真的掃到止損，微量假跌破不算獵取
                    if emitted:
                        emit("B", "low", lv, i, lv["extreme"])
                    # 失敗的獵取(emit 完)進 ZOMBIE 潛伏，等是否被「更深新針尖」再獵取一級
                    if emitted and cfg.get("rearm_enable") and lv["level_no"] < cfg["max_level"]:
                        lv["state"] = "ZOMBIE"; lv["tip1"] = lv["extreme"]; lv["zombie_since"] = i
                    else:
                        lv["state"] = "DEAD"
            elif lv["state"] == "SPRING_WAIT":
                # 大級別假跌破(macro spring)：強帶深掃後潛伏，等收盤強勢站回原線才 emit
                lv["extreme"] = min(lv["extreme"], low[i])      # 持續追最深針尖(抓 1.748)
                a = dev_atr_at(i)
                margin = cfg["spring_reclaim_atr"] * a if a else 0.0
                if c[i] < lvl * (1 - cfg["spring_max_pierce"]):     # 閘(a) 收盤崩破=真跌破
                    lv["state"] = "DEAD"
                else:
                    below = [l2["level"] for l2 in lows if l2 is not lv and l2["state"] == "NORMAL"
                             and l2["level"] < lvl and (l2["touches"] >= min_touches or l2["prominent"])]
                    if below and c[i] < max(below):                # 閘(b) 換戰場=價值轉移到下方新帶
                        lv["state"] = "DEAD"
                    elif i - lv["spring_since"] > cfg["spring_ttl"]:  # 閘(c) 逾時未收復
                        lv["state"] = "DEAD"
                    elif c[i] > lvl + margin:   # 收復觸發：收盤強勢站回原線 → 一律結算（不延遲，避免提早 V 轉變成延遲幽靈訊號）
                        if (lv["extreme"] < touch_floor                                           # 已真掃止損
                                and (lvl - lv["extreme"]) / lvl >= cfg["spring_min_depth"]         # 深掃閘：針尖夠深才算 macro spring
                                and (i - lv.get("spring_dev_idx", lv["dev_idx"])) >= cfg["spring_min_bars"]):  # 時長閘：住夠久才算大級別
                            lv["dev_idx"] = lv.get("spring_dev_idx", lv["dev_idx"])
                            emit("B", "low", lv, i, lv["extreme"])     # extreme=最深針尖(1.748)、level=原線(2.141)
                        lv["state"] = "DEAD"                        # 收復後一律結束潛伏：達標 emit、未達標靜默死
            elif lv["state"] == "ZOMBIE":
                a = dev_atr_at(i)
                # 閘1 先判：更深新針尖(>=rearm_gap ATR) → 復活進第二級 DEVIATED（當根不再判換戰場）
                if a and low[i] <= lv["tip1"] - cfg["rearm_gap"] * a:
                    lv["state"] = "DEVIATED"; lv["level_no"] += 1
                    lv["dev_idx"] = i; lv["extreme"] = low[i]; lv["dev_atr"] = a
                    lv["ref2"] = low[i]; lv["acc_dev"] = max(0.0, low[i] - c[i]) / a
                    continue
                # TP 觸及(獵取成功) → 線永久死（作者：清掃完畢退場）
                if lv.get("last_opp") is not None and c[i] >= lv["last_opp"]:
                    lv["state"] = "DEAD"; continue
                # 換戰場：收盤跌破下方最近有效聚集區 → 故事換到別條線，不復活舊線
                below = [l2["level"] for l2 in lows if l2 is not lv and l2["state"] == "NORMAL"
                         and l2["level"] < lv["level"] and (l2["touches"] >= min_touches or l2["prominent"])]
                if below and c[i] < max(below):
                    lv["state"] = "DEAD"; continue
                # 逾時
                if i - lv["zombie_since"] > cfg["zombie_ttl"]:
                    lv["state"] = "DEAD"

        highs = [lv for lv in highs if lv["state"] != "DEAD"]
        lows = [lv for lv in lows if lv["state"] != "DEAD"]

    # 不做任何「用未來走勢判斷反轉」的過濾（那是 lookahead bias，監控版也做不到）。
    # 訊號在「當下這根 K 棒」pattern 完成時即成立：A=影線插破收回(單根)、B=收盤插破後收回(收復那根)。
    signals = _dedup(signals)
    signals.sort(key=lambda s: s["sweep_time"])
    return signals


def _dedup(signals):
    """同一根突破、同方向的重複訊號（相近價帶各觸發一次）只留觸碰數最多的那個。"""
    best = {}
    for s in signals:
        # 含 rearm_level：更深針尖常同根造新生線→新線第一級與殭屍線第二級同 sweep_idx，不可互相吃掉
        key = (s["side"], s["type"], s["sweep_idx"], s.get("rearm_level", 1))
        if key not in best or s["touches"] > best[key]["touches"]:
            best[key] = s
    return list(best.values())


# 美股波動較大，價帶/回撤/插破/容忍帶 門檻放寬
STOCK_OVERRIDES = {
    "1h": {"eq_tol": 0.009, "sep_pct": 0.022, "max_pierce": 0.15, "touch_tol": 0.010},
    "1d": {"eq_tol": 0.022, "sep_pct": 0.060, "max_pierce": 0.30, "touch_tol": 0.025, "max_below": 8},
}


def get_config(timeframe: str, market: str = "crypto", overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULTS[timeframe])
    if market == "stock":
        cfg.update(STOCK_OVERRIDES[timeframe])
    if overrides:
        cfg.update(overrides)
    return cfg
