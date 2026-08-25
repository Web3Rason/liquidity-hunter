"""
taker/CVD 確認（日線層）—— 依「貝格先生」方法論：『插破後強力 taker 介入即是確認訊號』。

⚠ crypto 的 ⚡ 已升級為 1 分 K 精準判定（taker_1m.py，在 market_scan/backtest 覆寫
taker_ok/taker_kind）；本模組的日線 z 在 crypto 降為輔助顯示，美股放量判定維持主判。

每根 delta = taker買 − taker賣 = 2*taker_buy_base − volume，
z = (delta − 60根滾動均值) / 60根滾動標準差（自適應不同幣種/時框的量級）。
窗口內過半 K 棒 delta=0（死幣/停牌）時 z 不可信，視為無資料。

1d 解析度下「強力介入」有兩種樣貌（實證 BTC/ETH/SOL 2023~2026 歷史訊號）：
- 吸收(absorption)：插破段「獵取方向」主動量爆發、價格仍收回
  ——止損潮(市價單)被大戶限價單接走；經典抄底插針(如 2024-08-05)全為此型
- 反轉(reversal)：插破~收復段「反方向」主動量爆發——作者描述的 CVD 反向爆量
注意：A 型單根時兩成分互為相反數，taker_z 等價於 |z|（「異常量+幾何收回」即確認）；
方向標籤 taker_kind 供肉眼/回測分組用。

門檻（Šidák 校正）：B 型窗口取 max 等於「看了 m 次」，雜訊超標機率隨 m 上升，
固定門檻會讓長窗口訊號白拿 ⚡。故按比較次數 m 校正：
    th(m) = Φ⁻¹((1−ALPHA)^(1/m))，ALPHA=0.12
m=2(A型) → th≈1.54（與實證單根 |z| p88≈1.5 一致）；m=13(B型6根) → th≈2.33。
taker_ok = taker_z >= th(m)。軟標記不硬砍，回測可重掃 ALPHA。

美股無 taker 資料 → 退而求其次用「放量」確認：log(成交量) 的 60 根滾動 z-score
（成交量右偏，log 後才接近常態），同窗口、同 Šidák 校正；無方向性、單成分，
kind 標「放量」。獵取要嘛沒人理、要嘛放量，至少濾掉無量的假收復。
成交量資料也沒有時 → 三欄皆 None（不封殺訊號，下游顯示「—」）。
只讀 sweep_idx 及更早的 K 棒，無前視。
1h 時框沿用同一套（z-score 自適應量級），但實證校準基於 1d，回測時應分開檢驗。
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

# 單次比較的目標誤報率：實證底噪單根 |z| p80≈1.1~1.4、p95≈2.1~2.5，取前 12% 為「異常」
ALPHA = 0.12
_ROLL = 60
_norm = NormalDist()


def _th(m: int) -> float:
    """看 m 次取 max 的 Šidák 校正門檻（常態近似）。"""
    return _norm.inv_cdf((1 - ALPHA) ** (1 / max(m, 1)))


def rearm_blocked(s: dict) -> bool:
    """線復活第二級(L2)的硬性 taker 閘（設計 §10.4；使用者裁決：crypto 放寬為「插破→收復整段窗口」判定）。
    rearm_level==2 須 taker_ok is True 才可交易/顯示——None 與 False 都剔除（「有前科」的線需額外保證）；
    第一級(L1)不受此閘。所有消費層(market_scan/report)統一呼叫，避免「三處各自實作」漂移。"""
    return int(s.get("rearm_level", 1)) == 2 and s.get("taker_ok") is not True


def attach_taker(df: pd.DataFrame, signals: list) -> list:
    """就地為每個訊號補上 taker_z / taker_kind / taker_ok 三欄。

    有 taker 資料（crypto）→ 吸收/反轉雙成分；只有成交量（美股）→ 放量單成分。
    """
    has_taker = "taker_buy_base" in df.columns and not df["taker_buy_base"].isna().all()
    has_vol = "volume" in df.columns and bool((df["volume"] > 0).any())
    if not has_taker and not has_vol:
        for s in signals:
            s["taker_z"] = None
            s["taker_kind"] = None
            s["taker_ok"] = None
        return signals

    if has_taker:
        delta = 2 * df["taker_buy_base"] - df["volume"]
        z = (delta - delta.rolling(_ROLL).mean()) / delta.rolling(_ROLL).std()
        active = (delta != 0).rolling(_ROLL).sum() >= _ROLL // 2   # 死幣/停牌段的 z 不可信
    else:
        lv = pd.Series(np.log(df["volume"].where(df["volume"] > 0)))  # 成交量右偏，log 後近常態
        z = (lv - lv.rolling(_ROLL).mean()) / lv.rolling(_ROLL).std()
        active = (df["volume"] > 0).rolling(_ROLL).sum() >= _ROLL // 2
    z = z.where(active).values

    for s in signals:
        i = s["sweep_idx"]
        if s["type"] == "A":
            j0, j1 = i, i + 1
        else:
            j0, j1 = s.get("dev_idx", i), i + 1
        if has_taker:
            rev = 1.0 if s["side"] == "low" else -1.0   # 反方向：看多=買盤(+)、看空=賣盤(−)
            comps = {"吸收": [(-rev) * v for v in z[j0:i]] if s["type"] == "B" else [-rev * z[i]],
                     "反轉": [rev * v for v in z[j0:j1]]}
        else:
            comps = {"放量": list(z[j0:j1])}            # 無方向性，整段看有沒有爆量
        # NaN 先濾再取 max（max 遇 NaN 在首位會被毒化，不能事後濾）
        comps = {k: [v for v in vs if not math.isnan(v)] for k, vs in comps.items()}
        m = sum(len(vs) for vs in comps.values())       # 實際比較次數（門檻校正用）
        vals = [(max(vs), k) for k, vs in comps.items() if vs]
        if not vals:                                    # 開頭暖機段 / 死幣段
            s["taker_z"] = None
            s["taker_kind"] = None
            s["taker_ok"] = None
            continue
        tz, kind = max(vals, key=lambda x: x[0])
        s["taker_z"] = round(tz, 2)
        s["taker_kind"] = kind
        s["taker_ok"] = bool(tz >= _th(m))
    return signals
