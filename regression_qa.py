# 偵測器修訂回歸測試：人工標注確認的假陽性要死、教科書案例要活（永久測試，動偵測器必跑）
import sys
from pathlib import Path

import pandas as pd

from detector import detect_sweeps, get_config
from taker import attach_taker

# K 線快取放在專案目錄下。**由 backtest.py 產生**（第一次跑會抓 S&P500 + NASDAQ100
# 五年日線與全市場永續，要數十分鐘）。所以這支回歸測試必須先跑過 backtest.py 才有東西可讀。
CACHE = str(Path(__file__).resolve().parent / ".cache_klines")

# (market, symbol, side, sweep日期, 預期[, rearm_level]) — 預期 die=修正後應消失, live=應存活
#   第 6 欄 rearm_level（選填）：指定須命中的級別（2=線復活的第二級獵取）
CASES = [
    ("stock",  "CDNS", "high", "2026-06-05", "die"),   # +10% 站5根 真突破
    ("stock",  "FSLR", "high", "2026-06-05", "die"),   # +12% 站6-8根
    ("crypto", "INJUSDT", "high", "2026-06-04", "die"),  # +20% 站一週
    ("crypto", "ZAMAUSDT", "high", "2026-06-02", "die"),  # +13% 站近一週
    ("crypto", "LITUSDT", "high", "2026-06-04", "die"),  # 線誕生於密集區(birth_tan=10>5)：
    #   依「線須明顯」原則排除（曾是 acceptance 積分的深插速回對照組，該角色由合成測試承接）
    ("stock",  "WMT", "low", "2026-06-05", "live"),    # 教科書
    ("stock",  "TSCO", "low", "2026-06-04", "live"),   # 教科書
    ("stock",  "INTC", "low", "2026-06-08", "live"),   # 教科書
    ("stock",  "ACGL", "low", "2026-06-05", "live"),   # 原12月帶(btan=2)被門檻擋掉後，
    #   1月中誕生的新帶(btan=0、4次確認觸碰)完全合規——訊號由乾淨帶承接，應存活
    ("crypto", "BLESSUSDT", "low", "2026-06-06", "die"),  # 殭屍幣假等低(birth_tan=6>5)
    ("crypto", "CRVUSDT", "low", "2026-06-10", "live", 2),  # 線復活：0.20 線 06-03 第一級用過後，
    #   06-10 被更深針尖(0.1961→0.17)再獵取一級，收復舊線→emit 第二級（修「emit 一次就死」缺陷）
    ("crypto", "QTUMUSDT", "low", "2020-11-17", "live"),  # macro spring：2.141 強等低帶(touches=5)跌破後
    #   住約3週、深掃到1.748(-18%)、2020-11-17 強勢收盤站回原線=Wyckoff Spring（補抓非3~5根V轉的大週期獵取）
]

_stocks_pkl = Path(CACHE) / "stocks_5y.pkl"
if not _stocks_pkl.exists():
    sys.exit(
        f"找不到 K 線快取：{_stocks_pkl}\n"
        "這支回歸測試吃 backtest.py 產生的快取。請先跑一次：\n"
        "    python backtest.py\n"
        "（第一次會抓 S&P500 + NASDAQ100 五年日線與全市場永續，需數十分鐘）"
    )

stocks = pd.read_pickle(str(_stocks_pkl))
ok = bad = 0
for case in CASES:
    market, sym, side, date, expect = case[:5]
    want_level = case[5] if len(case) > 5 else None   # 選填：須命中的級別（2=線復活第二級）
    df = stocks.get(sym) if market == "stock" else pd.read_pickle(f"{CACHE}/{sym}_1d.pkl")
    sigs = attach_taker(df, detect_sweeps(df, get_config("1d", market)))
    hit = any(s["type"] == "B" and s["side"] == side and s["touches"] >= 2 and
              (want_level is None or s.get("rearm_level", 1) == want_level) and
              str(pd.Timestamp(int(s["sweep_time"]), unit="s").date()) == date
              for s in sigs)   # touches>=2 = 正式管線(scanner/backtest)的條件，回歸對齊生產行為
    status = "存在" if hit else "消失"
    if expect == "info":
        print(f"  [觀察] {sym:10s} B{'高' if side=='high' else '低'} {date} → {status}（已知限制，不計分）")
        continue
    want = (expect == "live") == hit
    mark = "OK" if want else "FAIL"
    ok += want
    bad += not want
    print(f"  [{mark}] {sym:10s} B{'高' if side=='high' else '低'} {date} 預期{'存活' if expect=='live' else '消失'} → {status}")
print(f"\n{ok}/{ok + bad} 符合預期" + ("" if bad == 0 else f"（{bad} 項不符）"))
sys.exit(0 if bad == 0 else 1)
