"""
歷史掃描主程式：抓資料 → 偵測流動性獵取 → 產生互動網頁。

標的同時支援加密貨幣（Binance 永續）與美股（yfinance）。

用法：
    python scan.py                         # 預設清單，1h+1d
    python scan.py --crypto BTCUSDT ETHUSDT
    python scan.py --stocks MU NVDA
"""
import argparse
import webbrowser
from pathlib import Path

from data import fetch_binance_klines, fetch_stock
from detector import detect_sweeps, get_config
from report import build_report
from taker import attach_taker

BARS = {"1h": 6000, "1d": 1000}
DEFAULT_CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_STOCKS = ["MU", "NVDA"]
TIMEFRAMES = ["1d", "1h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crypto", nargs="*", default=DEFAULT_CRYPTO)
    ap.add_argument("--stocks", nargs="*", default=DEFAULT_STOCKS)
    ap.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    targets = ([(s, "crypto") for s in args.crypto] +
               [(s, "stock") for s in args.stocks])

    datasets = {}
    for sym, market in targets:
        for tf in args.timeframes:
            print(f"抓取 {sym} ({market}) {tf} ...", end=" ", flush=True)
            try:
                if market == "crypto":
                    df = fetch_binance_klines(sym, tf, BARS[tf])
                else:
                    df = fetch_stock(sym, tf, BARS[tf])
            except Exception as e:
                print(f"失敗：{e}")
                continue
            if df.empty:
                print("無資料，跳過")
                continue
            cfg = get_config(tf, market=market)
            signals = attach_taker(df, detect_sweeps(df, cfg))
            n_a = sum(1 for s in signals if s["type"] == "A")
            n_b = sum(1 for s in signals if s["type"] == "B")
            n_t = sum(1 for s in signals if s["taker_ok"])
            print(f"{len(df)} 根，獵取 {len(signals)} 次 (A {n_a} / B {n_b}，taker確認 {n_t})")
            cdf = df[["time", "open", "high", "low", "close", "volume"]].copy()
            cdf["volume"] = cdf["volume"].fillna(0.0)   # NaN 進 JSON 會炸，補 0
            candles = cdf.to_dict("records")
            datasets[f"{sym}|{tf}"] = {"candles": candles, "signals": signals}

    if not datasets:
        print("沒有任何資料，結束。")
        return

    out_path = Path(__file__).parent / args.out
    build_report(datasets, str(out_path))
    print(f"\n報告已產生：{out_path}")
    if not args.no_open:
        webbrowser.open(out_path.as_uri())


if __name__ == "__main__":
    main()
