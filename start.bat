@echo off
REM 流動性獵取掃描器 — 用 HTTP server 服務 scanner.html
REM 只綁 127.0.0.1：這個目錄下有 .cache_klines/ 與 scan_history/，
REM 綁 0.0.0.0 等於把整包快取對區網開放。
cd /d "%~dp0"
python -m http.server 5035 --bind 127.0.0.1
