@echo off
REM 5035 流動性獵取 每日全市場掃描 + 新增命中推播
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo ==== %date% %time% ==== >> scan.log
python -W ignore market_scan.py --days 7 --notify >> scan.log 2>&1
