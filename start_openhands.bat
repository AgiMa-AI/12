@echo off
echo 正在启动OpenHands智能健康管家...
start "" "http://localhost:8000"
python start_openhands.py --local