#!/usr/bin/env bash
# 前台启动面板（调试用；生产建议用 systemd）
cd "$(dirname "$0")/../panel"
exec python3 panel.py
