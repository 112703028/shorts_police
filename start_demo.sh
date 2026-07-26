#!/bin/bash
# Demo 當天一鍵啟動：開兩個 Terminal 視窗，一個跑 FastAPI (main.py)、一個跑 ngrok tunnel。
# 用法：./start_demo.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 免費 ngrok 帳號送的固定 domain，設定一次後 LINE 後台的 Webhook URL 就不用再改
NGROK_DOMAIN="jogger-viscosity-carnage.ngrok-free.dev"

osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$SCRIPT_DIR' && source .venv/bin/activate && python main.py"
    do script "cd '$SCRIPT_DIR' && eval \"\$(/opt/homebrew/bin/brew shellenv)\" && ngrok http --domain=$NGROK_DOMAIN 8000"
end tell
EOF

echo "已開啟兩個 Terminal 視窗：main.py (FastAPI) 和 ngrok tunnel。"
echo "Webhook URL 固定為：https://$NGROK_DOMAIN/webhook（LINE 後台只需設定過一次）"
