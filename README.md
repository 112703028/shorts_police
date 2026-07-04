# shorts_police

## 系統需求

除了 `pip install -r requirements.txt` 之外，還需要另外安裝：

- **ffmpeg / ffprobe**（`downloader.py` 用 subprocess 直接呼叫，抽音軌、截幀都靠它）
  - Windows: `choco install ffmpeg` 或到 https://ffmpeg.org/download.html 下載後加入 PATH
  - macOS: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`
- `.env` 檔（參考 `.env.example`）填入 `OPENAI_API_KEY`、`LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN`

