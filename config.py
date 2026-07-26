import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

DB_PATH = "data/skipit.db"
TMP_DIR = "tmp"

# yt-dlp 遇到 YouTube 要求登入驗證（"Please sign in"）時用來帶身分：
# 兩者擇一設定，YT_DLP_COOKIES_FILE 優先。都不設就維持原本匿名抓取（可能被 YouTube 擋）。
YT_DLP_COOKIES_FILE = os.environ.get("YT_DLP_COOKIES_FILE", "")        # cookies.txt 檔案路徑
YT_DLP_COOKIES_BROWSER = os.environ.get("YT_DLP_COOKIES_BROWSER", "")  # 瀏覽器名稱，例如 chrome / safari / firefox
GPT_MODEL = "gpt-4o"
WHISPER_MODEL = "whisper-1"
IMPLICIT_BLACKLIST_THRESHOLD = 3  # 同一頻道連續幾次 verdict=trash，自動加入封鎖頻道
MAX_CONCURRENT_ANALYSES = 3       # 同時最多跑幾支影片的完整分析
