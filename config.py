import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

DB_PATH = "data/skipit.db"
TMP_DIR = "tmp"
GPT_MODEL = "gpt-4o"
WHISPER_MODEL = "whisper-1"
IMPLICIT_BLACKLIST_THRESHOLD = 3  # 同一頻道連續幾次 verdict=trash，自動加入封鎖頻道
MAX_CONCURRENT_ANALYSES = 3       # 同時最多跑幾支影片的完整分析
