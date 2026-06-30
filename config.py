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
FRAME_COUNT = 5
BLACKLIST_THRESHOLD = 3
LOW_SCORE_THRESHOLD = 4
TAG_DISLIKE_THRESHOLD = 5
