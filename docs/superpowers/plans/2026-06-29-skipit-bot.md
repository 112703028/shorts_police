# SkipIt Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LINE Bot (SkipIt Bot) that receives YouTube Shorts links in a group chat, runs a multi-agent LangGraph pipeline to judge whether the video is a "廢片", and replies with a 1–10 score, five-dimension analysis, and one-line reason.

**Architecture:** A FastAPI webhook server receives LINE messages, extracts YouTube URLs, and feeds them into a LangGraph graph. The Orchestrator node dynamically routes to Metadata, Vision, and Audio sub-agents based on video properties and blacklist state. A Scoring node aggregates all outputs; a Preference node updates SQLite history.

**Tech Stack:** Python 3.11+, FastAPI, line-bot-sdk v3, LangGraph, langchain-openai, openai (Whisper + GPT-4o), yt-dlp, SQLite (stdlib), pytest, ngrok (local tunnel)

## Global Constraints

- Python >= 3.11
- All OpenAI calls use model `gpt-4o`
- All Whisper calls use model `whisper-1`
- Agent output schema must match `AgentOutput` TypedDict defined in Task 1
- LangGraph state must use `SkipItState` TypedDict defined in Task 1
- LINE SDK v3 only (`linebot.v3.*`), not v2
- SQLite DB file at `data/skipit.db`
- Environment variables loaded from `.env`: `OPENAI_API_KEY`, `LINE_CHANNEL_SECRET`, `LINE_CHANNEL_ACCESS_TOKEN`

---

## File Map

```
skipit_bot/
├── config.py              # Env vars, constants
├── database.py            # SQLite schema + CRUD
├── downloader.py          # yt-dlp: download video, extract frames, extract audio
├── models.py              # Shared TypedDicts: AgentOutput, SkipItState
├── agents/
│   ├── __init__.py
│   ├── metadata_agent.py  # YouTube metadata via yt-dlp (吳廷翰)
│   ├── vision_agent.py    # GPT-4o frame analysis (Tim)
│   ├── audio_agent.py     # Whisper transcription + analysis (Tim)
│   ├── scoring_agent.py   # Score aggregation (Tim)
│   └── preference_agent.py # Blacklist R/W (吳廷翰)
├── graph.py               # LangGraph graph assembly (吳廷翰)
├── line_bot.py            # FastAPI + LINE webhook (Tim)
├── main.py                # Entrypoint: uvicorn
├── data/                  # SQLite DB (gitignored)
├── tmp/                   # Downloaded videos/frames (gitignored)
└── tests/
    ├── test_database.py
    ├── test_downloader.py
    ├── test_metadata_agent.py
    ├── test_vision_agent.py
    ├── test_audio_agent.py
    ├── test_scoring_agent.py
    ├── test_preference_agent.py
    ├── test_graph.py
    └── test_line_bot.py
```

---

## Task 1: Project Setup — Config, Models, Database

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config.py`
- Create: `models.py`
- Create: `database.py`
- Create: `tests/test_database.py`
- Create: `data/.gitkeep`, `tmp/.gitkeep`

**Interfaces:**
- Produces:
  - `AgentOutput` TypedDict (used by all agents)
  - `SkipItState` TypedDict (used by LangGraph graph)
  - `database.init_db()` → `None`
  - `database.get_creator_status(creator_id: str)` → `str | None`  ("blacklist" | "watchlist" | "whitelist" | None)
  - `database.record_analysis(url, creator_id, score, summary, tags)` → `None`
  - `database.update_creator(creator_id, score)` → `None`
  - `database.get_tag_dislike_count(tag: str)` → `int`
  - `database.increment_tag_dislike(tag: str)` → `None`

- [ ] **Step 1: Create requirements.txt**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
line-bot-sdk>=3.11.0
langgraph>=0.2.0
langchain-openai>=0.1.0
openai>=1.35.0
yt-dlp>=2024.5.0
python-dotenv>=1.0.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
Pillow>=10.0.0
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-...
LINE_CHANNEL_SECRET=...
LINE_CHANNEL_ACCESS_TOKEN=...
```

- [ ] **Step 3: Create config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]

DB_PATH = "data/skipit.db"
TMP_DIR = "tmp"
GPT_MODEL = "gpt-4o"
WHISPER_MODEL = "whisper-1"
FRAME_COUNT = 5
BLACKLIST_THRESHOLD = 3      # 累積幾次低分升為 blacklist
LOW_SCORE_THRESHOLD = 4      # 幾分以下算低分
TAG_DISLIKE_THRESHOLD = 5    # tag 累積幾次加重扣分
```

- [ ] **Step 4: Create models.py**

```python
from typing import TypedDict, Optional

class AgentOutput(TypedDict):
    agent: str           # "metadata" | "vision" | "audio" | "scoring"
    result: str          # 分析結果文字
    confidence: float    # 0.0 - 1.0
    tags: list[str]      # e.g. ["廣告", "AI生成", "無資訊價值"]

class SkipItState(TypedDict):
    url: str
    creator_id: Optional[str]
    creator_name: Optional[str]
    video_path: Optional[str]
    audio_path: Optional[str]
    frame_paths: Optional[list[str]]
    metadata_output: Optional[AgentOutput]
    vision_output: Optional[AgentOutput]
    audio_output: Optional[AgentOutput]
    should_early_stop: bool
    skip_audio: bool
    vision_retry_count: int
    score: Optional[int]
    summary: Optional[str]
    tags: Optional[list[str]]
    preference_updated: bool
```

- [ ] **Step 5: Write failing tests for database**

```python
# tests/test_database.py
import pytest
import os
from database import init_db, get_creator_status, record_analysis, update_creator, get_tag_dislike_count, increment_tag_dislike

TEST_DB = "data/test_skipit.db"

@pytest.fixture(autouse=True)
def setup_db(monkeypatch):
    monkeypatch.setenv("DB_PATH_OVERRIDE", TEST_DB)
    init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

def test_new_creator_status_is_none():
    assert get_creator_status("UCunknown", TEST_DB) is None

def test_update_creator_low_score_becomes_watchlist():
    update_creator("UC123", 3, TEST_DB)
    assert get_creator_status("UC123", TEST_DB) == "watchlist"

def test_update_creator_three_low_scores_becomes_blacklist():
    for _ in range(3):
        update_creator("UC456", 3, TEST_DB)
    assert get_creator_status("UC456", TEST_DB) == "blacklist"

def test_record_analysis_stores_row():
    record_analysis("https://yt.be/abc", "UC789", 7, "OK video", ["資訊"], TEST_DB)
    status = get_creator_status("UC789", TEST_DB)
    assert status is None  # score 7, not low

def test_tag_dislike_starts_at_zero():
    assert get_tag_dislike_count("廣告", TEST_DB) == 0

def test_increment_tag_dislike():
    increment_tag_dislike("廣告", TEST_DB)
    assert get_tag_dislike_count("廣告", TEST_DB) == 1
```

- [ ] **Step 6: Run tests — expect FAIL**

```
pytest tests/test_database.py -v
```
Expected: `ModuleNotFoundError: No module named 'database'`

- [ ] **Step 7: Create database.py**

```python
import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH, BLACKLIST_THRESHOLD, LOW_SCORE_THRESHOLD

def _conn(db_path: str = None) -> sqlite3.Connection:
    path = db_path or os.environ.get("DB_PATH_OVERRIDE") or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return sqlite3.connect(path)

def init_db(db_path: str = None) -> None:
    with _conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id          INTEGER PRIMARY KEY,
                url         TEXT,
                creator_id  TEXT,
                score       INTEGER,
                summary     TEXT,
                tags        TEXT,
                analyzed_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS creator_preferences (
                creator_id  TEXT PRIMARY KEY,
                status      TEXT,
                avg_score   REAL,
                count       INTEGER,
                updated_at  TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS tag_preferences (
                tag           TEXT PRIMARY KEY,
                dislike_count INTEGER DEFAULT 0,
                updated_at    TIMESTAMP
            );
        """)

def get_creator_status(creator_id: str, db_path: str = None) -> str | None:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM creator_preferences WHERE creator_id = ?",
            (creator_id,)
        ).fetchone()
    return row[0] if row else None

def record_analysis(url: str, creator_id: str, score: int, summary: str, tags: list[str], db_path: str = None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO analysis_history (url, creator_id, score, summary, tags, analyzed_at) VALUES (?,?,?,?,?,?)",
            (url, creator_id, score, summary, json.dumps(tags, ensure_ascii=False), datetime.now())
        )

def update_creator(creator_id: str, score: int, db_path: str = None) -> None:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT avg_score, count, status FROM creator_preferences WHERE creator_id = ?",
            (creator_id,)
        ).fetchone()
        if row:
            avg, count, _ = row
            new_avg = (avg * count + score) / (count + 1)
            new_count = count + 1
            low_score_rows = conn.execute(
                "SELECT COUNT(*) FROM analysis_history WHERE creator_id = ? AND score <= ?",
                (creator_id, LOW_SCORE_THRESHOLD)
            ).fetchone()[0]
            new_status = "blacklist" if low_score_rows >= BLACKLIST_THRESHOLD else "watchlist"
            conn.execute(
                "UPDATE creator_preferences SET avg_score=?, count=?, status=?, updated_at=? WHERE creator_id=?",
                (new_avg, new_count, new_status, datetime.now(), creator_id)
            )
        else:
            status = "watchlist" if score <= LOW_SCORE_THRESHOLD else None
            if status:
                conn.execute(
                    "INSERT INTO creator_preferences (creator_id, status, avg_score, count, updated_at) VALUES (?,?,?,?,?)",
                    (creator_id, status, float(score), 1, datetime.now())
                )

def get_tag_dislike_count(tag: str, db_path: str = None) -> int:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT dislike_count FROM tag_preferences WHERE tag = ?",
            (tag,)
        ).fetchone()
    return row[0] if row else 0

def increment_tag_dislike(tag: str, db_path: str = None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO tag_preferences (tag, dislike_count, updated_at) VALUES (?,1,?) "
            "ON CONFLICT(tag) DO UPDATE SET dislike_count=dislike_count+1, updated_at=?",
            (tag, datetime.now(), datetime.now())
        )
```

- [ ] **Step 8: Run tests — expect PASS**

```
pytest tests/test_database.py -v
```
Expected: 6 passed

- [ ] **Step 9: Create data/ and tmp/ directories + gitkeep**

```bash
mkdir -p data tmp
echo "" > data/.gitkeep
echo "" > tmp/.gitkeep
```

- [ ] **Step 10: Commit**

```bash
git init
git add requirements.txt .env.example config.py models.py database.py tests/test_database.py data/.gitkeep tmp/.gitkeep
git commit -m "feat: project setup — config, models, database schema"
```

---

## Task 2: Shared Downloader

**Files:**
- Create: `downloader.py`
- Create: `tests/test_downloader.py`

**Interfaces:**
- Consumes: `config.TMP_DIR`, `config.FRAME_COUNT`
- Produces:
  - `download_video(url: str) -> Path` — downloads to `tmp/{video_id}.mp4`
  - `extract_frames(video_path: Path, n: int = FRAME_COUNT) -> list[Path]` — returns list of JPEG paths
  - `extract_audio(video_path: Path) -> Path` — returns MP3 path
  - `has_audio_track(video_path: Path) -> bool`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_downloader.py
import pytest
from pathlib import Path
from downloader import download_video, extract_frames, extract_audio, has_audio_track

# Use a real short YouTube Shorts URL for integration test
YT_URL = "https://www.youtube.com/shorts/dQw4w9WgXcQ"  # replace with a stable short video

def test_download_video_returns_existing_path():
    path = download_video(YT_URL)
    assert path.exists()
    assert path.suffix == ".mp4"

def test_extract_frames_returns_n_images(tmp_path):
    video = download_video(YT_URL)
    frames = extract_frames(video, n=3)
    assert len(frames) == 3
    for f in frames:
        assert f.exists()
        assert f.suffix == ".jpg"

def test_extract_audio_returns_mp3(tmp_path):
    video = download_video(YT_URL)
    audio = extract_audio(video)
    assert audio.exists()
    assert audio.suffix == ".mp3"

def test_has_audio_track_true():
    video = download_video(YT_URL)
    assert has_audio_track(video) is True
```

- [ ] **Step 2: Run test — expect FAIL**

```
pytest tests/test_downloader.py -v
```
Expected: `ModuleNotFoundError: No module named 'downloader'`

- [ ] **Step 3: Create downloader.py**

```python
import subprocess
import hashlib
from pathlib import Path
import yt_dlp
from config import TMP_DIR, FRAME_COUNT

def _video_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]

def download_video(url: str) -> Path:
    out_dir = Path(TMP_DIR)
    out_dir.mkdir(exist_ok=True)
    vid_id = _video_id(url)
    out_path = out_dir / f"{vid_id}.mp4"
    if out_path.exists():
        return out_path
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out_path

def extract_frames(video_path: Path, n: int = FRAME_COUNT) -> list[Path]:
    out_dir = video_path.parent / f"{video_path.stem}_frames"
    out_dir.mkdir(exist_ok=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) == n:
        return frames
    # Get duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    interval = duration / (n + 1)
    frame_paths = []
    for i in range(1, n + 1):
        ts = interval * i
        out_file = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run([
            "ffmpeg", "-ss", str(ts), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out_file), "-y", "-loglevel", "error"
        ], check=True)
        frame_paths.append(out_file)
    return frame_paths

def extract_audio(video_path: Path) -> Path:
    out_path = video_path.with_suffix(".mp3")
    if out_path.exists():
        return out_path
    subprocess.run([
        "ffmpeg", "-i", str(video_path), "-vn",
        "-acodec", "libmp3lame", "-q:a", "4", str(out_path), "-y", "-loglevel", "error"
    ], check=True)
    return out_path

def has_audio_track(video_path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True
    )
    return "audio" in result.stdout
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_downloader.py -v
```
Expected: 4 passed (requires ffmpeg installed)

- [ ] **Step 5: Commit**

```bash
git add downloader.py tests/test_downloader.py
git commit -m "feat: shared downloader — yt-dlp video download, frame/audio extraction"
```

---

## Task 3: Metadata Agent (吳廷翰)

**Files:**
- Create: `agents/metadata_agent.py`
- Create: `agents/__init__.py`
- Create: `tests/test_metadata_agent.py`

**Interfaces:**
- Consumes: `models.AgentOutput`, `models.SkipItState`
- Produces:
  - `run_metadata_agent(state: SkipItState) -> dict` — returns partial state update:
    `{"metadata_output": AgentOutput, "creator_id": str, "creator_name": str}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metadata_agent.py
import pytest
from unittest.mock import patch, MagicMock
from agents.metadata_agent import run_metadata_agent
from models import SkipItState

BASE_STATE: SkipItState = {
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": None, "creator_name": None,
    "video_path": None, "audio_path": None, "frame_paths": None,
    "metadata_output": None, "vision_output": None, "audio_output": None,
    "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
    "score": None, "summary": None, "tags": None, "preference_updated": False,
}

MOCK_INFO = {
    "title": "超可愛貓咪影片",
    "description": "每天分享貓咪 #cat #cute",
    "uploader": "CatChannel",
    "channel_id": "UCcat123",
    "view_count": 50000,
    "like_count": 1200,
    "duration": 30,
    "tags": ["cat", "cute"],
}

def test_metadata_agent_returns_creator_id():
    with patch("agents.metadata_agent.yt_dlp.YoutubeDL") as mock_ydl:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = MOCK_INFO
        result = run_metadata_agent(BASE_STATE)
    assert result["creator_id"] == "UCcat123"

def test_metadata_agent_returns_agent_output():
    with patch("agents.metadata_agent.yt_dlp.YoutubeDL") as mock_ydl:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = MOCK_INFO
        result = run_metadata_agent(BASE_STATE)
    output = result["metadata_output"]
    assert output["agent"] == "metadata"
    assert isinstance(output["tags"], list)
    assert 0.0 <= output["confidence"] <= 1.0
```

- [ ] **Step 2: Run — expect FAIL**

```
pytest tests/test_metadata_agent.py -v
```

- [ ] **Step 3: Create agents/__init__.py (empty)**

```python
```

- [ ] **Step 4: Create agents/metadata_agent.py**

```python
import yt_dlp
from openai import OpenAI
from models import AgentOutput, SkipItState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

METADATA_PROMPT = """你是一個影片品質分析師。根據以下 YouTube Shorts 的 metadata，判斷這支影片是否為「廢片」。

Metadata:
標題: {title}
描述: {description}
頻道: {uploader}
觀看數: {view_count}
按讚數: {like_count}
時長: {duration} 秒
標籤: {tags}

請回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "result": "一句話描述這支影片的內容類型",
  "confidence": 0.0到1.0的信心分數,
  "tags": ["最多3個廢片標籤，例如：廣告、AI生成、無資訊價值、重複內容、純娛樂"]
}}
若無廢片特徵，tags 回空陣列。"""

def run_metadata_agent(state: SkipItState) -> dict:
    opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(state["url"], download=False)

    prompt = METADATA_PROMPT.format(
        title=info.get("title", ""),
        description=(info.get("description") or "")[:500],
        uploader=info.get("uploader", ""),
        view_count=info.get("view_count", 0),
        like_count=info.get("like_count", 0),
        duration=info.get("duration", 0),
        tags=", ".join(info.get("tags", [])),
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    import json
    parsed = json.loads(response.choices[0].message.content)

    output: AgentOutput = {
        "agent": "metadata",
        "result": parsed.get("result", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "tags": parsed.get("tags", []),
    }

    return {
        "metadata_output": output,
        "creator_id": info.get("channel_id", ""),
        "creator_name": info.get("uploader", ""),
    }
```

- [ ] **Step 5: Run tests — expect PASS**

```
pytest tests/test_metadata_agent.py -v
```

- [ ] **Step 6: Commit**

```bash
git add agents/__init__.py agents/metadata_agent.py tests/test_metadata_agent.py
git commit -m "feat: metadata agent — yt-dlp metadata extraction + GPT-4o analysis"
```

---

## Task 4: Vision Agent (Tim)

**Files:**
- Create: `agents/vision_agent.py`
- Create: `tests/test_vision_agent.py`

**Interfaces:**
- Consumes: `downloader.download_video`, `downloader.extract_frames`, `models.SkipItState`
- Produces:
  - `run_vision_agent(state: SkipItState) -> dict` — returns:
    `{"vision_output": AgentOutput, "frame_paths": list[str], "video_path": str}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_vision_agent.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from agents.vision_agent import run_vision_agent
from models import SkipItState

BASE_STATE: SkipItState = {
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": "UCtest", "creator_name": "TestChan",
    "video_path": None, "audio_path": None, "frame_paths": None,
    "metadata_output": None, "vision_output": None, "audio_output": None,
    "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
    "score": None, "summary": None, "tags": None, "preference_updated": False,
}

MOCK_GPT_RESPONSE = '{"result": "AI 生成動物影片，畫面重複", "confidence": 0.85, "tags": ["AI生成", "無資訊價值"]}'

def test_vision_agent_returns_vision_output():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake_image_bytes"

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 5), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT_RESPONSE
        result = run_vision_agent(BASE_STATE)

    assert result["vision_output"]["agent"] == "vision"
    assert result["vision_output"]["confidence"] == 0.85
    assert "AI生成" in result["vision_output"]["tags"]

def test_vision_agent_low_confidence_flag():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake"
    low_conf = '{"result": "模糊畫面", "confidence": 0.2, "tags": []}'

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 5), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = low_conf
        result = run_vision_agent(BASE_STATE)

    assert result["vision_output"]["confidence"] < 0.3
```

- [ ] **Step 2: Run — expect FAIL**

```
pytest tests/test_vision_agent.py -v
```

- [ ] **Step 3: Create agents/vision_agent.py**

```python
import base64
import json
from pathlib import Path
from openai import OpenAI
from downloader import download_video, extract_frames
from models import AgentOutput, SkipItState
from config import GPT_MODEL, OPENAI_API_KEY, FRAME_COUNT

_client = OpenAI(api_key=OPENAI_API_KEY)

VISION_PROMPT = """你是一個影片品質分析師。以下是一支 YouTube Shorts 的 {n} 張截圖（依時間順序）。

請判斷這支影片是否為「廢片」，回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "result": "一句話描述畫面內容",
  "confidence": 0.0到1.0的信心分數（若畫面模糊或無法判斷請給0.3以下）,
  "tags": ["最多3個廢片標籤，例如：AI生成、畫質差、重複畫面、無字幕、純特效"]
}}"""

def run_vision_agent(state: SkipItState) -> dict:
    video_path = download_video(state["url"])
    frames = extract_frames(video_path, n=FRAME_COUNT)

    images = []
    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode()
        images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT.format(n=len(frames))},
                *images,
            ],
        }],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    output: AgentOutput = {
        "agent": "vision",
        "result": parsed.get("result", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "tags": parsed.get("tags", []),
    }

    return {
        "vision_output": output,
        "video_path": str(video_path),
        "frame_paths": [str(f) for f in frames],
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_vision_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/vision_agent.py tests/test_vision_agent.py
git commit -m "feat: vision agent — GPT-4o frame analysis"
```

---

## Task 5: Audio Agent (Tim)

**Files:**
- Create: `agents/audio_agent.py`
- Create: `tests/test_audio_agent.py`

**Interfaces:**
- Consumes: `downloader.extract_audio`, `models.SkipItState`
- Produces:
  - `run_audio_agent(state: SkipItState) -> dict` — returns:
    `{"audio_output": AgentOutput, "audio_path": str}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio_agent.py
import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from agents.audio_agent import run_audio_agent
from models import SkipItState

BASE_STATE: SkipItState = {
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": "UCtest", "creator_name": "TestChan",
    "video_path": "tmp/test.mp4", "audio_path": None, "frame_paths": None,
    "metadata_output": None, "vision_output": None, "audio_output": None,
    "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
    "score": None, "summary": None, "tags": None, "preference_updated": False,
}

MOCK_TRANSCRIPT = "今天來開箱這個超厲害的產品，買就對了，限時優惠..."
MOCK_GPT = '{"result": "廣告推銷話術，誇大宣傳", "confidence": 0.9, "tags": ["廣告", "推銷"]}'

def test_audio_agent_returns_output():
    with patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = MOCK_TRANSCRIPT
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_audio_agent(BASE_STATE)

    assert result["audio_output"]["agent"] == "audio"
    assert "廣告" in result["audio_output"]["tags"]
    assert result["audio_output"]["confidence"] == 0.9

def test_audio_agent_sets_audio_path():
    with patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = MOCK_TRANSCRIPT
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_audio_agent(BASE_STATE)

    assert result["audio_path"] == "tmp/test.mp3"
```

- [ ] **Step 2: Run — expect FAIL**

```
pytest tests/test_audio_agent.py -v
```

- [ ] **Step 3: Create agents/audio_agent.py**

```python
import json
from pathlib import Path
from openai import OpenAI
from downloader import extract_audio
from models import AgentOutput, SkipItState
from config import GPT_MODEL, WHISPER_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_PROMPT = """你是一個影片品質分析師。以下是一支 YouTube Shorts 的語音轉錄文字：

「{transcript}」

請判斷這段語音內容是否為「廢片」，回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "result": "一句話描述語音內容",
  "confidence": 0.0到1.0的信心分數,
  "tags": ["最多3個廢片標籤，例如：廣告、推銷、無意義重複、機翻字幕、純音樂無內容"]
}}"""

def run_audio_agent(state: SkipItState) -> dict:
    video_path = Path(state["video_path"])
    audio_path = extract_audio(video_path)

    with open(audio_path, "rb") as f:
        transcript = _client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            language="zh",
        ).text

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": AUDIO_PROMPT.format(transcript=transcript[:2000])}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    output: AgentOutput = {
        "agent": "audio",
        "result": parsed.get("result", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "tags": parsed.get("tags", []),
    }

    return {
        "audio_output": output,
        "audio_path": str(audio_path),
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_audio_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/audio_agent.py tests/test_audio_agent.py
git commit -m "feat: audio agent — Whisper transcription + GPT-4o content analysis"
```

---

## Task 6: Scoring Agent (Tim)

**Files:**
- Create: `agents/scoring_agent.py`
- Create: `tests/test_scoring_agent.py`

**Interfaces:**
- Consumes: `models.SkipItState` (reads metadata_output, vision_output, audio_output), `database.get_tag_dislike_count`
- Produces:
  - `run_scoring_agent(state: SkipItState) -> dict` — returns:
    `{"score": int, "summary": str, "tags": list[str]}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scoring_agent.py
import pytest
from unittest.mock import patch
from agents.scoring_agent import run_scoring_agent
from models import SkipItState, AgentOutput

def _make_state(metadata_tags=[], vision_conf=0.8, audio_tags=[], vision_tags=[]) -> SkipItState:
    return {
        "url": "https://yt.be/x", "creator_id": "UC1", "creator_name": "Chan",
        "video_path": "tmp/x.mp4", "audio_path": "tmp/x.mp3", "frame_paths": [],
        "metadata_output": AgentOutput(agent="metadata", result="test meta", confidence=0.7, tags=metadata_tags),
        "vision_output": AgentOutput(agent="vision", result="test vision", confidence=vision_conf, tags=vision_tags),
        "audio_output": AgentOutput(agent="audio", result="test audio", confidence=0.8, tags=audio_tags),
        "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
        "score": None, "summary": None, "tags": None, "preference_updated": False,
    }

MOCK_SCORE_RESPONSE = '{"score": 3, "summary": "廣告推銷影片，無資訊價值", "tags": ["廣告", "推銷"]}'

def test_scoring_agent_returns_score():
    state = _make_state(metadata_tags=["廣告"], audio_tags=["推銷"])
    with patch("agents.scoring_agent._client") as mock_client, \
         patch("agents.scoring_agent.get_tag_dislike_count", return_value=0):
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_SCORE_RESPONSE
        result = run_scoring_agent(state)
    assert result["score"] == 3
    assert isinstance(result["summary"], str)
    assert isinstance(result["tags"], list)

def test_scoring_agent_score_in_range():
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client, \
         patch("agents.scoring_agent.get_tag_dislike_count", return_value=0):
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_SCORE_RESPONSE
        result = run_scoring_agent(state)
    assert 1 <= result["score"] <= 10
```

- [ ] **Step 2: Run — expect FAIL**

```
pytest tests/test_scoring_agent.py -v
```

- [ ] **Step 3: Create agents/scoring_agent.py**

```python
import json
from openai import OpenAI
from database import get_tag_dislike_count
from models import AgentOutput, SkipItState
from config import GPT_MODEL, OPENAI_API_KEY, TAG_DISLIKE_THRESHOLD

_client = OpenAI(api_key=OPENAI_API_KEY)

SCORING_PROMPT = """你是一個影片廢片評分系統。根據以下三個分析結果，給出最終評分。

Metadata 分析: {metadata_result} (信心: {meta_conf}, 標籤: {meta_tags})
畫面分析: {vision_result} (信心: {vision_conf}, 標籤: {vision_tags})
語音分析: {audio_result} (信心: {audio_conf}, 標籤: {audio_tags})

常見廢片標籤的累積不喜歡次數（越高越要扣分）:
{tag_weights}

評分標準（1=極度廢片，10=高品質）:
- 1-3: 廣告/推銷/AI生成/無任何資訊價值
- 4-5: 純娛樂但重複或無新意
- 6-7: 有一定娛樂或資訊價值
- 8-10: 高品質、有深度或高度娛樂性

回覆 JSON（不要加 markdown code block）:
{{
  "score": 1到10的整數,
  "summary": "一句話說明評分理由（15字以內）",
  "tags": ["最終廢片標籤列表，最多4個"]
}}"""

def run_scoring_agent(state: SkipItState) -> dict:
    meta = state["metadata_output"] or {}
    vision = state["vision_output"] or {}
    audio = state["audio_output"] or {}

    all_tags = list(set(
        (meta.get("tags") or []) +
        (vision.get("tags") or []) +
        (audio.get("tags") or [])
    ))
    tag_weights = {tag: get_tag_dislike_count(tag) for tag in all_tags}
    tag_weight_str = "\n".join(f"  {t}: {c}次" for t, c in tag_weights.items()) or "  無"

    prompt = SCORING_PROMPT.format(
        metadata_result=meta.get("result", "無"),
        meta_conf=meta.get("confidence", 0),
        meta_tags=", ".join(meta.get("tags") or []),
        vision_result=vision.get("result", "無"),
        vision_conf=vision.get("confidence", 0),
        vision_tags=", ".join(vision.get("tags") or []),
        audio_result=audio.get("result", "無語音"),
        audio_conf=audio.get("confidence", 0),
        audio_tags=", ".join(audio.get("tags") or []),
        tag_weights=tag_weight_str,
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    return {
        "score": max(1, min(10, int(parsed.get("score", 5)))),
        "summary": parsed.get("summary", ""),
        "tags": parsed.get("tags", []),
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_scoring_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/scoring_agent.py tests/test_scoring_agent.py
git commit -m "feat: scoring agent — multi-source aggregation with tag weight adjustment"
```

---

## Task 7: Preference Agent (吳廷翰)

**Files:**
- Create: `agents/preference_agent.py`
- Create: `tests/test_preference_agent.py`

**Interfaces:**
- Consumes: `database.*`, `models.SkipItState`
- Produces:
  - `run_preference_agent(state: SkipItState) -> dict` — returns:
    `{"preference_updated": True}`
  - `check_blacklist(state: SkipItState) -> dict` — returns:
    `{"should_early_stop": bool}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preference_agent.py
import pytest
import os
from database import init_db, update_creator
from agents.preference_agent import run_preference_agent, check_blacklist
from models import SkipItState

TEST_DB = "data/test_pref.db"

@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setenv("DB_PATH_OVERRIDE", TEST_DB)
    init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

def _state(creator_id="UC1", score=None, tags=None) -> SkipItState:
    return {
        "url": "https://yt.be/x", "creator_id": creator_id, "creator_name": "Chan",
        "video_path": None, "audio_path": None, "frame_paths": None,
        "metadata_output": None, "vision_output": None, "audio_output": None,
        "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
        "score": score, "summary": "test", "tags": tags or [], "preference_updated": False,
    }

def test_check_blacklist_returns_false_for_new_creator():
    result = check_blacklist(_state("UCnew"))
    assert result["should_early_stop"] is False

def test_check_blacklist_returns_true_for_blacklisted():
    for _ in range(3):
        update_creator("UCbad", 2, TEST_DB)
    result = check_blacklist(_state("UCbad"))
    assert result["should_early_stop"] is True

def test_preference_agent_updates_db():
    state = _state("UCtest", score=3, tags=["廣告"])
    run_preference_agent(state)
    from database import get_creator_status
    assert get_creator_status("UCtest", TEST_DB) == "watchlist"

def test_preference_agent_returns_updated_flag():
    state = _state("UCx", score=5, tags=[])
    result = run_preference_agent(state)
    assert result["preference_updated"] is True
```

- [ ] **Step 2: Run — expect FAIL**

```
pytest tests/test_preference_agent.py -v
```

- [ ] **Step 3: Create agents/preference_agent.py**

```python
from database import (
    get_creator_status, record_analysis,
    update_creator, increment_tag_dislike
)
from models import SkipItState
from config import LOW_SCORE_THRESHOLD

def check_blacklist(state: SkipItState) -> dict:
    creator_id = state.get("creator_id") or ""
    if not creator_id:
        return {"should_early_stop": False}
    status = get_creator_status(creator_id)
    return {"should_early_stop": status == "blacklist"}

def run_preference_agent(state: SkipItState) -> dict:
    creator_id = state.get("creator_id") or ""
    score = state.get("score") or 5
    tags = state.get("tags") or []

    if creator_id:
        record_analysis(
            url=state["url"],
            creator_id=creator_id,
            score=score,
            summary=state.get("summary") or "",
            tags=tags,
        )
        update_creator(creator_id, score)

    if score <= LOW_SCORE_THRESHOLD:
        for tag in tags:
            increment_tag_dislike(tag)

    return {"preference_updated": True}
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_preference_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add agents/preference_agent.py tests/test_preference_agent.py
git commit -m "feat: preference agent — blacklist check and preference learning"
```

---

## Task 8: Orchestrator + LangGraph Graph Assembly (吳廷翰)

**Files:**
- Create: `graph.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: all agents, `models.SkipItState`
- Produces:
  - `build_graph()` → `CompiledGraph`
  - `run_pipeline(url: str) -> dict` — returns `{"score", "summary", "tags", "creator_name", "should_early_stop"}`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph.py
import pytest
from unittest.mock import patch, MagicMock
from graph import build_graph, run_pipeline

MOCK_META = {"metadata_output": {"agent":"metadata","result":"test","confidence":0.7,"tags":[]}, "creator_id": "UC1", "creator_name": "Chan"}
MOCK_VISION = {"vision_output": {"agent":"vision","result":"ok","confidence":0.8,"tags":[]}, "video_path": "tmp/x.mp4", "frame_paths": []}
MOCK_AUDIO = {"audio_output": {"agent":"audio","result":"ok","confidence":0.8,"tags":[]}, "audio_path": "tmp/x.mp3"}
MOCK_SCORE = {"score": 7, "summary": "正常影片", "tags": []}
MOCK_PREF = {"preference_updated": True}

def test_pipeline_returns_score():
    with patch("graph.run_metadata_agent", return_value=MOCK_META), \
         patch("graph.run_vision_agent", return_value=MOCK_VISION), \
         patch("graph.run_audio_agent", return_value=MOCK_AUDIO), \
         patch("graph.run_scoring_agent", return_value=MOCK_SCORE), \
         patch("graph.run_preference_agent", return_value=MOCK_PREF), \
         patch("graph.check_blacklist", return_value={"should_early_stop": False}), \
         patch("graph.has_audio_track", return_value=True):
        result = run_pipeline("https://yt.be/test")
    assert result["score"] == 7

def test_pipeline_early_stops_on_blacklist():
    with patch("graph.check_blacklist", return_value={"should_early_stop": True}):
        result = run_pipeline("https://yt.be/test")
    assert result["should_early_stop"] is True
    assert result["score"] == 1

def test_pipeline_skips_audio_when_no_audio():
    audio_called = []
    def mock_audio(state):
        audio_called.append(True)
        return MOCK_AUDIO

    with patch("graph.run_metadata_agent", return_value=MOCK_META), \
         patch("graph.run_vision_agent", return_value=MOCK_VISION), \
         patch("graph.run_audio_agent", side_effect=mock_audio), \
         patch("graph.run_scoring_agent", return_value=MOCK_SCORE), \
         patch("graph.run_preference_agent", return_value=MOCK_PREF), \
         patch("graph.check_blacklist", return_value={"should_early_stop": False}), \
         patch("graph.has_audio_track", return_value=False):
        run_pipeline("https://yt.be/test")
    assert len(audio_called) == 0
```

- [ ] **Step 2: Run — expect FAIL**

```
pytest tests/test_graph.py -v
```

- [ ] **Step 3: Create graph.py**

```python
from pathlib import Path
from langgraph.graph import StateGraph, END
from models import SkipItState
from agents.metadata_agent import run_metadata_agent
from agents.vision_agent import run_vision_agent
from agents.audio_agent import run_audio_agent
from agents.scoring_agent import run_scoring_agent
from agents.preference_agent import run_preference_agent, check_blacklist
from downloader import has_audio_track
from database import init_db

init_db()

def _orchestrate(state: SkipItState) -> dict:
    """Decide skip_audio based on downloaded video."""
    video_path = state.get("video_path")
    if video_path and not has_audio_track(Path(video_path)):
        return {"skip_audio": True}
    return {"skip_audio": False}

def _vision_with_retry(state: SkipItState) -> dict:
    result = run_vision_agent(state)
    conf = result.get("vision_output", {}).get("confidence", 1.0)
    if conf < 0.3 and state.get("vision_retry_count", 0) < 1:
        result["vision_retry_count"] = state.get("vision_retry_count", 0) + 1
        import shutil
        frames_dir = Path(result["video_path"]).parent / f"{Path(result['video_path']).stem}_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        result = run_vision_agent({**state, **result})
    return result

def _early_stop_node(state: SkipItState) -> dict:
    return {"score": 1, "summary": "頻道已列入黑名單，自動跳過", "tags": ["黑名單"]}

def _route_after_blacklist(state: SkipItState) -> str:
    return "early_stop" if state["should_early_stop"] else "metadata"

def _route_after_orchestrate(state: SkipItState) -> str:
    return "vision_only" if state["skip_audio"] else "vision_and_audio"

def _route_after_vision(state: SkipItState) -> str:
    skip = state.get("skip_audio", False)
    return "scoring" if skip else "audio"

def build_graph():
    graph = StateGraph(SkipItState)

    graph.add_node("check_blacklist", check_blacklist)
    graph.add_node("early_stop", _early_stop_node)
    graph.add_node("metadata", run_metadata_agent)
    graph.add_node("orchestrate", _orchestrate)
    graph.add_node("vision", _vision_with_retry)
    graph.add_node("audio", run_audio_agent)
    graph.add_node("scoring", run_scoring_agent)
    graph.add_node("preference", run_preference_agent)

    graph.set_entry_point("check_blacklist")
    graph.add_conditional_edges("check_blacklist", _route_after_blacklist, {
        "early_stop": "early_stop",
        "metadata": "metadata",
    })
    graph.add_edge("early_stop", "preference")
    graph.add_edge("metadata", "orchestrate")
    graph.add_conditional_edges("orchestrate", _route_after_orchestrate, {
        "vision_only": "vision",
        "vision_and_audio": "vision",
    })
    graph.add_conditional_edges("vision", _route_after_vision, {
        "audio": "audio",
        "scoring": "scoring",
    })
    graph.add_edge("audio", "scoring")
    graph.add_edge("scoring", "preference")
    graph.add_edge("preference", END)

    return graph.compile()

_compiled_graph = build_graph()

def run_pipeline(url: str) -> dict:
    initial: SkipItState = {
        "url": url,
        "creator_id": None, "creator_name": None,
        "video_path": None, "audio_path": None, "frame_paths": None,
        "metadata_output": None, "vision_output": None, "audio_output": None,
        "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
        "score": None, "summary": None, "tags": None, "preference_updated": False,
    }
    final = _compiled_graph.invoke(initial)
    return {
        "score": final.get("score"),
        "summary": final.get("summary"),
        "tags": final.get("tags"),
        "creator_name": final.get("creator_name"),
        "should_early_stop": final.get("should_early_stop", False),
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```
pytest tests/test_graph.py -v
```

- [ ] **Step 5: Commit**

```bash
git add graph.py tests/test_graph.py
git commit -m "feat: LangGraph pipeline — orchestrator with dynamic routing and blacklist early-stop"
```

---

## Task 9: LINE Bot + FastAPI Server (Tim)

**Files:**
- Create: `line_bot.py`
- Create: `main.py`
- Create: `tests/test_line_bot.py`

**Interfaces:**
- Consumes: `graph.run_pipeline`
- Produces: FastAPI app at `POST /webhook`, `GET /health`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_line_bot.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from line_bot import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_extract_youtube_url_from_text():
    from line_bot import extract_youtube_url
    text = "看這個 https://www.youtube.com/shorts/abc123 超好笑"
    assert extract_youtube_url(text) == "https://www.youtube.com/shorts/abc123"

def test_extract_youtube_url_returns_none_for_non_url():
    from line_bot import extract_youtube_url
    assert extract_youtube_url("hello world") is None

def test_format_reply_score():
    from line_bot import format_reply
    result = format_reply(score=3, summary="廣告推銷影片", tags=["廣告", "推銷"],
                          creator_name="TestChan", early_stop=False)
    assert "3/10" in result
    assert "廣告推銷影片" in result

def test_format_reply_early_stop():
    from line_bot import format_reply
    result = format_reply(score=1, summary="黑名單頻道", tags=["黑名單"],
                          creator_name="SpamChan", early_stop=True)
    assert "黑名單" in result
```

- [ ] **Step 2: Run — expect FAIL**

```
pytest tests/test_line_bot.py -v
```

- [ ] **Step 3: Create line_bot.py**

```python
import re
import asyncio
from fastapi import FastAPI, Request, Header, HTTPException, BackgroundTasks
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from graph import run_pipeline
from config import LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN

app = FastAPI()
handler = WebhookHandler(LINE_CHANNEL_SECRET)
_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

YT_SHORTS_PATTERN = re.compile(
    r"https?://(?:www\.)?youtube\.com/shorts/[\w-]+"
    r"|https?://youtu\.be/[\w-]+"
)

def extract_youtube_url(text: str) -> str | None:
    m = YT_SHORTS_PATTERN.search(text)
    return m.group(0) if m else None

def format_reply(score: int, summary: str, tags: list[str],
                 creator_name: str, early_stop: bool) -> str:
    score_emoji = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
    label = "廢片" if score <= 4 else "普通" if score <= 6 else "好片"
    tag_str = " ".join(f"#{t}" for t in tags) if tags else "無"
    early_note = "\n⛔ 頻道已列入黑名單，略過完整分析" if early_stop else ""
    return (
        f"{score_emoji} {label}（{score}/10）\n"
        f"📝 {summary}\n"
        f"🏷️ {tag_str}\n"
        f"👤 {creator_name or '未知頻道'}"
        f"{early_note}"
    )

def _reply(reply_token: str, text: str) -> None:
    with ApiClient(_config) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text)],
            )
        )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_line_signature: str = Header(...),
):
    body = await request.body()
    try:
        handler.handle(body.decode(), x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    url = extract_youtube_url(event.message.text)
    if not url:
        return
    _reply(event.reply_token, "🔍 分析中，請稍後...")

    def _analyze():
        try:
            result = run_pipeline(url)
            text = format_reply(
                score=result["score"],
                summary=result["summary"],
                tags=result["tags"] or [],
                creator_name=result.get("creator_name") or "",
                early_stop=result.get("should_early_stop", False),
            )
        except Exception as e:
            text = f"❌ 分析失敗：{str(e)[:100]}"
        # Use push message for async reply (reply token already used)
        with ApiClient(_config) as api_client:
            MessagingApi(api_client).push_message(
                to=event.source.group_id or event.source.user_id,
                messages=[TextMessage(text=text)],
            )

    import threading
    threading.Thread(target=_analyze, daemon=True).start()
```

- [ ] **Step 4: Create main.py**

```python
import uvicorn
from line_bot import app
from database import init_db

if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 5: Run tests — expect PASS**

```
pytest tests/test_line_bot.py -v
```

- [ ] **Step 6: Commit**

```bash
git add line_bot.py main.py tests/test_line_bot.py
git commit -m "feat: LINE bot webhook — message handler, URL extraction, formatted reply"
```

---

## Task 10: Integration Test + ngrok Setup

**Files:**
- Create: `tests/test_integration.py`
- Create: `start_demo.sh`

**Interfaces:**
- Consumes: all tasks
- Validates: end-to-end pipeline with real YouTube Shorts URL

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""
End-to-end integration test. Requires real API keys in .env.
Run with: pytest tests/test_integration.py -v -m integration
"""
import pytest
from graph import run_pipeline

WASTE_URL = "https://www.youtube.com/shorts/REPLACE_WITH_REAL_WASTE_URL"
GOOD_URL  = "https://www.youtube.com/shorts/REPLACE_WITH_REAL_GOOD_URL"

@pytest.mark.integration
def test_waste_video_scores_low():
    result = run_pipeline(WASTE_URL)
    assert result["score"] is not None
    assert 1 <= result["score"] <= 10
    assert isinstance(result["summary"], str)
    print(f"\nWaste video score: {result['score']} — {result['summary']}")

@pytest.mark.integration
def test_good_video_scores_high():
    result = run_pipeline(GOOD_URL)
    assert result["score"] is not None
    print(f"\nGood video score: {result['score']} — {result['summary']}")
```

- [ ] **Step 2: Create start_demo.sh**

```bash
#!/bin/bash
# Install ngrok first: https://ngrok.com/download
# Set your LINE webhook URL in LINE Developers Console to the ngrok HTTPS URL + /webhook

echo "Starting SkipIt Bot demo..."
echo "1. Starting FastAPI server on port 8000..."
python main.py &
SERVER_PID=$!

echo "2. Starting ngrok tunnel..."
ngrok http 8000 &
NGROK_PID=$!

echo "Server PID: $SERVER_PID"
echo "ngrok PID: $NGROK_PID"
echo ""
echo "Copy the ngrok HTTPS URL and set it in LINE Developers Console:"
echo "  Webhook URL: https://xxxx.ngrok-free.app/webhook"
echo ""
echo "Press Ctrl+C to stop."
wait
```

- [ ] **Step 3: Run unit tests (all tasks)**

```
pytest tests/ -v --ignore=tests/test_integration.py
```
Expected: all unit tests pass

- [ ] **Step 4: Run integration test with real URLs**

Replace `REPLACE_WITH_REAL_*_URL` in `test_integration.py` with actual YouTube Shorts URLs, then:

```
pytest tests/test_integration.py -v -m integration
```

- [ ] **Step 5: Start demo locally**

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY, LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN
bash start_demo.sh
```

- [ ] **Step 6: Final commit**

```bash
git add tests/test_integration.py start_demo.sh
git commit -m "feat: integration test and demo startup script"
```

---

## Self-Review

**Spec coverage:**
- ✅ LINE Bot receives YouTube Shorts URL → Task 9
- ✅ Orchestrator dynamic routing (blacklist early-stop, skip audio) → Task 8
- ✅ Metadata Agent (yt-dlp) → Task 3
- ✅ Vision Agent (GPT-4o frames) → Task 4
- ✅ Audio Agent (Whisper) → Task 5
- ✅ Scoring Agent (1–10 + summary + tags) → Task 6
- ✅ Preference Agent (blacklist learning) → Task 7
- ✅ SQLite schema (analysis_history, creator_preferences, tag_preferences) → Task 1
- ✅ Shared downloader (yt-dlp + ffmpeg) → Task 2
- ✅ Reply format with score + emoji + tags → Task 9

**Type consistency:**
- `AgentOutput` defined in Task 1, used consistently in Tasks 3–7
- `SkipItState` defined in Task 1, used consistently in Tasks 3–9
- `run_pipeline` returns `{"score", "summary", "tags", "creator_name", "should_early_stop"}` — matches `format_reply` call in Task 9 ✅
