import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH, BLACKLIST_THRESHOLD, LOW_SCORE_THRESHOLD


def _conn(db_path: str = None) -> sqlite3.Connection:
    # 優先用傳入的 db_path，其次是測試用的環境變數覆寫，最後才是正式環境的 DB_PATH
    path = db_path or os.environ.get("DB_PATH_OVERRIDE") or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return sqlite3.connect(path)


def init_db(db_path: str = None) -> None:
    # 三張表：分析紀錄、頻道偏好（黑/觀察名單）、標籤不喜歡次數
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
            CREATE TABLE IF NOT EXISTS taste_profile (
                id         INTEGER PRIMARY KEY CHECK (id = 1),
                content    TEXT,
                updated_at TIMESTAMP
            );
        """)


def get_creator_status(creator_id: str, db_path: str = None) -> str | None:
    # 回傳 "blacklist" | "watchlist" | None，Orchestrator 用這個決定要不要提早終止分析
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
    # 每次分析完都呼叫，動態更新頻道的平均分與名單狀態（黑名單學習機制）
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT avg_score, count, status FROM creator_preferences WHERE creator_id = ?",
            (creator_id,)
        ).fetchone()
        if row:
            avg, count, _ = row
            new_avg = (avg * count + score) / (count + 1)
            new_count = count + 1
            # 累積低分次數達到門檻才升級為黑名單，否則只是觀察名單
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
            # 第一次出現的頻道：只有低分才建檔列入觀察名單，正常分數不用記
            status = "watchlist" if score <= LOW_SCORE_THRESHOLD else None
            if status:
                conn.execute(
                    "INSERT INTO creator_preferences (creator_id, status, avg_score, count, updated_at) VALUES (?,?,?,?,?)",
                    (creator_id, status, float(score), 1, datetime.now())
                )


def get_tag_dislike_count(tag: str, db_path: str = None) -> int:
    # Scoring Agent 會用這個數字加重扣分：某標籤被討厭越多次，代表使用者越不喜歡這類影片
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT dislike_count FROM tag_preferences WHERE tag = ?",
            (tag,)
        ).fetchone()
    return row[0] if row else 0


def increment_tag_dislike(tag: str, db_path: str = None) -> None:
    # 標籤第一次出現就 insert，之後每次都 +1（ON CONFLICT UPSERT）
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO tag_preferences (tag, dislike_count, updated_at) VALUES (?,1,?) "
            "ON CONFLICT(tag) DO UPDATE SET dislike_count=dislike_count+1, updated_at=?",
            (tag, datetime.now(), datetime.now())
        )


DEFAULT_TASTE_PROFILE = """【使用者品味檔案】
（尚無使用者回饋，使用預設標準）
- 討厭：純廣告推銷、營銷號搬運內容、無資訊價值的 AI 生成影片
- 一般標準：娛樂性和資訊價值至少要有一項"""


def get_taste_profile(db_path: str = None) -> str:
    # Scoring Agent 評分前讀取；Feedback Agent 學習時的修改基礎
    with _conn(db_path) as conn:
        row = conn.execute("SELECT content FROM taste_profile WHERE id = 1").fetchone()
    return row[0] if row else DEFAULT_TASTE_PROFILE


def save_taste_profile(content: str, db_path: str = None) -> None:
    # Feedback Agent 重寫品味檔案後存回（全系統只有一份，UPSERT 到 id=1）
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO taste_profile (id, content, updated_at) VALUES (1,?,?) "
            "ON CONFLICT(id) DO UPDATE SET content=?, updated_at=?",
            (content, datetime.now(), content, datetime.now())
        )


def get_last_analysis(db_path: str = None) -> dict | None:
    # 使用者回饋「這個不錯」時，「這個」指的就是最近一筆分析結果
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT url, creator_id, score, summary, tags FROM analysis_history "
            "ORDER BY analyzed_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    return {
        "url": row[0],
        "creator_id": row[1],
        "score": row[2],
        "summary": row[3],
        "tags": json.loads(row[4]) if row[4] else [],
    }


def set_creator_status(creator_id: str, status: str, db_path: str = None) -> None:
    # Feedback Agent 依使用者指示直接改頻道狀態（例如「封鎖這個頻道」→ blacklist）
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO creator_preferences (creator_id, status, avg_score, count, updated_at) "
            "VALUES (?,?,0,0,?) "
            "ON CONFLICT(creator_id) DO UPDATE SET status=?, updated_at=?",
            (creator_id, status, datetime.now(), status, datetime.now())
        )
