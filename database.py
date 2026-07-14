import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime
from config import DB_PATH


@contextmanager
def _conn(db_path: str = None):
    # sqlite3.Connection 內建的 context manager 只處理 commit/rollback、不會關閉連線，
    # 在 Windows 上會導致檔案句柄不釋放，這裡自己包一層確保 close()
    path = db_path or os.environ.get("DB_PATH_OVERRIDE") or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = None) -> None:
    # 兩張表：使用者品味檔案（自然語言全文）、分析歷史紀錄
    with _conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS taste_profiles (
                user_id     TEXT PRIMARY KEY,
                profile     TEXT,
                updated_at  TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS analysis_history (
                id            INTEGER PRIMARY KEY,
                user_id       TEXT,
                url           TEXT,
                creator_id    TEXT,
                verdict       TEXT,
                overall_score INTEGER,
                summary       TEXT,
                tags          TEXT,
                analyzed_at   TIMESTAMP
            );
        """)


def get_taste_profile(user_id: str, db_path: str = None) -> str | None:
    # 回傳 None 代表使用者從未互動過，Orchestrator 用這個判斷要不要觸發首次使用問卷
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT profile FROM taste_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row[0] if row else None


def save_taste_profile(user_id: str, profile: str, db_path: str = None) -> None:
    # preference_agent 每次重寫完 taste_profile 都呼叫這個存回去（第一次是 insert，之後都是覆寫）
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO taste_profiles (user_id, profile, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET profile=excluded.profile, updated_at=excluded.updated_at",
            (user_id, profile, datetime.now())
        )


def record_analysis(user_id: str, url: str, creator_id: str, verdict: str,
                     overall_score: int, summary: str, tags: list[str], db_path: str = None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO analysis_history (user_id, url, creator_id, verdict, overall_score, summary, tags, analyzed_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (user_id, url, creator_id, verdict, overall_score, summary,
             json.dumps(tags, ensure_ascii=False), datetime.now())
        )


def count_consecutive_trash(user_id: str, creator_id: str, db_path: str = None) -> int:
    # 從最新紀錄往回數連續幾支被判 trash，遇到非 trash 就停
    # preference_agent 用這個判斷要不要觸發隱性黑名單訊號（見 v2 spec）
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT verdict FROM analysis_history WHERE user_id = ? AND creator_id = ? ORDER BY id DESC",
            (user_id, creator_id)
        ).fetchall()
    count = 0
    for (verdict,) in rows:
        if verdict != "trash":
            break
        count += 1
    return count
