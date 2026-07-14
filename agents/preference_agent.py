from database import (
    get_creator_status,
    record_analysis,
    update_creator,
    increment_tag_dislike,
)
from models import SkipItState
from config import LOW_SCORE_THRESHOLD


def check_blacklist(state: SkipItState) -> dict:
    """
    分析前呼叫。查這個頻道是否在黑名單。
    若是，Orchestrator 會直接早停，不做完整分析。
    """
    creator_id = state.get("creator_id") or ""
    if not creator_id:
        return {"should_early_stop": False}

    status = get_creator_status(creator_id)
    return {"should_early_stop": status == "blacklist"}


def run_preference_agent(state: SkipItState) -> dict:
    """
    分析後呼叫。把這次結果存進 SQLite，並更新頻道累積狀態與標籤計數。
    """
    creator_id = state.get("creator_id") or ""
    score = state.get("score") or 5
    tags = state.get("tags") or []
    summary = state.get("summary") or ""
    url = state.get("url") or ""

    # 1. 存這次分析紀錄
    if creator_id:
        record_analysis(
            url=url,
            creator_id=creator_id,
            score=score,
            summary=summary,
            tags=tags,
        )

        # 2. 更新頻道累積分數與名單狀態
        #    累積 3 次 ≤ 4 分 → 自動升為 blacklist
        update_creator(creator_id, score)

    # 3. 低分影片的 tag 累積不喜歡次數
    #    Scoring Agent 之後會對高次數 tag 加重扣分
    if score <= LOW_SCORE_THRESHOLD:
        for tag in tags:
            increment_tag_dislike(tag)

    return {"preference_updated": True}
