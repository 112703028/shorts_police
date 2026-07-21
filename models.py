from typing import TypedDict, Optional


class AgentState(TypedDict):
    user_id: str
    url: str
    creator_id: Optional[str]
    creator_name: Optional[str]
    video_path: Optional[str]
    frames: Optional[list[str]]
    transcript: Optional[str]
    metadata_signals: Optional[list[str]]
    vision_signals: Optional[list[str]]
    audio_signals: Optional[list[str]]
    tags: Optional[list[str]]
    scores: Optional[dict]          # ai_generated / emotional_manipulation / originality / information_value / visual_quality
    overall_score: Optional[int]
    verdict: Optional[str]          # "trash" | "review" | "keep"
    summary: Optional[str]
    taste_profile: str              # 使用者品味檔案（自然語言全文）
    user_feedback: Optional[str]    # 使用者回饋原文（👍/👎 或文字說明）
    should_early_stop: bool         # 命中封鎖頻道，提早終止
    skip_audio: bool                # 影片無語音，跳過 Audio Agent
    needs_reflection: bool          # Vision / Audio 結果矛盾，觸發二次 reflection
