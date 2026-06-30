from typing import TypedDict, Optional


class AgentOutput(TypedDict):
    agent: str          # "metadata" | "vision" | "audio" | "scoring"
    result: str         # 分析結果文字
    confidence: float   # 0.0 - 1.0
    tags: list[str]     # e.g. ["廣告", "AI生成", "無資訊價值"]


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
