import json
from openai import OpenAI
from database import get_tag_dislike_count
from models import SkipItState
from config import GPT_MODEL, OPENAI_API_KEY

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
    # 1. 取出三個子 agent 的結果；audio_output 可能是 None（無語音時 Orchestrator 會跳過 Audio Agent）
    meta = state["metadata_output"] or {}
    vision = state["vision_output"] or {}
    audio = state["audio_output"] or {}

    # 2. 彙整三方標籤，查詢每個標籤的歷史不喜歡次數（Preference Agent 累積的學習結果）
    all_tags = list(set(
        (meta.get("tags") or []) +
        (vision.get("tags") or []) +
        (audio.get("tags") or [])
    ))
    tag_weights = {tag: get_tag_dislike_count(tag) for tag in all_tags}
    tag_weight_str = "\n".join(f"  {t}: {c}次" for t, c in tag_weights.items()) or "  無"

    # 3. 組成 prompt 送給 GPT-4o；audio 沒有結果時用「無語音」代替，讓 prompt 依然合理
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

    # 4. 強制夾在 1-10 範圍，避免 GPT 偶爾回覆超出範圍的分數
    return {
        "score": max(1, min(10, int(parsed.get("score", 5)))),
        "summary": parsed.get("summary", ""),
        "tags": parsed.get("tags", []),
    }
