import json
from openai import OpenAI
from models import AgentState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

SCORING_PROMPT = """你是一個影片評分系統，要根據以下分析結果和使用者的個人品味，給出評分。

Metadata 訊號：{metadata_signals}
畫面訊號：{vision_signals}
語音訊號：{audio_signals}

使用者的品味檔案：
{taste_profile}

若品味檔案裡的「例外規則」或「喜歡的頻道」跟這支影片相關，請依照例外規則調整判斷，不要死板套用一般標準。

給出五個維度評分（每項 0-10，分數越高代表這個維度越沒問題/品質越好）：
- ai_generated：內容不是 AI 生成的程度（10=確定不是AI生成，0=高度確定AI生成）
- emotional_manipulation：沒有情緒操弄的程度（10=完全沒有雞湯/恐懼訴求話術，0=大量情緒操弄）
- originality：原創度（10=高度原創，0=完全搬運/重複）
- information_value：資訊價值（10=資訊密度高，0=毫無資訊價值）
- visual_quality：畫面品質（10=製作精良，0=粗製濫造）

回覆 JSON（不要加 markdown code block）:
{{
  "scores": {{"ai_generated": 0到10, "emotional_manipulation": 0到10, "originality": 0到10, "information_value": 0到10, "visual_quality": 0到10}},
  "overall_score": 0到100的整數,
  "verdict": "trash 或 review 或 keep",
  "summary": "一句話說明理由（20字以內）",
  "tags": ["最多4個標籤，例如：AI生成、雞湯、廣告、搬運"]
}}

verdict 判斷標準：overall_score < 40 是 trash，40-69 是 review，70 以上是 keep。"""


def run_scoring_agent(state: AgentState) -> dict:
    # 1. 彙整三個 agent 的 signals；沒跑過的 agent（例如無語音跳過）signals 會是 None
    metadata_signals = state.get("metadata_signals") or []
    vision_signals = state.get("vision_signals") or []
    audio_signals = state.get("audio_signals") or []
    taste_profile = state.get("taste_profile") or "（尚無品味檔案）"

    # 2. 組成 prompt，把 taste_profile 放進去做個人化評分
    prompt = SCORING_PROMPT.format(
        metadata_signals="、".join(metadata_signals) or "無",
        vision_signals="、".join(vision_signals) or "無",
        audio_signals="、".join(audio_signals) or "無語音",
        taste_profile=taste_profile,
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    # 3. verdict 強制限定在三個合法值內，GPT 偶爾亂回時 fallback 到 review
    verdict = parsed.get("verdict")
    if verdict not in ("trash", "review", "keep"):
        verdict = "review"

    return {
        "scores": parsed.get("scores", {}),
        "overall_score": max(0, min(100, int(parsed.get("overall_score", 50)))),
        "verdict": verdict,
        "summary": parsed.get("summary", ""),
        "tags": parsed.get("tags", []),
    }
