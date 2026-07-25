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
  "summary": "一句話說明理由（20字以內）",
  "tags": ["<最多4個標籤，只能從上面提供的 metadata/畫面/語音訊號裡歸納出來，不要憑空套用常見標籤>"]
}}

tags 必須對應到上面實際列出的訊號，如果沒有訊號支持某個標籤就不要加。"""

# overall_score 跟 verdict 都不讓 LLM 自己回傳，改由程式碼從五個維度算出來，
# 保證數字之間永遠邏輯一致：
#   overall_score = 五個維度 0-10 分的平均 × 10（每個維度權重相同）
#   verdict：overall_score < 40 是 trash，40-69 是 review，70 以上是 keep

DIMENSIONS = ["ai_generated", "emotional_manipulation", "originality", "information_value", "visual_quality"]


def _clamp_scores(raw_scores: dict) -> dict:
    # 缺欄位預設 5（中性）、超出 0-10 範圍就夾住，避免單一離譜數字拉歪 overall_score
    return {dim: max(0, min(10, int(raw_scores.get(dim, 5)))) for dim in DIMENSIONS}


def _overall_score_from_dimensions(scores: dict) -> int:
    avg = sum(scores.values()) / len(scores)
    return max(0, min(100, round(avg * 10)))


def _verdict_from_score(overall_score: int) -> str:
    if overall_score < 40:
        return "trash"
    if overall_score < 70:
        return "review"
    return "keep"


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

    # 3. 五個維度先夾範圍，overall_score/verdict 全部從這五個數字算出來，不再信任 LLM 自己回報的總分
    scores = _clamp_scores(parsed.get("scores", {}))
    overall_score = _overall_score_from_dimensions(scores)

    return {
        "scores": scores,
        "overall_score": overall_score,
        "verdict": _verdict_from_score(overall_score),
        "summary": parsed.get("summary", ""),
        "tags": parsed.get("tags", []),
    }
