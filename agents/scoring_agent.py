import json
from openai import OpenAI
from models import AgentState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

SCORING_PROMPT = """你是一個影片評分系統，要根據以下分析結果和使用者的個人品味，給出評分。

Metadata 訊號：{metadata_signals}
畫面訊號：{vision_signals}
畫面內容描述：{vision_description}
語音訊號：{audio_signals}
語音逐字稿：{transcript}

使用者的品味檔案：
{taste_profile}

若品味檔案裡的「例外規則」或「喜歡的頻道」跟這支影片相關，請依照例外規則調整判斷，不要死板套用一般標準。

請比對「畫面內容描述」跟「語音逐字稿」是否對得上（例如畫面是可愛寵物但語音在講完全無關的話題、語音內容明顯跟畫面時序兜不起來），
這種落差代表可能是拼接或內容農場產物，屬於可疑訊號；如果兩者本來就沒有強關聯（例如純知識分享搭配任意背景畫面），不要硬套成問題。
若語音逐字稿是「（無逐字稿）」或影片本身沒有語音，代表根本沒有語音內容可以比對，content_mismatch 一律回 false，
不要把「沒有語音」這件事本身當成不一致。
{reflection_block}
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
  "tags": ["<最多4個標籤，只能從上面提供的 metadata/畫面/語音訊號裡歸納出來，不要憑空套用常見標籤>"],
  "content_mismatch": true 或 false,
  "mismatch_reason": "<content_mismatch 是 true 時，一句話說明畫面跟語音哪裡對不上；false 就回空字串>"
}}

tags 必須對應到上面實際列出的訊號，如果沒有訊號支持某個標籤就不要加。"""

REFLECTION_BLOCK = """
【二次檢視】上一輪你判斷畫面與語音內容疑似不一致（原因：{mismatch_reason}），
請重新仔細比對兩者是否真的兜不起來，並據此調整評分——如果確認是拼接/農場內容，
originality 和 information_value 應該對應調低；如果重新檢視後覺得其實還好，也可以維持原判斷。
"""

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
    vision_description = state.get("vision_description") or "（無描述）"
    transcript = state.get("transcript") or "（無逐字稿）"

    # needs_reflection 是 Orchestrator 偵測到上一輪 content_mismatch=True 後才會設，
    # 這裡把上一輪判斷的落差原因帶回 prompt，讓這次真的是「針對那個落差重新檢視」，
    # 不是盲目把同一個問題再問一次
    reflection_block = ""
    if state.get("needs_reflection"):
        reflection_block = REFLECTION_BLOCK.format(mismatch_reason=state.get("mismatch_reason", ""))

    # 2. 組成 prompt，把 taste_profile 放進去做個人化評分
    prompt = SCORING_PROMPT.format(
        metadata_signals="、".join(metadata_signals) or "無",
        vision_signals="、".join(vision_signals) or "無",
        vision_description=vision_description,
        audio_signals="、".join(audio_signals) or "無語音",
        transcript=transcript[:1000],
        taste_profile=taste_profile,
        reflection_block=reflection_block,
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content)

    # 3. 五個維度先夾範圍，overall_score/verdict 全部從這五個數字算出來，不再信任 LLM 自己回報的總分
    scores = _clamp_scores(parsed.get("scores", {}))
    overall_score = _overall_score_from_dimensions(scores)

    # 沒有真的逐字稿（無語音/轉錄失敗）就沒有東西可以跟畫面比對，不管 LLM 怎麼回都強制視為 false，
    # 避免「沒有語音」本身被誤判成「內容不一致」，白白多觸發一次 reflection
    has_real_transcript = bool(transcript.strip()) and transcript != "（無逐字稿）"
    content_mismatch = bool(parsed.get("content_mismatch", False)) and has_real_transcript

    return {
        "scores": scores,
        "overall_score": overall_score,
        "verdict": _verdict_from_score(overall_score),
        "summary": parsed.get("summary", ""),
        "tags": parsed.get("tags", []),
        "content_mismatch": content_mismatch,
        "mismatch_reason": parsed.get("mismatch_reason", "") if content_mismatch else "",
    }
