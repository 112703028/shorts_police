import json
from openai import OpenAI
from database import (
    get_taste_profile,
    save_taste_profile,
    get_last_analysis,
    set_creator_status,
)
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

FEEDBACK_PROMPT = """你是 SkipIt Bot 的偏好學習 Agent。使用者剛在 LINE 群組傳了一則訊息，
你要判斷這是不是對「影片分析結果」的回饋，如果是，就更新使用者的品味檔案。

【使用者的訊息】
「{message}」

【最近一次分析結果（使用者說「這個」通常指它）】
{last_analysis}

【現有的品味檔案】
{taste_profile}

---
判斷規則：
1. 群組訊息大多是閒聊。以下兩種才算回饋（is_feedback: true）：
   a. 對最近分析結果的評論（「給太低了吧」「這就是廢片沒錯」「封鎖這個頻道」）
   b. 對影片類型的偏好陳述（「搞笑片我都愛看」「開箱的不要再推了」「這種我沒興趣」）
   與影片完全無關的訊息（吃飯、約時間、日常聊天）回 is_feedback: false。
2. 修改品味檔案時要「細化」而不是「推翻」：
   若回饋與現有檔案矛盾，先推理使用者是改變心意還是舊規則寫得太粗。
   例：檔案寫「討厭開箱」，但使用者稱讚一支深度評測開箱
   → 細化為「討厭純推銷開箱，接受深度評測」，不要直接刪掉舊規則。
3. 品味檔案最多 15 條規則，超過就合併相似規則，且第一行必須保留「【使用者品味檔案】」標題。
   來自使用者明確回饋的規則，結尾附上日期。今天的日期是 {today}，
   標註日期時必須一字不差地使用 {today}，禁止自行編造其他日期。
4. 只有使用者明確表示要封鎖/解封某頻道時，creator_action 才有值。

回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "is_feedback": true或false,
  "feedback_type": "disagree_low | disagree_high | block_creator | like_creator | general_preference | none",
  "creator_action": "blacklist | whitelist | none",
  "new_taste_profile": "重寫後的完整品味檔案（is_feedback 為 false 時回空字串）",
  "learned": "一句話總結這次學到什麼（is_feedback 為 false 時回空字串）",
  "reply": "回覆給使用者的訊息，親切簡短（is_feedback 為 false 時回空字串）"
}}"""


def run_feedback_agent(message: str) -> dict:
    """
    處理使用者的自然語言回饋：
    1. 判斷是不是回饋（不是就保持沉默）
    2. 理解回饋內容，重寫品味檔案（細化而非推翻）
    3. 需要時直接更新頻道黑/白名單
    """
    from datetime import date

    last = get_last_analysis()
    last_str = (
        f"頻道: {last['creator_id']}\n分數: {last['score']}/10\n"
        f"摘要: {last['summary']}\n標籤: {', '.join(last['tags'])}"
        if last else "（尚無分析紀錄）"
    )

    prompt = FEEDBACK_PROMPT.format(
        message=message,
        last_analysis=last_str,
        taste_profile=get_taste_profile(),
        today=date.today().isoformat(),
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    if not parsed.get("is_feedback"):
        return {"is_feedback": False, "reply": "", "learned": ""}

    # 更新品味檔案（Agent 重寫後的版本）
    new_profile = parsed.get("new_taste_profile", "")
    if new_profile.strip():
        save_taste_profile(new_profile)

    # 硬性操作：黑/白名單直接寫進 creator_preferences
    action = parsed.get("creator_action", "none")
    if action in ("blacklist", "whitelist") and last and last.get("creator_id"):
        set_creator_status(last["creator_id"], action)

    return {
        "is_feedback": True,
        "feedback_type": parsed.get("feedback_type", "general_preference"),
        "creator_action": action,
        "learned": parsed.get("learned", ""),
        "reply": parsed.get("reply", "收到你的回饋了！"),
    }


if __name__ == "__main__":
    import sys
    from database import init_db
    init_db()

    msg = sys.argv[1] if len(sys.argv) > 1 else "這個其實還不錯啊，你給太低分了吧"
    result = run_feedback_agent(msg)

    print("是否回饋:", result["is_feedback"])
    if result["is_feedback"]:
        print("回饋類型:", result["feedback_type"])
        print("頻道操作:", result["creator_action"])
        print("學到:", result["learned"])
        print("回覆:", result["reply"])
        print("\n--- 更新後的品味檔案 ---")
        print(get_taste_profile())
