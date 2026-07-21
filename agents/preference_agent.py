import json
from datetime import date
from openai import OpenAI
from database import (
    get_taste_profile,
    save_taste_profile,
    is_blacklisted,
    add_to_blacklist,
    remove_from_blacklist,
    count_consecutive_trash,
)
from config import GPT_MODEL, OPENAI_API_KEY, IMPLICIT_BLACKLIST_THRESHOLD

_client = OpenAI(api_key=OPENAI_API_KEY)

PREFERENCE_PROMPT = """你是 SkipIt Bot 的偏好學習 Agent，負責維護使用者的個人品味檔案。

{context_block}

【現有品味檔案】
{taste_profile}

---
判斷規則：
1. 修改品味檔案時要「細化」而不是「推翻」：若使用者的話跟現有規則矛盾，
   先推理是使用者改變心意，還是舊規則寫得太粗（例：檔案寫「討厭開箱」，
   但使用者稱讚一支深度評測開箱 → 應細化為「討厭純推銷開箱，接受深度評測」，
   不要直接刪掉舊規則）。
2. 品味檔案最多 15 條規則，超過就合併相似規則，且第一行必須保留
   「【使用者品味檔案】」標題。來自使用者明確回饋的規則附上日期，
   今天的日期是 {today}，必須一字不差使用這個日期，不可自行編造其他日期。
3. 若使用者提到「封鎖」「不想再看到」「別再推薦」這類字眼，creator_action 設為
   blacklist；提到「解封」「其實還好」「不要擋他」設為 whitelist；否則設為 none。
   這裡的 creator_action 只代表使用者的「意圖」，實際要封鎖哪個頻道由程式決定，
   不需要你指出頻道 ID。

回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "new_taste_profile": "重寫後的完整品味檔案",
  "creator_action": "blacklist | whitelist | none",
  "learned": "一句話總結這次學到什麼"
}}"""

ONBOARDING_QUESTIONS = """1 = 心靈雞湯/勵志語錄
2 = 開箱業配
3 = AI生成動物/寵物
4 = 標題黨/誇張縮圖
5 = 其他（使用者會直接打字說明）
0 = 都沒有"""

ONBOARDING_CONTEXT = """【情境：首次使用問卷】
使用者剛回覆了偏好問卷，選項對照如下：
{questions}

使用者原始回覆：「{message}」
請把回覆的數字對照回實際類別名稱，再轉換成品味檔案的初始規則
（例如回覆「1,3」要寫成「討厭：心靈雞湯/勵志語錄、AI生成動物/寵物」，
不要直接寫「類別1」「類別3」這種沒有意義的字）。"""

FEEDBACK_CONTEXT = """【情境：針對剛看完的判定結果回饋】
使用者剛收到一支影片的判定，標籤是 {tags}，現在說：「{message}」"""


def run_preference_agent(state: dict) -> dict:
    """
    統一入口，處理兩種情境：
    - 問卷（taste_profile == ""）
    - 針對剛看完判定的回饋（taste_profile 有內容，state 帶 creator_id）
    使用者要封鎖/解封的頻道一律是「剛剛那支影片」的 creator_id，不做模糊比對。
    回傳 {"learned": str, "reply_note": str}，reply_note 給 line_bot 決定要不要回話。
    """
    user_id = state["user_id"]
    existing_profile = state.get("taste_profile") or ""
    message = state.get("user_feedback", "")
    tags = state.get("tags") or []
    creator_id = state.get("creator_id")
    creator_name = state.get("creator_name")

    is_onboarding = existing_profile == ""
    context_block = (
        ONBOARDING_CONTEXT.format(questions=ONBOARDING_QUESTIONS, message=message) if is_onboarding
        else FEEDBACK_CONTEXT.format(tags=tags, message=message)
    )

    prompt = PREFERENCE_PROMPT.format(
        context_block=context_block,
        taste_profile=existing_profile or "（尚無品味檔案）",
        today=date.today().isoformat(),
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    new_profile = parsed.get("new_taste_profile", "")
    if new_profile.strip():
        save_taste_profile(user_id, new_profile)

    action = parsed.get("creator_action", "none")
    reply_note = ""
    if action in ("blacklist", "whitelist") and creator_id:
        if action == "blacklist":
            add_to_blacklist(user_id, creator_id, reason="使用者明確封鎖")
            reply_note = f"已封鎖『{creator_name or creator_id}』"
        else:
            remove_from_blacklist(user_id, creator_id)
            reply_note = f"已解除封鎖『{creator_name or creator_id}』"

    return {
        "learned": parsed.get("learned", ""),
        "reply_note": reply_note,
    }


def check_blacklist(user_id: str, creator_id: str) -> dict:
    """graph.py 的 orchestrator 節點用：早停判斷，純 SQL 查詢，不碰 LLM。"""
    return {"should_early_stop": is_blacklisted(user_id, creator_id)}


def check_implicit_blacklist(user_id: str, creator_id: str) -> None:
    """每次 Scoring Agent 跑完後呼叫（不需要使用者回饋觸發）。
    同一頻道連續 N 次都被判 trash，自動封鎖，並標註信心較低的原因。"""
    if count_consecutive_trash(user_id, creator_id) >= IMPLICIT_BLACKLIST_THRESHOLD:
        add_to_blacklist(user_id, creator_id, reason="連續3次trash自動偵測")


if __name__ == "__main__":
    from database import init_db
    init_db()

    print("=== 情境 1：問卷 ===")
    result = run_preference_agent({
        "user_id": "demo_user",
        "taste_profile": "",
        "user_feedback": "1,3",
        "tags": [],
    })
    print(result)
    print("\n--- 品味檔案 ---")
    print(get_taste_profile("demo_user"))

    print("\n=== 情境 2：針對判定的回饋（含封鎖）===")
    result = run_preference_agent({
        "user_id": "demo_user",
        "taste_profile": get_taste_profile("demo_user"),
        "user_feedback": "這頻道整天發這種內容，封鎖掉",
        "tags": ["雞湯", "情緒操弄"],
        "creator_id": "UCtest123",
        "creator_name": "測試頻道",
    })
    print(result)

    from database import is_blacklisted
    print("是否已封鎖:", is_blacklisted("demo_user", "UCtest123"))
