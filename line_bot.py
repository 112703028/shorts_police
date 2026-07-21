import re
import threading
from fastapi import FastAPI, Request, Header, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from graph import run_pipeline
from agents.preference_agent import run_preference_agent
from database import get_taste_profile
from config import LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN, MAX_CONCURRENT_ANALYSES

app = FastAPI()
handler = WebhookHandler(LINE_CHANNEL_SECRET)
_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
# 限制同時跑幾支影片的完整分析（下載+GPT+Whisper），避免多人同時貼連結時打爆 API/資源
_analysis_semaphore = threading.Semaphore(MAX_CONCURRENT_ANALYSES)

# process 內的暫存狀態（重啟就清空，demo 規模夠用，之後有需要再搬進 DB）：
# 等問卷回覆的使用者 -> 他原本要分析的連結
_pending_onboarding: dict[str, str] = {}
# 等 👍/👎 回饋的「聊天室」(target_id) -> 最近一次判定結果（tags/creator_id/creator_name）
# 用 target_id 而非 user_id 當 key，讓群組裡任何人都能對同一次判定回饋，不侷限貼連結的那個人；
# 每個人回饋時各自用自己的 user_id 查/寫 taste_profile，維持 per-user 個人化。
_pending_feedback: dict[str, dict] = {}

YT_SHORTS_PATTERN = re.compile(
    r"https?://(?:www\.)?youtube\.com/shorts/[\w-]+"
    r"|https?://youtu\.be/[\w-]+"
)

ONBOARDING_PROMPT = """👋 我是 SkipIt Bot，幫你判斷 YouTube Shorts 值不值得看。
先了解一下你的偏好（可複選，用逗號分開回覆數字，例如 1,3）：

1️⃣ 心靈雞湯/勵志語錄
2️⃣ 開箱業配
3️⃣ AI生成動物/寵物
4️⃣ 標題黨/誇張縮圖
5️⃣ 其他（請直接打字說明，不用選數字）
0️⃣ 都沒有，讓我用久了再學

有沒有想先封鎖的頻道？直接貼頻道名稱，沒有就回「無」。"""


def extract_youtube_url(text: str) -> str | None:
    # 群組訊息裡可能夾雜其他文字，只抓出符合 Shorts 網址格式的部分
    m = YT_SHORTS_PATTERN.search(text)
    return m.group(0) if m else None


def format_reply(overall_score: int, verdict: str, summary: str) -> str:
    verdict_emoji = {"trash": "❌", "review": "⚠️", "keep": "✅"}.get(verdict, "⚠️")
    verdict_label = {"trash": "廢片", "review": "普通", "keep": "好片"}.get(verdict, "普通")
    return (
        f"{verdict_emoji} {verdict_label}（{overall_score}分）— {summary}\n\n"
        f"這個判定準確嗎？回覆 👍 或 👎"
    )


def _reply(reply_token: str, text: str) -> None:
    # reply_token 只能用一次，且必須在收到 webhook 後短時間內回覆
    with ApiClient(_config) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=text)])
        )


def _push(target_id: str, text: str) -> None:
    # reply_token 已經用掉（或這則訊息不是回覆某個 webhook 事件），改用 push_message 主動推播
    with ApiClient(_config) as api_client:
        MessagingApi(api_client).push_message(to=target_id, messages=[TextMessage(text=text)])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_line_signature: str = Header(...),
):
    # LINE 官方要求驗證簽章，確保請求真的來自 LINE 平台
    body = await request.body()
    try:
        handler.handle(body.decode(), x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"


def _run_analysis(user_id: str, target_id: str, url: str) -> None:
    try:
        with _analysis_semaphore:
            result = run_pipeline(user_id, url)
    except Exception as e:
        _push(target_id, f"❌ 分析失敗：{str(e)[:100]}")
        return
    # 記住這次判定，等聊天室裡任何人回覆 👍/👎 時都知道要把哪些 tags/頻道餵給 preference_agent
    _pending_feedback[target_id] = {
        "tags": result.get("tags") or [],
        "creator_id": result.get("creator_id"),
        "creator_name": result.get("creator_name"),
    }
    text = format_reply(
        overall_score=result.get("overall_score", 0),
        verdict=result.get("verdict", "review"),
        summary=result.get("summary", ""),
    )
    _push(target_id, text)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    # 一對一聊天時 event.source 是 UserSource，沒有 group_id 屬性，用 getattr 避免 AttributeError
    target_id = getattr(event.source, "group_id", None) or user_id
    text = event.message.text.strip()
    url = extract_youtube_url(text)

    if url:
        if user_id in _pending_onboarding:
            # 還在等問卷回覆時又貼了連結，先更新成最新這支要分析的，不重複發問卷
            _pending_onboarding[user_id] = url
            _reply(event.reply_token, "還在等你回答上面的問卷喔，回覆完我就開始分析～")
            return
        if get_taste_profile(user_id) is None:
            # 第一次互動：先問卷，分析留到問卷回覆後才開始
            _pending_onboarding[user_id] = url
            _reply(event.reply_token, ONBOARDING_PROMPT)
            return
        # 先立即回覆「分析中」，因為 reply_token 有時效性，完整分析要幾十秒跑不完
        _reply(event.reply_token, "🔍 分析中，請稍後...")
        threading.Thread(target=_run_analysis, args=(user_id, target_id, url), daemon=True).start()
        return

    if user_id in _pending_onboarding:
        # 這則訊息是問卷回覆：先建立初始 taste_profile，再分析原本等待的那支影片
        pending_url = _pending_onboarding.pop(user_id)
        _reply(event.reply_token, "已記錄你的偏好，開始分析剛剛那支影片... ⏳")

        def _onboard_then_analyze():
            run_preference_agent({
                "user_id": user_id, "taste_profile": "",
                "user_feedback": text, "tags": [],
            })
            _run_analysis(user_id, target_id, pending_url)

        threading.Thread(target=_onboard_then_analyze, daemon=True).start()
        return

    if target_id in _pending_feedback:
        # 這則訊息是判定後的 👍/👎（或文字說明）。不 pop：讓聊天室裡每個人都能對同一次判定
        # 各自回饋，各自更新自己的 taste_profile（回饋人 = 這則訊息的 user_id，不一定是貼連結的人）；
        # 使用者若提到「封鎖這頻道」，一律指這次判定的頻道，不做模糊比對
        pending = _pending_feedback[target_id]
        existing_profile = get_taste_profile(user_id) or ""

        def _handle_feedback():
            result = run_preference_agent({
                "user_id": user_id, "taste_profile": existing_profile,
                "user_feedback": text, "tags": pending["tags"],
                "creator_id": pending.get("creator_id"),
                "creator_name": pending.get("creator_name"),
            })
            if result.get("reply_note"):
                _push(target_id, result["reply_note"])

        threading.Thread(target=_handle_feedback, daemon=True).start()
        return

    # 其他訊息（沒連結、沒在等問卷/回饋）就忽略，不回覆
