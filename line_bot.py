import re
import threading
from fastapi import FastAPI, Request, Header, HTTPException
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, PushMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from graph import run_pipeline, run_pipeline_multi
from agents.preference_agent import run_preference_agent
from database import get_taste_profile
from downloader import load_signal_cache
from config import LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN, MAX_CONCURRENT_ANALYSES

app = FastAPI()
handler = WebhookHandler(LINE_CHANNEL_SECRET)
_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
# 限制同時跑幾支影片的完整分析（下載+GPT+Whisper），避免多人同時貼連結時打爆 API/資源
_analysis_semaphore = threading.Semaphore(MAX_CONCURRENT_ANALYSES)

# Demo 用：群組訊息要秀「同一支影片、各自品味各自評分」的固定帳號清單。
# LINE API 沒有「列出群組所有成員」的權限，沒辦法真的對群組裡每個人都自動評分，
# 所以先寫死這幾個 demo 帳號的 user_id（跟 config.py 一樣不做成通用功能）。
DEMO_USER_IDS: list[str] = [
    "Ufc25c7d999673cb1955948b6386af391",
    "Uf0df89c44fb27b44e3c4966cd736baae",
]

# process 內的暫存狀態（重啟就清空，demo 規模夠用，之後有需要再搬進 DB）：
# 等問卷回覆的使用者 -> 他原本要分析的連結（觸發問卷的那則訊息不一定帶連結，可能是 None）
_pending_onboarding: dict[str, str | None] = {}
# 等 👍/👎 回饋的「聊天室」(target_id) -> {user_id: 該使用者這次的判定結果}。
# 一次分析可能對多個 user_id 各自算出不同結果（見 DEMO_USER_IDS），
# 所以用 user_id 當第二層 key，回饋時只套用「回饋者自己」那一份，不會錯用別人的 tags/verdict。
# 這是「沒有引用回覆」時的 fallback：把回饋當作針對聊天室最新一次判定。
_pending_feedback: dict[str, dict[str, dict]] = {}
# 判定訊息的 message_id -> {user_id: 該使用者這次的判定結果}。使用者「引用回覆」某則判定訊息時，
# 用 quoted_message_id 精準對應到那一支影片，不受後續又分析了幾支影響。
# process 內暫存（重啟清空，demo 規模夠用）。
_verdict_by_message: dict[str, dict[str, dict]] = {}

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


def format_verdict(overall_score: int, verdict: str, summary: str, tags: list[str] | None = None,
                    name: str | None = None) -> str:
    # 只給 verdict + 總分 + 一句話理由 + tags：五維細項數字沒有各自的理由支撐，
    # 秀出來反而像隨機分數，細節留給終端機 log；tags 是自我解釋的關鍵字，不需要額外理由佐證，適合秀出來
    verdict_emoji = {"trash": "❌", "review": "⚠️", "keep": "✅"}.get(verdict, "⚠️")
    verdict_label = {"trash": "廢片", "review": "普通", "keep": "好片"}.get(verdict, "普通")
    prefix = f"👤 {name}：" if name else ""
    header = f"{prefix}{verdict_emoji} {verdict_label}（{overall_score}分）— {summary}"
    tag_line = f"\n🏷️ {'、'.join(tags)}" if tags else ""
    return f"{header}{tag_line}"


def _display_name(user_id: str, group_id: str | None = None) -> str:
    # 群組場景優先用 get_group_member_profile，因為對方不一定跟 bot 加過一對一好友；
    # 拿不到名稱（例如已離開群組）就退回顯示 user_id 前 8 碼，至少還能分辨是誰。
    try:
        with ApiClient(_config) as api_client:
            api = MessagingApi(api_client)
            if group_id:
                return api.get_group_member_profile(group_id, user_id).display_name
            return api.get_profile(user_id).display_name
    except Exception:
        return user_id[:8]


def _reply(reply_token: str, text: str) -> None:
    # reply_token 只能用一次，且必須在收到 webhook 後短時間內回覆
    with ApiClient(_config) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(reply_token=reply_token, messages=[TextMessage(text=text)])
        )


def _push(target_id: str, text: str) -> str | None:
    # reply_token 已經用掉（或這則訊息不是回覆某個 webhook 事件），改用 push_message 主動推播。
    # 回傳這則訊息的 message_id，讓呼叫端可以建立 message_id -> 判定結果 的對照（供引用回覆查詢）。
    with ApiClient(_config) as api_client:
        resp = MessagingApi(api_client).push_message(
            PushMessageRequest(to=target_id, messages=[TextMessage(text=text)])
        )
    sent = getattr(resp, "sent_messages", None)
    return sent[0].id if sent else None


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
    # target_id 只有群組訊息才會是 group_id，私聊時 target_id == user_id（見 handle_message）
    is_group = target_id != user_id
    try:
        with _analysis_semaphore:
            if is_group:
                results = run_pipeline_multi(DEMO_USER_IDS, url)
            else:
                results = {user_id: run_pipeline(user_id, url)}
    except Exception as e:
        _push(target_id, f"❌ 分析失敗：{str(e)[:100]}")
        return

    # 記住這次判定，等聊天室裡任何人回覆 👍/👎 時，各自對應各自那一份 tags/頻道餵給 preference_agent；
    # 存 url 是為了回饋當下能回頭查 vision/audio 的 signal cache，讓 preference_agent 看得到原始畫面/語音內容
    pending = {
        uid: {
            "url": url,
            "tags": r.get("tags") or [],
            "creator_id": r.get("creator_id"),
            "creator_name": r.get("creator_name"),
        }
        for uid, r in results.items()
    }
    _pending_feedback[target_id] = pending  # fallback：沒引用時當作針對最新一次判定

    # 每個人各自發一則獨立判定訊息（不合併成一則），
    # message_id 只對應「這一個人」的 pending，之後引用回覆會精準對到本人，不會混到別人的資料
    for uid, r in results.items():
        name = _display_name(uid, target_id) if is_group else None
        text = format_verdict(
            overall_score=r.get("overall_score", 0),
            verdict=r.get("verdict", "review"),
            summary=r.get("summary", ""),
            tags=r.get("tags"),
            name=name,
        )
        message_id = _push(target_id, text)
        if message_id:
            _verdict_by_message[message_id] = {uid: pending[uid]}


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    # 一對一聊天時 event.source 是 UserSource，沒有 group_id 屬性，用 getattr 避免 AttributeError
    target_id = getattr(event.source, "group_id", None) or user_id
    text = event.message.text.strip()
    url = extract_youtube_url(text)

    if user_id in _pending_onboarding:
        if url:
            # 還在等問卷回覆時又貼了連結，先更新成最新這支要分析的，不重複發問卷
            _pending_onboarding[user_id] = url
            _reply(event.reply_token, "還在等你回答上面的問卷喔，回覆完我就開始分析～")
            return
        # 這則訊息是問卷回覆：先建立初始 taste_profile；如果當初是貼連結才觸發問卷，
        # 回覆完接著分析那支影片，否則（單純打招呼觸發的）就只建檔案，不用接著分析
        pending_url = _pending_onboarding.pop(user_id)
        reply_text = "已記錄你的偏好！" + ("開始分析剛剛那支影片... ⏳" if pending_url else "之後貼 YouTube Shorts 連結我就會幫你判定囉～")
        _reply(event.reply_token, reply_text)

        def _onboard_then_analyze():
            run_preference_agent({
                "user_id": user_id, "taste_profile": "",
                "user_feedback": text, "tags": [],
            })
            if pending_url:
                _run_analysis(user_id, target_id, pending_url)

        threading.Thread(target=_onboard_then_analyze, daemon=True).start()
        return

    if get_taste_profile(user_id) is None:
        # 第一次跟 bot 互動就發問卷，不限定要貼連結——只要是這個 user_id 第一次傳訊息就觸發；
        # 如果這則訊息剛好帶連結，記住等問卷回覆完再分析，沒有就是 None（只是打招呼）
        _pending_onboarding[user_id] = url
        _reply(event.reply_token, ONBOARDING_PROMPT)
        return

    if url:
        # 先立即回覆「分析中」，因為 reply_token 有時效性，完整分析要幾十秒跑不完
        _reply(event.reply_token, "🔍 分析中，請稍後...")
        threading.Thread(target=_run_analysis, args=(user_id, target_id, url), daemon=True).start()
        return

    # 判定回饋分支：優先看有沒有「引用回覆」某則特定判定訊息，其次才用聊天室最新一次判定
    quoted_id = getattr(event.message, "quoted_message_id", None)
    if quoted_id and quoted_id in _verdict_by_message:
        # 使用者引用回覆了某則判定訊息 → 精準對應那一支影片，不受後續又分析了幾支影響
        pending_map = _verdict_by_message[quoted_id]
    elif target_id in _pending_feedback:
        # 沒引用 → 當作針對聊天室最近一次判定
        pending_map = _pending_feedback[target_id]
    else:
        pending_map = None

    # 只有「這次分析有算過他這份」的人，回饋才會生效（例如群組場景只對 DEMO_USER_IDS 算過），
    # 避免拿到別人的 tags/頻道去更新自己的 taste_profile
    pending = pending_map.get(user_id) if pending_map else None

    if pending is not None:
        # 回饋人 = 這則訊息的 user_id（不一定是貼連結的人），各自更新自己的 taste_profile；
        # 使用者若提到「封鎖這頻道」，一律指這次判定的頻道，不做模糊比對
        existing_profile = get_taste_profile(user_id) or ""

        def _handle_feedback(pending=pending):
            # 用這次判定的 url 回頭查 signal cache，讓 preference_agent 看得到真正的畫面/語音內容，
            # 不是只靠濃縮過的 tags 猜使用者在講什麼；快取可能已被清掉，查不到就給空字串
            video_url = pending.get("url")
            vision_cache = load_signal_cache(video_url, "vision") if video_url else None
            audio_cache = load_signal_cache(video_url, "audio") if video_url else None

            result = run_preference_agent({
                "user_id": user_id, "taste_profile": existing_profile,
                "user_feedback": text, "tags": pending["tags"],
                "creator_id": pending.get("creator_id"),
                "creator_name": pending.get("creator_name"),
                "vision_description": (vision_cache or {}).get("vision_description", ""),
                "transcript": (audio_cache or {}).get("transcript", ""),
            })
            if result.get("reply_note"):
                _push(target_id, result["reply_note"])
            # 回饋處理完，秀出「這個人現在的品味檔案」——證明剛剛的回饋真的改了什麼
            profile = get_taste_profile(user_id) or "（尚無品味檔案）"
            name = _display_name(user_id, target_id) if target_id != user_id else None
            header = f"📝 {name} 的品味檔案：" if name else "📝 你的品味檔案："
            _push(target_id, f"{header}\n{profile}")

        threading.Thread(target=_handle_feedback, daemon=True).start()
        return

    # 其他訊息（沒連結、沒在等問卷/回饋）就忽略，不回覆
