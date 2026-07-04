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
from config import LINE_CHANNEL_SECRET, LINE_CHANNEL_ACCESS_TOKEN

app = FastAPI()
handler = WebhookHandler(LINE_CHANNEL_SECRET)
_config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)

YT_SHORTS_PATTERN = re.compile(
    r"https?://(?:www\.)?youtube\.com/shorts/[\w-]+"
    r"|https?://youtu\.be/[\w-]+"
)


def extract_youtube_url(text: str) -> str | None:
    # 群組訊息裡可能夾雜其他文字，只抓出符合 Shorts 網址格式的部分
    m = YT_SHORTS_PATTERN.search(text)
    return m.group(0) if m else None


def format_reply(score: int, summary: str, tags: list[str],
                  creator_name: str, early_stop: bool) -> str:
    # 分數決定 emoji 跟廢片/普通/好片標籤，早停（黑名單）時額外附註
    score_emoji = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
    label = "廢片" if score <= 4 else "普通" if score <= 6 else "好片"
    tag_str = " ".join(f"#{t}" for t in tags) if tags else "無"
    early_note = "\n⛔ 頻道已列入黑名單，略過完整分析" if early_stop else ""
    return (
        f"{score_emoji} {label}（{score}/10）\n"
        f"📝 {summary}\n"
        f"🏷️ {tag_str}\n"
        f"👤 {creator_name or '未知頻道'}"
        f"{early_note}"
    )


def _reply(reply_token: str, text: str) -> None:
    # reply_token 只能用一次，且必須在收到 webhook 後短時間內回覆
    with ApiClient(_config) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text)],
            )
        )


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


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    url = extract_youtube_url(event.message.text)
    if not url:
        # 訊息裡沒有 YouTube Shorts 連結就忽略，不回覆
        return
    # 先立即回覆「分析中」，因為 reply_token 有時效性，完整分析要幾十秒跑不完
    _reply(event.reply_token, "🔍 分析中，請稍後...")

    def _analyze():
        try:
            result = run_pipeline(url)
            text = format_reply(
                score=result["score"],
                summary=result["summary"],
                tags=result["tags"] or [],
                creator_name=result.get("creator_name") or "",
                early_stop=result.get("should_early_stop", False),
            )
        except Exception as e:
            text = f"❌ 分析失敗：{str(e)[:100]}"
        # reply_token 已經用掉了，這裡改用 push_message 主動推播結果
        with ApiClient(_config) as api_client:
            MessagingApi(api_client).push_message(
                to=event.source.group_id or event.source.user_id,
                messages=[TextMessage(text=text)],
            )

    # 丟到背景執行緒跑，才不會擋住 webhook 的回應
    threading.Thread(target=_analyze, daemon=True).start()
