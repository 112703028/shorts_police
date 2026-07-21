import json
from datetime import datetime
import yt_dlp
from openai import OpenAI
from models import AgentState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

METADATA_PROMPT = """你是一個專門識別「廢片」與「營銷號」的分析師。

【這支影片的資訊】
標題: {title}
描述: {description}
頻道名稱: {uploader}
觀看數: {view_count}
按讚數: {like_count}
按讚率: {like_ratio}%（正常頻道約 2-5%，低於 0.5% 高度可疑）
時長: {duration} 秒
標籤: {tags}
描述含外部連結: {has_link}

【這個頻道最近發的影片標題（判斷主題一致性）】
{recent_titles}

【發片頻率】
{post_frequency}

---
請根據以下「營銷號」特徵找出所有問題訊號：
1. 盜用內容 — 標題暗示搬運他人影片
2. 捏造/誇大 — 標題使用「震驚」「不敢相信」「內幕」「曝光」「陰謀論」等煽情用語
3. 內容農場 — 頻道同時發政治、娛樂、生活等完全不相關的主題
4. 批量發片 — 一天多支，主題雷同或隨機
5. 買流量 — 按讚率低於 0.5%，觀看數異常高
6. 導流廣告 — 描述含外部連結 + 推銷話術

回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "signals": ["每個訊號一句話描述，例如：按讚率0.3%疑似買流量、頻道主題橫跨政治娛樂疑似內容農場、標題使用震驚等煽情用語"]
}}
若沒有發現任何問題，signals 回空陣列。"""


def _get_recent_channel_info(channel_id: str, n: int = 8) -> tuple[list[str], str]:
    """抓頻道最近 n 支影片的標題和上傳日期，回傳 (標題列表, 發片頻率描述)"""
    channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
    opts = {
        "skip_download": True,
        "quiet": True,
        "playlistend": n,
        "extract_flat": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
        entries = info.get("entries") or []
        titles = [e.get("title", "") for e in entries]

        dates = [e.get("upload_date", "") for e in entries if e.get("upload_date")]
        if len(dates) >= 2:
            parsed_dates = [datetime.strptime(d, "%Y%m%d") for d in dates if len(d) == 8]
            if len(parsed_dates) >= 2:
                days_span = (parsed_dates[0] - parsed_dates[-1]).days or 1
                freq = round(len(parsed_dates) / days_span, 1)
                frequency_str = f"最近 {len(parsed_dates)} 支影片分佈在 {days_span} 天內（平均每天 {freq} 支）"
            else:
                frequency_str = "無法計算"
        else:
            frequency_str = "資料不足"

        return titles, frequency_str
    except Exception:
        return [], "無法取得"


def run_metadata_agent(state: AgentState) -> dict:
    # 1. 抓這支影片的 metadata（不下載影片本體）
    opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(state["url"], download=False)

    channel_id = info.get("channel_id", "")
    view_count = info.get("view_count") or 1
    like_count = info.get("like_count") or 0
    like_ratio = round(like_count / view_count * 100, 2)
    description = info.get("description") or ""
    has_link = "是" if "http" in description else "否"

    # 2. 抓頻道最近影片 + 發片頻率，判斷主題一致性/批量發片
    recent_titles, post_frequency = _get_recent_channel_info(channel_id)
    recent_titles_str = "\n".join(f"- {t}" for t in recent_titles) if recent_titles else "（無法取得）"

    # 3. 送給 GPT-4o 判斷
    prompt = METADATA_PROMPT.format(
        title=info.get("title", ""),
        description=description[:500],
        uploader=info.get("uploader", ""),
        view_count=view_count,
        like_count=like_count,
        like_ratio=like_ratio,
        duration=info.get("duration", 0),
        tags=", ".join(info.get("tags", [])),
        has_link=has_link,
        recent_titles=recent_titles_str,
        post_frequency=post_frequency,
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    return {
        "metadata_signals": parsed.get("signals", []),
        "creator_id": channel_id,
        "creator_name": info.get("uploader", ""),
    }


if __name__ == "__main__":
    result = run_metadata_agent({"url": "https://www.youtube.com/shorts/okFg7q3wjVA"})
    print("頻道:", result["creator_name"], f"({result['creator_id']})")
    print("metadata_signals:", result["metadata_signals"])
