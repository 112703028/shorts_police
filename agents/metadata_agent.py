import json
import yt_dlp
from openai import OpenAI
from models import AgentOutput, SkipItState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

METADATA_PROMPT = """你是一個影片品質分析師。根據以下 YouTube Shorts 的 metadata，判斷這支影片是否為「廢片」。

Metadata:
標題: {title}
描述: {description}
頻道: {uploader}
觀看數: {view_count}
按讚數: {like_count}
時長: {duration} 秒
標籤: {tags}

請回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "result": "一句話描述這支影片的內容類型",
  "confidence": 0.0到1.0的信心分數,
  "tags": ["最多3個廢片標籤，例如：廣告、AI生成、無資訊價值、重複內容、純娛樂"]
}}
若無廢片特徵，tags 回空陣列。"""


def run_metadata_agent(state: SkipItState) -> dict:
    # 1. 用 yt-dlp 抓 metadata（不下載影片）
    opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(state["url"], download=False)

    # 2. 組成 prompt 送給 GPT-4o
    prompt = METADATA_PROMPT.format(
        title=info.get("title", ""),
        description=(info.get("description") or "")[:500],
        uploader=info.get("uploader", ""),
        view_count=info.get("view_count", 0),
        like_count=info.get("like_count", 0),
        duration=info.get("duration", 0),
        tags=", ".join(info.get("tags", [])),
    )

    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    # 3. 包成 AgentOutput 格式
    output: AgentOutput = {
        "agent": "metadata",
        "result": parsed.get("result", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "tags": parsed.get("tags", []),
    }

    return {
        "metadata_output": output,
        "creator_id": info.get("channel_id", ""),
        "creator_name": info.get("uploader", ""),
    }

if __name__ == "__main__":
    result = run_metadata_agent({"url": "https://www.youtube.com/shorts/99ObPP9MoBw"})
    out = result["metadata_output"]
    print("頻道:", result["creator_name"])
    print("分析:", out["result"])
    print("信心:", out["confidence"])
    print("標籤:", out["tags"])