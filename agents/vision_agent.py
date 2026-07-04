import base64
import json
from openai import OpenAI
from downloader import download_video, extract_frames
from models import AgentOutput, SkipItState
from config import GPT_MODEL, OPENAI_API_KEY, FRAME_COUNT

_client = OpenAI(api_key=OPENAI_API_KEY)

VISION_PROMPT = """你是一個影片品質分析師。以下是一支 YouTube Shorts 的 {n} 張截圖（依時間順序）。

請判斷這支影片是否為「廢片」，回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "result": "一句話描述畫面內容",
  "confidence": 0.0到1.0的信心分數（若畫面模糊或無法判斷請給0.3以下）,
  "tags": ["最多3個廢片標籤，例如：AI生成、畫質差、重複畫面、無字幕、純特效"]
}}"""


def run_vision_agent(state: SkipItState) -> dict:
    # 1. 下載影片、截 n 張幀（downloader.py 內建快取，重複分析不會重下載/重截）
    video_path = download_video(state["url"])
    frames = extract_frames(video_path, n=FRAME_COUNT)

    # 2. 轉成 base64 data URL 給 GPT-4o；detail="low" 只是粗判廢片不需要高解析度，省 token
    images = []
    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode()
        images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })

    # 3. 組成 prompt 送給 GPT-4o
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT.format(n=len(frames))},
                *images,
            ],
        }],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    # 4. 包成 AgentOutput 格式
    output: AgentOutput = {
        "agent": "vision",
        "result": parsed.get("result", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "tags": parsed.get("tags", []),
    }

    # video_path / frame_paths 一併回傳，讓 Audio Agent 可以重用同一支下載好的影片
    return {
        "vision_output": output,
        "video_path": str(video_path),
        "frame_paths": [str(f) for f in frames],
    }


if __name__ == "__main__":
    result = run_vision_agent({"url": "https://www.youtube.com/shorts/99ObPP9MoBw"})
    out = result["vision_output"]
    print("影片路徑:", result["video_path"])
    print("分析:", out["result"])
    print("信心:", out["confidence"])
    print("標籤:", out["tags"])
