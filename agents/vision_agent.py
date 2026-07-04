import base64
import json
from openai import OpenAI
from downloader import download_video, extract_frames
from models import AgentOutput, SkipItState
from config import GPT_MODEL, OPENAI_API_KEY, FRAME_COUNT

_client = OpenAI(api_key=OPENAI_API_KEY)

CONTENT_PROMPT = """你是一個影片品質分析師。以下是一支 YouTube Shorts 的 {n} 張截圖（依時間順序）。

請判斷畫面內容本身的品質，回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "content_result": "一句話描述畫面內容",
  "confidence": 0.0到1.0的信心分數（若畫面模糊或無法判斷請給0.3以下）,
  "content_tags": ["最多2個標籤，例如：畫質差、重複畫面、無字幕、純特效"]
}}"""

AUTHENTICITY_PROMPT = """你是一個影片真實性分析師。以下是一支 YouTube Shorts 的 {n} 張截圖（依時間順序）。

請判斷這支影片是否為「盜用搬運」或「AI 生成」，注意以下線索：
1. 浮水印/搬運痕跡 — 畫面角落是否殘留其他平台的浮水印或帳號 ID（例如抖音、小紅書、TikTok 的 logo 或 @帳號名）
2. AI 生成痕跡 — 手指/肢體變形、物理不合理（例如物體穿透、光影不連貫）、材質過度平滑像塑膠、背景細節扭曲

回覆 JSON（不要加 markdown code block）:
{{
  "authenticity_result": "一句話說明是否有搬運或 AI 生成痕跡",
  "has_watermark": true或false,
  "is_ai_generated": true或false,
  "authenticity_tags": ["最多2個標籤，例如：搬運浮水印、AI生成"]
}}"""


def _analyze_content(images: list[dict], n: int) -> dict:
    """GPT-4o 分析畫面內容本身的品質"""
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": CONTENT_PROMPT.format(n=n)}, *images],
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _analyze_authenticity(images: list[dict], n: int) -> dict:
    """GPT-4o 判斷是否為搬運（浮水印）或 AI 生成內容"""
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": AUTHENTICITY_PROMPT.format(n=n)}, *images],
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


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

    # 3. 同時分析：畫面內容品質 + 搬運/AI生成真實性
    content = _analyze_content(images, len(frames))
    authenticity = _analyze_authenticity(images, len(frames))

    # 4. 合併兩個分析的標籤
    all_tags = list(set(
        content.get("content_tags", []) +
        authenticity.get("authenticity_tags", [])
    ))

    result_text = f"{content.get('content_result', '')}；{authenticity.get('authenticity_result', '')}".strip("；")

    output: AgentOutput = {
        "agent": "vision",
        "result": result_text,
        "confidence": float(content.get("confidence", 0.5)),
        "tags": all_tags,
        "has_watermark": authenticity.get("has_watermark", False),
        "is_ai_generated": authenticity.get("is_ai_generated", False),
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
    print("是否有浮水印:", out.get("has_watermark", False))
    print("是否 AI 生成:", out.get("is_ai_generated", False))
