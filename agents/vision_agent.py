import base64
import json
from openai import OpenAI
from downloader import download_video, extract_frames
from models import AgentState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

VISION_PROMPT = """你是一個影片品質分析師。以下是一支 YouTube Shorts 依時間順序、每秒一張的截圖。

請找出畫面中出現的所有問題訊號，回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "signals": ["每個訊號一句話描述，例如：手指變形疑似AI生成、縮圖與內容不符、濾鏡過度、畫面右下角有搬運浮水印、後半段畫面重複"]
}}
偵測重點：AI 生成跡象（肢體變形、材質過度平滑）、誇張縮圖（縮圖標題黨）、過度濾鏡、品牌 logo/浮水印、畫面重複度。
若沒有發現任何問題，signals 回空陣列。"""


def run_vision_agent(state: AgentState) -> dict:
    # 1. 下載影片、每秒抽一幀（downloader.py 內建快取，重複分析不會重下載/重截）
    video_path = download_video(state["url"])
    frames = extract_frames(video_path)

    # 2. 轉成 base64 data URL 給 GPT-4o
    images = []
    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode()
        images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    # 3. 組成 prompt 送給 GPT-4o
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISION_PROMPT},
                *images,
            ],
        }],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    return {
        "vision_signals": parsed.get("signals", []),
        "video_path": str(video_path),
        "frames": [str(f) for f in frames],
    }


if __name__ == "__main__":
    result = run_vision_agent({"url": "https://www.youtube.com/shorts/99ObPP9MoBw"})
    print("影片路徑:", result["video_path"])
    print("幀數:", len(result["frames"]))
    print("vision_signals:", result["vision_signals"])
