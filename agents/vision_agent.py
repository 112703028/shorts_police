import base64
import json
import urllib.request
from openai import OpenAI
from downloader import download_video, extract_frames, get_thumbnail_url
from models import AgentState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

VISION_PROMPT = """你是一個影片品質分析師。第一張圖是這支 YouTube Shorts 的官方縮圖（使用者滑動時會看到的畫面），
之後依時間順序、每秒一張的是影片實際截圖。

偵測重點：
1. 縮圖與內容不符 — 明確比對第一張縮圖跟後面影片截圖是否呈現同一件事，只有真的看出落差才回報
2. AI 生成跡象（肢體變形、材質過度平滑）
3. 過度濾鏡
4. 品牌 logo/浮水印
5. 畫面重複度

回覆 JSON 格式（不要加 markdown code block）:
{{
  "signals": ["<你真的在畫面中觀察到的問題，一句話具體描述，例如指出是哪一張截圖看到什麼>", "..."]
}}
重要：只回報你真的在提供的圖片裡看到的問題，不要憑空推測或套用常見說法；每一項都必須能指出具體是哪張圖、看到什麼異狀。
沒有把握或沒有發現任何問題，signals 回空陣列，不要為了有內容而硬掰。"""

VISION_PROMPT_NO_THUMBNAIL = """你是一個影片品質分析師。以下是一支 YouTube Shorts 依時間順序、每秒一張的截圖
（沒有取得官方縮圖，不需要也不要判斷縮圖是否符合內容）。

偵測重點：AI 生成跡象（肢體變形、材質過度平滑）、過度濾鏡、品牌 logo/浮水印、畫面重複度。

回覆 JSON 格式（不要加 markdown code block）:
{{
  "signals": ["<你真的在畫面中觀察到的問題，一句話具體描述，例如指出是哪一張截圖看到什麼>", "..."]
}}
重要：只回報你真的在提供的圖片裡看到的問題，不要憑空推測或套用常見說法；每一項都必須能指出具體是哪張圖、看到什麼異狀。
沒有把握或沒有發現任何問題，signals 回空陣列，不要為了有內容而硬掰。"""


def _fetch_thumbnail_bytes(url: str) -> bytes | None:
    # 縮圖抓不到（網路問題、影片沒有縮圖等）就靜靜跳過，不影響其他分析
    thumbnail_url = get_thumbnail_url(url)
    if not thumbnail_url:
        return None
    try:
        with urllib.request.urlopen(thumbnail_url, timeout=10) as resp:
            return resp.read()
    except Exception:
        return None


def run_vision_agent(state: AgentState) -> dict:
    # 1. 下載影片、每秒抽一幀（downloader.py 內建快取，重複分析不會重下載/重截）
    video_path = download_video(state["url"])
    frames = extract_frames(video_path)

    # 2. 額外抓官方縮圖，讓「縮圖與內容不符」這個訊號有真正的比對依據，而不是模型憑空腦補
    thumbnail_bytes = _fetch_thumbnail_bytes(state["url"])

    images = []
    if thumbnail_bytes:
        b64_thumb = base64.b64encode(thumbnail_bytes).decode()
        images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_thumb}"},
        })

    for frame in frames:
        b64 = base64.b64encode(frame.read_bytes()).decode()
        images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    prompt_text = VISION_PROMPT if thumbnail_bytes else VISION_PROMPT_NO_THUMBNAIL

    # 3. 組成 prompt 送給 GPT-4o
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
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
