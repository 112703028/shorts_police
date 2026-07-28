import base64
import json
import urllib.request
from pathlib import Path
import imagehash
from PIL import Image
from openai import OpenAI
from downloader import download_video, extract_frames, get_thumbnail_url, load_signal_cache, save_signal_cache
from models import AgentState
from config import GPT_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

_AI_SIGNAL_CRITERIA = """AI 生成跡象 — 優先找現代生成模型常見的破綻（比舊式的手指變形更可靠，因為新模型已經很少犯舊錯誤）：
   - 畫面中的文字/招牌扭曲、筆畫錯誤或呈現無意義符號
   - 背景細節無意義地重複或糊化（真實場景不會有這種規律性）
   - 光影方向不一致（主體光源跟背景陰影對不上）
   - 皮膚/毛髮/材質過度均勻、呈現蠟感
   - 物理不合理（液體/布料/毛髮運動方式不符合真實物理）
   - 肢體變形（手指數量錯誤、關節不自然）——這條線索較舊，現代模型較少犯，權重放低，不要只靠這條就下結論"""

_FRAME_CONSISTENCY_CRITERION = """相鄰截圖之間的不連貫 — 對照上方「相鄰幀視覺差異分數」（程式實際算出來的數據，
   不是猜的），異常偏高的那幾對，具體看那兩張截圖哪裡不合理（正常運鏡/動作造成的合理變化不算，
   要能講出「這不是鏡頭或動作能解釋的」；不要因為有數字就硬掰出問題）"""

_SIGNAL_FORMAT_NOTE = """回覆 JSON 格式（不要加 markdown code block）:
{{
  "description": "<一句話描述畫面實際內容，例如：一隻橘貓在客廳地板追逐紅色雷射光點>",
  "signals": ["<具體描述，必須包含：(1) 是第幾張截圖或第幾張跟第幾張之間 (2) 看到什麼異狀。
  例如：第3張截圖手指有6根，AI生成跡象；第2張與第3張之間背景招牌文字內容不同但鏡頭沒有移動>", "..."]
}}
重要：只回報你真的在提供的圖片裡看到的問題，不要憑空推測或套用常見說法；每一項都必須指出具體是哪張圖（或哪兩張之間）、看到什麼異狀。
沒有把握或沒有發現任何問題，signals 回空陣列，不要為了有內容而硬掰。"""

VISION_PROMPT_TEMPLATE = """你是一個影片品質分析師。第一張圖是這支 YouTube Shorts 的官方縮圖（使用者滑動時會看到的畫面），
之後依時間順序、每秒一張的是影片實際截圖（第1張截圖、第2張截圖...依序編號，縮圖不算在內）。

【相鄰幀視覺差異分數（perceptual hash 距離，程式實際計算，不是猜的）】
{frame_diffs}

偵測重點：
1. 縮圖與內容不符 — 明確比對第一張縮圖跟後面影片截圖是否呈現同一件事，只有真的看出落差才回報
2. {ai_criteria}
3. {frame_consistency}
4. 過度濾鏡
5. 品牌 logo/浮水印
6. 畫面重複度（可參考上方差異分數是否持續偏低）

{signal_format_note}"""

VISION_PROMPT_NO_THUMBNAIL_TEMPLATE = """你是一個影片品質分析師。以下是一支 YouTube Shorts 依時間順序、每秒一張的截圖
（第1張截圖、第2張截圖...依序編號；沒有取得官方縮圖，不需要也不要判斷縮圖是否符合內容）。

【相鄰幀視覺差異分數（perceptual hash 距離，程式實際計算，不是猜的）】
{frame_diffs}

偵測重點：
1. {ai_criteria}
2. {frame_consistency}
3. 過度濾鏡
4. 品牌 logo/浮水印
5. 畫面重複度（可參考上方差異分數是否持續偏低）

{signal_format_note}"""


def _frame_diff_scores(frames: list[Path]) -> list[int]:
    """相鄰幀之間的視覺差異（perceptual hash 漢明距離），數字越大代表兩張截圖內容差異越大。"""
    hashes = [imagehash.phash(Image.open(f)) for f in frames]
    return [int(hashes[i] - hashes[i + 1]) for i in range(len(hashes) - 1)]


def _describe_frame_diffs(diffs: list[int]) -> str:
    """把差異分數轉成給 GPT 看的證據文字，標出異常偏高的幀對，把注意力指到具體哪裡，
    而不是叫模型自己憑印象掃過全部截圖找不連貫。"""
    if not diffs:
        return "（幀數不足，無法比對）"
    # 「正常」範圍因內容而異（遊戲實況本來就比訪談類動得快），不寫死絕對數字，
    # 只給這支影片自己的平均值當基準，異常判斷也用相對這支片自己的平均去算，不跟其他影片比
    avg = sum(diffs) / len(diffs)
    lines = [f"這支影片平均差異分數：{avg:.1f}（僅供對照下面的異常值，不代表跨影片的絕對標準）"]

    threshold = max(20, avg * 3)
    outliers = [(i, d) for i, d in enumerate(diffs, start=1) if d > threshold]
    if outliers:
        detail = "、".join(f"第{i}→{i + 1}張截圖:{d}" for i, d in outliers[:5])
        lines.append(f"異常偏高的幀間差異（遠高於平均，值得仔細檢查）：{detail}")

    near_zero = sum(1 for d in diffs if d <= 2)
    if near_zero >= max(3, len(diffs) // 3):
        lines.append(f"有 {near_zero} 對相鄰截圖差異極小（≤2），畫面可能重複或停滯")

    return "\n".join(lines)


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
    # 畫面訊號跟「誰在問」無關，同一支片重複分析直接用快取，不重打 GPT-4o Vision
    cached = load_signal_cache(state["url"], "vision")
    if cached is not None:
        return cached

    # 1. 下載影片、每秒抽一幀（downloader.py 內建快取，重複分析不會重下載/重截）
    video_path = download_video(state["url"])
    frames = extract_frames(video_path)

    # 1b. 算相鄰幀的實際視覺差異分數，取代單純叫模型「憑印象」比對相鄰截圖
    frame_diffs_desc = _describe_frame_diffs(_frame_diff_scores(frames))

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

    prompt_template = VISION_PROMPT_TEMPLATE if thumbnail_bytes else VISION_PROMPT_NO_THUMBNAIL_TEMPLATE
    prompt_text = prompt_template.format(
        frame_diffs=frame_diffs_desc,
        ai_criteria=_AI_SIGNAL_CRITERIA,
        frame_consistency=_FRAME_CONSISTENCY_CRITERION,
        signal_format_note=_SIGNAL_FORMAT_NOTE,
    )

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
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content)

    result = {
        "vision_signals": parsed.get("signals", []),
        "vision_description": parsed.get("description", ""),
        "video_path": str(video_path),
        "frames": [str(f) for f in frames],
    }
    save_signal_cache(state["url"], "vision", result)
    return result


if __name__ == "__main__":
    result = run_vision_agent({"url": "https://www.youtube.com/shorts/99ObPP9MoBw"})
    print("影片路徑:", result["video_path"])
    print("幀數:", len(result["frames"]))
    print("vision_signals:", result["vision_signals"])
