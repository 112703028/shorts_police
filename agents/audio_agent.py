import json
from pathlib import Path
from openai import OpenAI
from downloader import extract_audio
from models import AgentOutput, SkipItState
from config import GPT_MODEL, WHISPER_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_PROMPT = """你是一個影片品質分析師。以下是一支 YouTube Shorts 的語音轉錄文字：

「{transcript}」

請判斷這段語音內容是否為「廢片」，回覆以下 JSON 格式（不要加 markdown code block）:
{{
  "result": "一句話描述語音內容",
  "confidence": 0.0到1.0的信心分數,
  "tags": ["最多3個廢片標籤，例如：廣告、推銷、無意義重複、機翻字幕、純音樂無內容"]
}}"""


def run_audio_agent(state: SkipItState) -> dict:
    # 1. 重用 Vision Agent 已下載的影片，抽出音軌（不重新下載）
    video_path = Path(state["video_path"])
    audio_path = extract_audio(video_path)

    # 2. Whisper 轉錄成逐字稿
    with open(audio_path, "rb") as f:
        transcript = _client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            language="zh",
        ).text

    # 3. 組成 prompt 送給 GPT-4o 分析逐字稿內容（截 2000 字避免 prompt 過長）
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": AUDIO_PROMPT.format(transcript=transcript[:2000])}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    # 4. 包成 AgentOutput 格式
    output: AgentOutput = {
        "agent": "audio",
        "result": parsed.get("result", ""),
        "confidence": float(parsed.get("confidence", 0.5)),
        "tags": parsed.get("tags", []),
    }

    return {
        "audio_output": output,
        "audio_path": str(audio_path),
    }


if __name__ == "__main__":
    from downloader import download_video

    test_url = "https://www.youtube.com/shorts/99ObPP9MoBw"
    video_path = download_video(test_url)
    result = run_audio_agent({"video_path": str(video_path)})
    out = result["audio_output"]
    print("音訊路徑:", result["audio_path"])
    print("分析:", out["result"])
    print("信心:", out["confidence"])
    print("標籤:", out["tags"])
