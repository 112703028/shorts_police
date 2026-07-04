import json
from pathlib import Path
from openai import OpenAI
from downloader import extract_audio
from models import AgentOutput, SkipItState
from config import GPT_MODEL, WHISPER_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

CONTENT_PROMPT = """你是一個影片品質分析師，根據以下語音轉錄文字判斷內容品質。

【語音轉錄文字】
「{transcript}」

請分析內容本身：
- 是否有廣告推銷話術？（「買就對了」「限時優惠」「點擊下方連結」）
- 內容是否有實際資訊價值？
- 是否只是重複同樣的話？
- 語氣是否過於誇張煽情？

回覆 JSON（不要加 markdown code block）:
{{
  "transcript_summary": "一句話摘要語音內容",
  "content_result": "一句話描述內容問題",
  "content_tags": ["最多2個標籤，例如：廣告話術、內容空洞、重複唸稿、煽情誇大"]
}}"""

TONE_PROMPT = """你是一個語音品質分析師，從語音轉錄文字推測說話者的語調特徵。

【語音轉錄文字】
「{transcript}」

TTS 機器人聲在文字上有這些特徵：
1. 完全沒有語助詞（嗯、啊、那個、就是、對不對）
2. 沒有自我修正（「不對，我是說...」「等等...」）
3. 每句話結構完整、正式，像在唸稿
4. 句子長度高度均勻，沒有長短變化
5. 語氣詞、感嘆詞完全缺席（哇、真的假的、天啊）

回覆 JSON（不要加 markdown code block）:
{{
  "tone_result": "一句話描述語調特徵",
  "is_tts": true或false,
  "has_filler_words": true或false,
  "tone_tags": ["最多2個標籤，例如：TTS機器人聲、語調平板、機械節奏、唸稿感強"]
}}"""


def _transcribe(audio_path: Path) -> str:
    """Whisper 轉錄語音為文字"""
    with open(audio_path, "rb") as f:
        return _client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            language="zh",
        ).text


def _analyze_tone(transcript: str) -> dict:
    """從 Whisper 轉錄文字推測語調特徵"""
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": TONE_PROMPT.format(transcript=transcript[:2000])}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def _analyze_content(transcript: str) -> dict:
    """GPT-4o 分析文字內容"""
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": CONTENT_PROMPT.format(transcript=transcript[:2000])}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def run_audio_agent(state: SkipItState) -> dict:
    video_path = Path(state["video_path"])
    audio_path = extract_audio(video_path)

    # 1. Whisper 轉錄
    transcript = _transcribe(audio_path)

    # 無語音處理
    if len(transcript.strip()) < 10:
        output: AgentOutput = {
            "agent": "audio",
            "result": "影片幾乎無語音或純音樂",
            "confidence": 0.9,
            "tags": ["無語音內容"],
            "is_tts": False,
            "has_filler_words": False,
            "transcript_summary": "無語音",
        }
        return {"audio_output": output, "audio_path": str(audio_path)}

    # 2. 同時分析：文字內容 + 語調推測
    content = _analyze_content(transcript)
    tone = _analyze_tone(transcript)

    # 3. 合併兩個分析的標籤
    all_tags = list(set(
        content.get("content_tags", []) +
        tone.get("tone_tags", [])
    ))

    result_text = f"{content.get('content_result', '')}；{tone.get('tone_result', '')}".strip("；")

    output: AgentOutput = {
        "agent": "audio",
        "result": result_text,
        "confidence": 0.85 if tone.get("is_tts") or all_tags else 0.6,
        "tags": all_tags,
        "is_tts": tone.get("is_tts", False),
        "has_filler_words": tone.get("has_filler_words", True),
        "transcript_summary": content.get("transcript_summary", ""),
    }

    return {"audio_output": output, "audio_path": str(audio_path)}


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/shorts/XGX5roufnrs"

    from downloader import download_video
    video_path = download_video(url)

    fake_state: SkipItState = {
        "url": url, "video_path": str(video_path),
        "creator_id": None, "creator_name": None,
        "audio_path": None, "frame_paths": None,
        "metadata_output": None, "vision_output": None, "audio_output": None,
        "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
        "score": None, "summary": None, "tags": None, "preference_updated": False,
    }

    result = run_audio_agent(fake_state)
    out = result["audio_output"]
    print("摘要:", out.get("transcript_summary", ""))
    print("分析:", out["result"])
    print("標籤:", out["tags"])
    print("是否 TTS:", out.get("is_tts", False))
    print("有語助詞:", out.get("has_filler_words", True))
