import json
from pathlib import Path
from openai import OpenAI
from downloader import extract_audio
from models import AgentState
from config import GPT_MODEL, WHISPER_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_PROMPT = """你是一個影片語音品質分析師。以下是一支 YouTube Shorts 的語音轉錄文字：

「{transcript}」

偵測重點：
1. 雞湯關鍵字（勵志語錄、心靈雞湯話術）
2. 恐懼訴求（「不看後悔」「小心」型話術）
3. AI 語音特徵（TTS 機器人聲：完全沒有語助詞、沒有自我修正、句子結構過度工整、語氣詞缺席）
4. 資訊密度低（實質內容少、填充廢話多）

回覆 JSON（不要加 markdown code block）:
{{
  "signals": ["<你真的在逐字稿裡看到的問題，一句話具體描述，可引用逐字稿中的字句>"]
}}
只回報逐字稿裡真的看得出來的問題，不要憑空推測。若沒有發現任何問題，signals 回空陣列。"""


def _transcribe(audio_path: Path) -> str:
    """Whisper 轉錄語音為文字"""
    with open(audio_path, "rb") as f:
        return _client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            language="zh",
        ).text


def run_audio_agent(state: AgentState) -> dict:
    # 1. 重用 Vision Agent 已下載的影片，抽出音軌（不重新下載）
    video_path = Path(state["video_path"])
    audio_path = extract_audio(video_path)

    # 2. Whisper 轉錄成逐字稿
    transcript = _transcribe(audio_path)

    # 幾乎無語音時直接回報，不用再打一次 GPT（Orchestrator 也是靠 has_audio_track 決定跳不跳過這個 agent，這裡是保險）
    if len(transcript.strip()) < 10:
        return {"audio_signals": ["幾乎無語音或純音樂"], "transcript": transcript}

    # 3. 送 GPT-4o 分析雞湯關鍵字/恐懼訴求/AI語音特徵/資訊密度（截 2000 字避免 prompt 過長）
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": AUDIO_PROMPT.format(transcript=transcript[:2000])}],
        response_format={"type": "json_object"},
    )
    parsed = json.loads(response.choices[0].message.content)

    return {
        "audio_signals": parsed.get("signals", []),
        "transcript": transcript,
    }


if __name__ == "__main__":
    import sys
    from downloader import download_video

    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/shorts/XGX5roufnrs"
    video_path = download_video(url)
    result = run_audio_agent({"video_path": str(video_path)})
    print("逐字稿:", result["transcript"][:200])
    print("audio_signals:", result["audio_signals"])
