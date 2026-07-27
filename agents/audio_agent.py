import json
from pathlib import Path
from openai import OpenAI
from downloader import extract_audio, load_signal_cache, save_signal_cache
from acoustics import analyze_prosody, describe_acoustics
from models import AgentState
from config import GPT_MODEL, WHISPER_MODEL, OPENAI_API_KEY

_client = OpenAI(api_key=OPENAI_API_KEY)

AUDIO_PROMPT = """你是一個影片語音品質分析師。

【語音轉錄文字】
「{transcript}」

【聲學特徵（程式從音訊實際量測，不是猜的）】
{acoustics}

偵測重點：
1. 雞湯關鍵字（勵志語錄、心靈雞湯話術）
2. 誇大言詞（「不看後悔」「小心」型話術）
3. AI 語音特徵（TTS 機器人聲）— 綜合兩種證據判斷：
   - 文字面：完全沒有語助詞、沒有自我修正、句子結構過度工整、語氣詞缺席
   - 聲學面：音高變化與音量起伏都異常平穩（見上方數據）
   兩種證據都指向平穩才較有把握判為 TTS；只是文字正式、但聲學有明顯起伏，可能只是正經的真人朗讀，不要判成 TTS
4. 資訊密度低（實質內容少、填充廢話多）

回覆 JSON（不要加 markdown code block）:
{{
  "signals": ["<你真的在逐字稿或聲學數據裡看到的問題，一句話具體描述；若判為 TTS 請引用聲學數字佐證>"]
}}
只回報真的看得出來的問題，不要憑空推測。若沒有發現任何問題，signals 回空陣列。"""


def _transcribe(audio_path: Path) -> str:
    """Whisper 轉錄語音為文字"""
    with open(audio_path, "rb") as f:
        return _client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            language="zh",
        ).text


def run_audio_agent(state: AgentState) -> dict:
    # 語音訊號跟「誰在問」無關，同一支片重複分析直接用快取，不重打 Whisper/GPT-4o
    url = state.get("url")
    if url:
        cached = load_signal_cache(url, "audio")
        if cached is not None:
            return cached

    # 1. 重用 Vision Agent 已下載的影片，抽出音軌（不重新下載）
    video_path = Path(state["video_path"])
    audio_path = extract_audio(video_path)

    # 2. Whisper 轉錄成逐字稿
    transcript = _transcribe(audio_path)

    # 幾乎無語音時直接回報，不用再打一次 GPT（Orchestrator 也是靠 has_audio_track 決定跳不跳過這個 agent，這裡是保險）
    if len(transcript.strip()) < 10:
        result = {"audio_signals": ["幾乎無語音或純音樂"], "transcript": transcript}
        if url:
            save_signal_cache(url, "audio", result)
        return result

    # 3. 量測聲學特徵（音高/音量平穩度），讓 TTS 判斷有客觀數據佐證，而非純從文字猜
    acoustic_features = analyze_prosody(audio_path)

    # 4. 送 GPT-4o 綜合逐字稿 + 聲學數據判斷（截 2000 字避免 prompt 過長）
    response = _client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": AUDIO_PROMPT.format(
            transcript=transcript[:2000],
            acoustics=describe_acoustics(acoustic_features),
        )}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    parsed = json.loads(response.choices[0].message.content)

    result = {
        "audio_signals": parsed.get("signals", []),
        "transcript": transcript,
        "acoustic_features": acoustic_features,
    }
    if url:
        save_signal_cache(url, "audio", result)
    return result


if __name__ == "__main__":
    import sys
    from downloader import download_video

    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/shorts/XGX5roufnrs"
    video_path = download_video(url)
    result = run_audio_agent({"video_path": str(video_path), "url": url})
    print("逐字稿:", result["transcript"][:200])
    print("聲學特徵:", result.get("acoustic_features"))
    print("audio_signals:", result["audio_signals"])
