import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from agents.audio_agent import run_audio_agent
from models import SkipItState

BASE_STATE: SkipItState = {
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": "UCtest", "creator_name": "TestChan",
    "video_path": "tmp/test.mp4", "audio_path": None, "frame_paths": None,
    "metadata_output": None, "vision_output": None, "audio_output": None,
    "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
    "score": None, "summary": None, "tags": None, "preference_updated": False,
}

MOCK_TRANSCRIPT = "今天來開箱這個超厲害的產品，買就對了，限時優惠..."

MOCK_CONTENT = '{"transcript_summary": "推銷開箱影片", "content_result": "廣告推銷話術，誇大宣傳", "content_tags": ["廣告話術"]}'
MOCK_TONE = '{"tone_result": "語調平板無語助詞", "is_tts": true, "has_filler_words": false, "tone_tags": ["TTS機器人聲"]}'


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_audio_agent_returns_output():
    with patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = MOCK_TRANSCRIPT
        mock_client.chat.completions.create.side_effect = [
            _mock_response(MOCK_CONTENT),
            _mock_response(MOCK_TONE),
        ]
        result = run_audio_agent(BASE_STATE)

    output = result["audio_output"]
    assert output["agent"] == "audio"
    assert "廣告話術" in output["tags"]
    assert "TTS機器人聲" in output["tags"]
    assert output["is_tts"] is True


def test_audio_agent_no_speech_short_circuits():
    with patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = "  "
        result = run_audio_agent(BASE_STATE)

    output = result["audio_output"]
    assert output["tags"] == ["無語音內容"]
    assert output["confidence"] == 0.9
    # 沒有語音時不應該再打 GPT 分析內容/語調
    mock_client.chat.completions.create.assert_not_called()


def test_audio_agent_sets_audio_path():
    with patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = MOCK_TRANSCRIPT
        mock_client.chat.completions.create.side_effect = [
            _mock_response(MOCK_CONTENT),
            _mock_response(MOCK_TONE),
        ]
        result = run_audio_agent(BASE_STATE)

    assert Path(result["audio_path"]) == Path("tmp/test.mp3")
