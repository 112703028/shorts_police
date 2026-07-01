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
MOCK_GPT = '{"result": "廣告推銷話術，誇大宣傳", "confidence": 0.9, "tags": ["廣告", "推銷"]}'


def test_audio_agent_returns_output():
    with patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = MOCK_TRANSCRIPT
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_audio_agent(BASE_STATE)

    assert result["audio_output"]["agent"] == "audio"
    assert "廣告" in result["audio_output"]["tags"]
    assert result["audio_output"]["confidence"] == 0.9


def test_audio_agent_sets_audio_path():
    with patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = MOCK_TRANSCRIPT
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_audio_agent(BASE_STATE)

    assert Path(result["audio_path"]) == Path("tmp/test.mp3")
