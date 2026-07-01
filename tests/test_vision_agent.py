import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from agents.vision_agent import run_vision_agent
from models import SkipItState

BASE_STATE: SkipItState = {
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": "UCtest", "creator_name": "TestChan",
    "video_path": None, "audio_path": None, "frame_paths": None,
    "metadata_output": None, "vision_output": None, "audio_output": None,
    "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
    "score": None, "summary": None, "tags": None, "preference_updated": False,
}

MOCK_GPT_RESPONSE = '{"result": "AI 生成動物影片，畫面重複", "confidence": 0.85, "tags": ["AI生成", "無資訊價值"]}'


def test_vision_agent_returns_vision_output():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake_image_bytes"

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 5), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT_RESPONSE
        result = run_vision_agent(BASE_STATE)

    assert result["vision_output"]["agent"] == "vision"
    assert result["vision_output"]["confidence"] == 0.85
    assert "AI生成" in result["vision_output"]["tags"]


def test_vision_agent_low_confidence_flag():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake"
    low_conf = '{"result": "模糊畫面", "confidence": 0.2, "tags": []}'

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 5), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = low_conf
        result = run_vision_agent(BASE_STATE)

    assert result["vision_output"]["confidence"] < 0.3
