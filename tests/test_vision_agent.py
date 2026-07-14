import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from agents.vision_agent import run_vision_agent
from models import AgentState

BASE_STATE: AgentState = {
    "user_id": "U1",
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": None, "video_path": None, "frames": None, "transcript": None,
    "metadata_signals": None, "vision_signals": None, "audio_signals": None,
    "tags": None, "scores": None, "overall_score": None, "verdict": None,
    "summary": None, "taste_profile": "", "user_feedback": None,
    "should_early_stop": False, "skip_audio": False, "needs_reflection": False,
}

MOCK_RESPONSE = '{"signals": ["手指變形疑似AI生成", "縮圖與內容不符"]}'


def test_vision_agent_returns_signals():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake_image_bytes"

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 10), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_RESPONSE
        result = run_vision_agent(BASE_STATE)

    assert result["vision_signals"] == ["手指變形疑似AI生成", "縮圖與內容不符"]
    assert len(result["frames"]) == 10


def test_vision_agent_no_signals_when_clean():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake"
    clean_response = '{"signals": []}'

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 5), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = clean_response
        result = run_vision_agent(BASE_STATE)

    assert result["vision_signals"] == []
