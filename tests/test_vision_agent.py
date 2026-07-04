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

MOCK_CONTENT = '{"content_result": "畫面重複，無資訊價值", "confidence": 0.85, "content_tags": ["重複畫面"]}'
MOCK_AUTHENTICITY = '{"authenticity_result": "手指變形，疑似AI生成", "has_watermark": false, "is_ai_generated": true, "authenticity_tags": ["AI生成"]}'


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_vision_agent_returns_vision_output():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake_image_bytes"

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 5), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response(MOCK_CONTENT),
            _mock_response(MOCK_AUTHENTICITY),
        ]
        result = run_vision_agent(BASE_STATE)

    output = result["vision_output"]
    assert output["agent"] == "vision"
    assert output["confidence"] == 0.85
    assert "重複畫面" in output["tags"]
    assert "AI生成" in output["tags"]
    assert output["is_ai_generated"] is True
    assert output["has_watermark"] is False


def test_vision_agent_low_confidence_flag():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake"
    low_conf = '{"content_result": "模糊畫面", "confidence": 0.2, "content_tags": []}'
    no_authenticity_issue = '{"authenticity_result": "無異常", "has_watermark": false, "is_ai_generated": false, "authenticity_tags": []}'

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 5), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response(low_conf),
            _mock_response(no_authenticity_issue),
        ]
        result = run_vision_agent(BASE_STATE)

    assert result["vision_output"]["confidence"] < 0.3
