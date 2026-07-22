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
         patch("agents.vision_agent._fetch_thumbnail_bytes", return_value=b"fake_thumbnail"), \
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
         patch("agents.vision_agent._fetch_thumbnail_bytes", return_value=b"fake_thumbnail"), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = clean_response
        result = run_vision_agent(BASE_STATE)

    assert result["vision_signals"] == []


def test_vision_agent_includes_thumbnail_image_when_available():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake_frame"
    clean_response = '{"signals": []}'

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 3), \
         patch("agents.vision_agent._fetch_thumbnail_bytes", return_value=b"fake_thumbnail_bytes"), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = clean_response
        run_vision_agent(BASE_STATE)

    # 縮圖抓到時，送進 GPT-4o 的內容應該是「縮圖 + 3 張截幀」= 4 張圖 + 1 段文字
    sent_content = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    image_blocks = [b for b in sent_content if b["type"] == "image_url"]
    assert len(image_blocks) == 4


def test_vision_agent_falls_back_when_no_thumbnail():
    mock_frame = MagicMock(spec=Path)
    mock_frame.read_bytes.return_value = b"fake_frame"
    clean_response = '{"signals": []}'

    with patch("agents.vision_agent.download_video", return_value=Path("tmp/test.mp4")), \
         patch("agents.vision_agent.extract_frames", return_value=[mock_frame] * 3), \
         patch("agents.vision_agent._fetch_thumbnail_bytes", return_value=None), \
         patch("agents.vision_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value.choices[0].message.content = clean_response
        run_vision_agent(BASE_STATE)

    # 縮圖抓不到時，只送 3 張截幀，且改用不提縮圖的 prompt
    sent_content = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    image_blocks = [b for b in sent_content if b["type"] == "image_url"]
    text_block = next(b for b in sent_content if b["type"] == "text")
    assert len(image_blocks) == 3
    assert "縮圖" not in text_block["text"] or "沒有取得官方縮圖" in text_block["text"]
