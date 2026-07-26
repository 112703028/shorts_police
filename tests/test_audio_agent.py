import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from agents.audio_agent import run_audio_agent
from models import AgentState

BASE_STATE: AgentState = {
    "user_id": "U1",
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": None, "video_path": "tmp/test.mp4", "frames": None, "transcript": None,
    "metadata_signals": None, "vision_signals": None, "audio_signals": None,
    "tags": None, "scores": None, "overall_score": None, "verdict": None,
    "summary": None, "taste_profile": "", "user_feedback": None,
    "should_early_stop": False, "skip_audio": False, "needs_reflection": False,
}

MOCK_TRANSCRIPT = "只要相信自己，你就能成功，人生沒有過不去的坎，加油！"
MOCK_GPT = '{"signals": ["雞湯關鍵字：勵志語錄", "資訊密度低"]}'


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_audio_agent_returns_signals():
    with patch("agents.audio_agent.load_signal_cache", return_value=None), \
         patch("agents.audio_agent.save_signal_cache"), \
         patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = MOCK_TRANSCRIPT
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_GPT)
        result = run_audio_agent(BASE_STATE)

    assert result["audio_signals"] == ["雞湯關鍵字：勵志語錄", "資訊密度低"]
    assert result["transcript"] == MOCK_TRANSCRIPT


def test_audio_agent_short_circuits_on_no_speech():
    with patch("agents.audio_agent.load_signal_cache", return_value=None), \
         patch("agents.audio_agent.save_signal_cache"), \
         patch("agents.audio_agent.extract_audio", return_value=Path("tmp/test.mp3")), \
         patch("builtins.open", mock_open(read_data=b"fake_audio")), \
         patch("agents.audio_agent._client") as mock_client:
        mock_client.audio.transcriptions.create.return_value.text = "  "
        result = run_audio_agent(BASE_STATE)

    assert result["audio_signals"] == ["幾乎無語音或純音樂"]
    mock_client.chat.completions.create.assert_not_called()
