import pytest
from unittest.mock import patch, MagicMock
from agents.scoring_agent import run_scoring_agent
from models import AgentState


def _make_state(metadata_signals=None, vision_signals=None, audio_signals=None, taste_profile="") -> AgentState:
    return {
        "user_id": "U1", "url": "https://yt.be/x", "creator_id": "UC1",
        "video_path": "tmp/x.mp4", "frames": [], "transcript": "",
        "metadata_signals": metadata_signals, "vision_signals": vision_signals,
        "audio_signals": audio_signals, "tags": None, "scores": None,
        "overall_score": None, "verdict": None, "summary": None,
        "taste_profile": taste_profile, "user_feedback": None,
        "should_early_stop": False, "skip_audio": False, "needs_reflection": False,
    }


MOCK_RESPONSE = '''{
  "scores": {"ai_generated": 1, "emotional_manipulation": 2, "originality": 1, "information_value": 1, "visual_quality": 3},
  "overall_score": 15,
  "verdict": "trash",
  "summary": "AI生成動物影片，無資訊價值",
  "tags": ["AI生成", "無資訊價值"]
}'''


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_scoring_agent_returns_verdict_and_scores():
    state = _make_state(vision_signals=["手指變形疑似AI生成"], audio_signals=["幾乎無語音或純音樂"])
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        result = run_scoring_agent(state)

    assert result["verdict"] == "trash"
    assert result["overall_score"] == 15
    assert result["scores"]["ai_generated"] == 1
    assert "AI生成" in result["tags"]


def test_scoring_agent_falls_back_to_review_on_invalid_verdict():
    bad_response = MOCK_RESPONSE.replace('"trash"', '"garbage"')
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(bad_response)
        result = run_scoring_agent(state)

    assert result["verdict"] == "review"


def test_scoring_agent_clamps_overall_score_range():
    out_of_range = MOCK_RESPONSE.replace('"overall_score": 15', '"overall_score": 150')
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(out_of_range)
        result = run_scoring_agent(state)

    assert result["overall_score"] == 100
