import pytest
from unittest.mock import patch
from agents.scoring_agent import run_scoring_agent
from models import SkipItState, AgentOutput


def _make_state(metadata_tags=[], vision_conf=0.8, audio_tags=[], vision_tags=[]) -> SkipItState:
    return {
        "url": "https://yt.be/x", "creator_id": "UC1", "creator_name": "Chan",
        "video_path": "tmp/x.mp4", "audio_path": "tmp/x.mp3", "frame_paths": [],
        "metadata_output": AgentOutput(agent="metadata", result="test meta", confidence=0.7, tags=metadata_tags),
        "vision_output": AgentOutput(agent="vision", result="test vision", confidence=vision_conf, tags=vision_tags),
        "audio_output": AgentOutput(agent="audio", result="test audio", confidence=0.8, tags=audio_tags),
        "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
        "score": None, "summary": None, "tags": None, "preference_updated": False,
    }


MOCK_SCORE_RESPONSE = '{"score": 3, "summary": "廣告推銷影片，無資訊價值", "tags": ["廣告", "推銷"]}'


def test_scoring_agent_returns_score():
    state = _make_state(metadata_tags=["廣告"], audio_tags=["推銷"])
    with patch("agents.scoring_agent._client") as mock_client, \
         patch("agents.scoring_agent.get_tag_dislike_count", return_value=0):
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_SCORE_RESPONSE
        result = run_scoring_agent(state)
    assert result["score"] == 3
    assert isinstance(result["summary"], str)
    assert isinstance(result["tags"], list)


def test_scoring_agent_score_in_range():
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client, \
         patch("agents.scoring_agent.get_tag_dislike_count", return_value=0):
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_SCORE_RESPONSE
        result = run_scoring_agent(state)
    assert 1 <= result["score"] <= 10
