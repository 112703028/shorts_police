import pytest
from unittest.mock import patch, MagicMock
from agents.scoring_agent import run_scoring_agent, _verdict_from_score
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
  "summary": "AI生成動物影片，無資訊價值",
  "tags": ["AI生成", "無資訊價值"]
}'''


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_scoring_agent_computes_overall_score_from_dimension_average():
    # scores 1,2,1,1,3 -> 平均 1.6 -> overall_score 16 -> trash
    state = _make_state(vision_signals=["手指變形疑似AI生成"], audio_signals=["幾乎無語音或純音樂"])
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        result = run_scoring_agent(state)

    assert result["overall_score"] == 16
    assert result["verdict"] == "trash"
    assert result["scores"]["ai_generated"] == 1
    assert "AI生成" in result["tags"]


def test_scoring_agent_ignores_llm_provided_overall_score_and_verdict():
    # 就算 LLM 硬塞矛盾的 overall_score/verdict，也完全不採用，一律從 scores 重新算
    conflicting = MOCK_RESPONSE.replace(
        '"summary"', '"overall_score": 95, "verdict": "keep", "summary"'
    )
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(conflicting)
        result = run_scoring_agent(state)

    assert result["overall_score"] == 16
    assert result["verdict"] == "trash"


def test_scoring_agent_clamps_out_of_range_dimension_scores():
    out_of_range = MOCK_RESPONSE.replace('"ai_generated": 1,', '"ai_generated": 999,')
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(out_of_range)
        result = run_scoring_agent(state)

    assert result["scores"]["ai_generated"] == 10  # 夾在 0-10
    # (10+2+1+1+3)/5 = 3.4 -> 34
    assert result["overall_score"] == 34


def test_scoring_agent_defaults_missing_dimension_to_neutral_five():
    missing_dimension = '{"scores": {"ai_generated": 10, "emotional_manipulation": 10, "originality": 10, "information_value": 10}, "summary": "x", "tags": []}'
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(missing_dimension)
        result = run_scoring_agent(state)

    assert result["scores"]["visual_quality"] == 5  # 缺欄位預設中性值 5
    # (10+10+10+10+5)/5 = 9 -> 90
    assert result["overall_score"] == 90
    assert result["verdict"] == "keep"


@pytest.mark.parametrize("score,expected", [
    (0, "trash"), (39, "trash"),
    (40, "review"), (69, "review"),
    (70, "keep"), (100, "keep"),
])
def test_verdict_from_score_boundaries(score, expected):
    assert _verdict_from_score(score) == expected
