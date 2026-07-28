import pytest
from unittest.mock import patch, MagicMock
from agents.scoring_agent import run_scoring_agent, _verdict_from_score
from models import AgentState


def _make_state(metadata_signals=None, vision_signals=None, audio_signals=None, taste_profile="",
                 transcript="") -> AgentState:
    return {
        "user_id": "U1", "url": "https://yt.be/x", "creator_id": "UC1",
        "video_path": "tmp/x.mp4", "frames": [], "transcript": transcript,
        "metadata_signals": metadata_signals, "vision_signals": vision_signals,
        "audio_signals": audio_signals, "tags": None, "scores": None,
        "overall_score": None, "verdict": None, "summary": None,
        "taste_profile": taste_profile, "user_feedback": None,
        "should_early_stop": False, "skip_audio": False, "needs_reflection": False,
        "mismatch_reason": None,
    }


MOCK_RESPONSE = '''{
  "scores": {"authenticity": 1, "sincerity": 2, "originality": 1, "information_value": 1, "visual_quality": 3},
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
    assert result["scores"]["authenticity"] == 1
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
    out_of_range = MOCK_RESPONSE.replace('"authenticity": 1,', '"authenticity": 999,')
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(out_of_range)
        result = run_scoring_agent(state)

    assert result["scores"]["authenticity"] == 10  # 夾在 0-10
    # (10+2+1+1+3)/5 = 3.4 -> 34
    assert result["overall_score"] == 34


def test_scoring_agent_defaults_missing_dimension_to_neutral_five():
    missing_dimension = '{"scores": {"authenticity": 10, "sincerity": 10, "originality": 10, "information_value": 10}, "summary": "x", "tags": []}'
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


MOCK_MISMATCH_RESPONSE = '''{
  "scores": {"authenticity": 3, "sincerity": 4, "originality": 2, "information_value": 2, "visual_quality": 5},
  "summary": "畫面與語音疑似拼接",
  "tags": ["內容農場"],
  "content_mismatch": true,
  "mismatch_reason": "畫面是貓咪影片但語音在講政治新聞"
}'''


def test_scoring_agent_parses_content_mismatch_fields():
    state = _make_state(vision_signals=["貓咪追逐光點"], audio_signals=["提到政治話題"],
                         transcript="今天要來跟大家聊聊最新的選舉新聞")
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_MISMATCH_RESPONSE)
        result = run_scoring_agent(state)

    assert result["content_mismatch"] is True
    assert result["mismatch_reason"] == "畫面是貓咪影片但語音在講政治新聞"


def test_scoring_agent_forces_content_mismatch_false_when_no_transcript():
    # 沒有逐字稿（無語音/轉錄失敗）時，就算 LLM 誤判 content_mismatch=true 也要被強制改回 false，
    # 避免「沒有語音」本身被當成內容不一致，白白觸發一次不必要的 reflection
    state = _make_state(vision_signals=["貓咪追逐光點"], audio_signals=["幾乎無語音或純音樂"], transcript="")
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_MISMATCH_RESPONSE)
        result = run_scoring_agent(state)

    assert result["content_mismatch"] is False
    assert result["mismatch_reason"] == ""


def test_scoring_agent_defaults_content_mismatch_false_when_absent():
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        result = run_scoring_agent(state)

    assert result["content_mismatch"] is False
    assert result["mismatch_reason"] == ""


def test_scoring_agent_includes_reflection_block_when_needs_reflection():
    state = _make_state()
    state["needs_reflection"] = True
    state["mismatch_reason"] = "畫面是貓咪影片但語音在講政治新聞"
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        run_scoring_agent(state)

    sent_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "二次檢視" in sent_prompt
    assert "畫面是貓咪影片但語音在講政治新聞" in sent_prompt


def test_scoring_agent_omits_reflection_block_when_not_reflecting():
    state = _make_state()
    with patch("agents.scoring_agent._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        run_scoring_agent(state)

    sent_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "二次檢視" not in sent_prompt
