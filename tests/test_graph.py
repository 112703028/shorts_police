from unittest.mock import patch, DEFAULT
from graph import _score_with_reflection, run_pipeline_multi
from models import AgentState


def _make_state(needs_reflection=False) -> AgentState:
    return {
        "user_id": "U1", "url": "https://yt.be/x", "creator_id": "UC1", "creator_name": "C",
        "video_path": "tmp/x.mp4", "frames": [], "transcript": "",
        "metadata_signals": [], "vision_signals": [], "vision_description": None,
        "audio_signals": [], "tags": None, "scores": None,
        "overall_score": None, "verdict": None, "summary": None,
        "taste_profile": "", "user_feedback": None,
        "should_early_stop": False, "skip_audio": False,
        "needs_reflection": needs_reflection, "mismatch_reason": None,
    }


def test_score_with_reflection_triggers_second_call_on_content_mismatch():
    first_result = {"overall_score": 30, "verdict": "review", "summary": "s", "tags": [],
                     "scores": {}, "content_mismatch": True, "mismatch_reason": "貓咪畫面配政治語音"}
    second_result = {"overall_score": 15, "verdict": "trash", "summary": "s2", "tags": [],
                      "scores": {}, "content_mismatch": True, "mismatch_reason": "貓咪畫面配政治語音"}

    with patch("graph.run_scoring_agent", side_effect=[first_result, second_result]) as mock_scoring:
        result = _score_with_reflection(_make_state())

    assert mock_scoring.call_count == 2
    second_call_state = mock_scoring.call_args_list[1].args[0]
    assert second_call_state["needs_reflection"] is True
    assert second_call_state["mismatch_reason"] == "貓咪畫面配政治語音"
    assert result["overall_score"] == 15
    assert result["needs_reflection"] is True


def test_score_with_reflection_single_call_when_no_mismatch():
    result_no_mismatch = {"overall_score": 80, "verdict": "keep", "summary": "s", "tags": [],
                           "scores": {}, "content_mismatch": False, "mismatch_reason": ""}

    with patch("graph.run_scoring_agent", return_value=result_no_mismatch) as mock_scoring:
        result = _score_with_reflection(_make_state())

    assert mock_scoring.call_count == 1
    assert result["overall_score"] == 80


def test_score_with_reflection_does_not_loop_when_already_reflecting():
    # needs_reflection 已經是 True（代表這已經是第二輪），就算又回報 content_mismatch 也不再觸發第三輪
    still_mismatched = {"overall_score": 20, "verdict": "trash", "summary": "s", "tags": [],
                         "scores": {}, "content_mismatch": True, "mismatch_reason": "still off"}

    with patch("graph.run_scoring_agent", return_value=still_mismatched) as mock_scoring:
        result = _score_with_reflection(_make_state(needs_reflection=True))

    assert mock_scoring.call_count == 1
    assert result["overall_score"] == 20


_FAKE_METADATA = lambda state: {"metadata_signals": [], "creator_id": "UCblocked", "creator_name": "C"}
_FAKE_SCORED = lambda state: {
    "overall_score": 80, "verdict": "keep", "summary": "s", "tags": [], "scores": {},
    "content_mismatch": False, "mismatch_reason": "",
}


def test_run_pipeline_multi_skips_shared_analysis_when_all_blacklisted():
    with patch.multiple(
        "graph",
        _metadata_node=_FAKE_METADATA,
        check_blacklist=lambda uid, cid: {"should_early_stop": True},
        _download_node=DEFAULT, _vision_node=DEFAULT, _audio_node=DEFAULT,
        get_taste_profile=lambda uid: "",
        record_analysis=lambda **kwargs: None,
        check_implicit_blacklist=lambda uid, cid: None,
        _score_with_reflection=_FAKE_SCORED,
    ) as mocks:
        results = run_pipeline_multi(["A", "B"], "https://yt.be/x")

    mocks["_download_node"].assert_not_called()
    mocks["_vision_node"].assert_not_called()
    mocks["_audio_node"].assert_not_called()
    assert results["A"]["verdict"] == "trash"
    assert results["B"]["verdict"] == "trash"


def test_run_pipeline_multi_runs_shared_analysis_when_not_everyone_blacklisted():
    with patch.multiple(
        "graph",
        _metadata_node=_FAKE_METADATA,
        check_blacklist=lambda uid, cid: {"should_early_stop": uid == "A"},
        _download_node=DEFAULT,
        _vision_node=lambda state: {"vision_signals": [], "vision_description": ""},
        _audio_node=lambda state: {"audio_signals": [], "transcript": ""},
        get_taste_profile=lambda uid: "",
        record_analysis=lambda **kwargs: None,
        check_implicit_blacklist=lambda uid, cid: None,
        _score_with_reflection=_FAKE_SCORED,
    ) as mocks:
        mocks["_download_node"].return_value = {"video_path": "tmp/x.mp4", "skip_audio": True}
        results = run_pipeline_multi(["A", "B"], "https://yt.be/x")

    mocks["_download_node"].assert_called_once()
    assert results["A"]["verdict"] == "trash"  # 黑名單命中，早停
    assert results["B"]["verdict"] == "keep"   # 沒被封鎖，正常評分
