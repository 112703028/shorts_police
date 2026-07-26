from unittest.mock import patch, MagicMock
from agents.preference_agent import run_preference_agent, check_blacklist, check_implicit_blacklist

MOCK_RESPONSE = '''{
  "new_taste_profile": "【使用者品味檔案】\\n討厭：AI生成動物/寵物 [2026-07-25]",
  "creator_action": "none",
  "learned": "使用者不喜歡AI生成動物"
}'''


def _mock_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = content
    return response


def test_feedback_prompt_includes_vision_and_audio_content():
    with patch("agents.preference_agent._client") as mock_client, \
         patch("agents.preference_agent.save_taste_profile"):
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        run_preference_agent({
            "user_id": "U1", "taste_profile": "既有品味檔案",
            "user_feedback": "他說的『不看後悔』其實是反諷啦", "tags": ["雞湯"],
            "vision_description": "一隻貓在追雷射光點",
            "transcript": "今天要跟大家分享一個不看後悔的秘密",
        })

    sent_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "一隻貓在追雷射光點" in sent_prompt
    assert "今天要跟大家分享一個不看後悔的秘密" in sent_prompt


def test_feedback_prompt_defaults_when_vision_and_audio_missing():
    with patch("agents.preference_agent._client") as mock_client, \
         patch("agents.preference_agent.save_taste_profile"):
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        run_preference_agent({
            "user_id": "U1", "taste_profile": "既有品味檔案",
            "user_feedback": "還可以", "tags": ["雞湯"],
        })

    sent_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "（無資料）" in sent_prompt


def test_onboarding_does_not_require_vision_or_audio():
    with patch("agents.preference_agent._client") as mock_client, \
         patch("agents.preference_agent.save_taste_profile") as mock_save:
        mock_client.chat.completions.create.return_value = _mock_response(MOCK_RESPONSE)
        result = run_preference_agent({
            "user_id": "U1", "taste_profile": "",
            "user_feedback": "1,3", "tags": [],
        })

    assert result["learned"] == "使用者不喜歡AI生成動物"
    mock_save.assert_called_once()


def test_feedback_triggers_blacklist_action():
    blacklist_response = MOCK_RESPONSE.replace('"creator_action": "none"', '"creator_action": "blacklist"')
    with patch("agents.preference_agent._client") as mock_client, \
         patch("agents.preference_agent.save_taste_profile"), \
         patch("agents.preference_agent.add_to_blacklist") as mock_add:
        mock_client.chat.completions.create.return_value = _mock_response(blacklist_response)
        result = run_preference_agent({
            "user_id": "U1", "taste_profile": "既有品味檔案",
            "user_feedback": "封鎖這頻道", "tags": ["雞湯"],
            "creator_id": "UCtest", "creator_name": "測試頻道",
        })

    mock_add.assert_called_once_with("U1", "UCtest", reason="使用者明確封鎖")
    assert "已封鎖" in result["reply_note"]


def test_check_blacklist_reflects_is_blacklisted():
    with patch("agents.preference_agent.is_blacklisted", return_value=True):
        assert check_blacklist("U1", "UCabc")["should_early_stop"] is True
    with patch("agents.preference_agent.is_blacklisted", return_value=False):
        assert check_blacklist("U1", "UCabc")["should_early_stop"] is False


def test_check_implicit_blacklist_adds_after_threshold():
    with patch("agents.preference_agent.count_consecutive_trash", return_value=3), \
         patch("agents.preference_agent.add_to_blacklist") as mock_add:
        check_implicit_blacklist("U1", "UCabc")
    mock_add.assert_called_once_with("U1", "UCabc", reason="連續3次trash自動偵測")


def test_check_implicit_blacklist_does_not_add_below_threshold():
    with patch("agents.preference_agent.count_consecutive_trash", return_value=2), \
         patch("agents.preference_agent.add_to_blacklist") as mock_add:
        check_implicit_blacklist("U1", "UCabc")
    mock_add.assert_not_called()
