import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from line_bot import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_extract_youtube_url_from_text():
    from line_bot import extract_youtube_url
    text = "看這個 https://www.youtube.com/shorts/abc123 超好笑"
    assert extract_youtube_url(text) == "https://www.youtube.com/shorts/abc123"


def test_extract_youtube_url_returns_none_for_non_url():
    from line_bot import extract_youtube_url
    assert extract_youtube_url("hello world") is None


def test_format_reply_score():
    from line_bot import format_reply
    result = format_reply(score=3, summary="廣告推銷影片", tags=["廣告", "推銷"],
                           creator_name="TestChan", early_stop=False)
    assert "3/10" in result
    assert "廣告推銷影片" in result


def test_format_reply_early_stop():
    from line_bot import format_reply
    result = format_reply(score=1, summary="黑名單頻道", tags=["黑名單"],
                           creator_name="SpamChan", early_stop=True)
    assert "黑名單" in result
