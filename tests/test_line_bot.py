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


def test_format_verdict_trash():
    from line_bot import format_verdict
    result = format_verdict(overall_score=15, verdict="trash", summary="AI生成動物影片，無資訊價值")
    assert "❌" in result
    assert "15分" in result
    assert "AI生成動物影片，無資訊價值" in result
    assert "👍" not in result and "👎" not in result


def test_format_verdict_keep():
    from line_bot import format_verdict
    result = format_verdict(overall_score=85, verdict="keep", summary="製作精良，資訊豐富")
    assert "✅" in result
    assert "好片" in result


def test_format_verdict_review_fallback_for_unknown_verdict():
    from line_bot import format_verdict
    result = format_verdict(overall_score=50, verdict="unexpected", summary="邊界案例")
    assert "⚠️" in result
    assert "普通" in result


def test_format_verdict_includes_name_prefix_when_given():
    from line_bot import format_verdict
    result = format_verdict(overall_score=60, verdict="review", summary="邊界案例", name="Rachel")
    assert result.startswith("👤 Rachel：")


def test_format_verdict_no_name_prefix_by_default():
    from line_bot import format_verdict
    result = format_verdict(overall_score=60, verdict="review", summary="邊界案例")
    assert "👤" not in result


def test_format_verdict_includes_tags_when_given():
    from line_bot import format_verdict
    result = format_verdict(overall_score=15, verdict="trash", summary="測試", tags=["AI生成", "無資訊價值"])
    assert "🏷️ AI生成、無資訊價值" in result


def test_format_verdict_no_tag_line_when_absent():
    from line_bot import format_verdict
    result = format_verdict(overall_score=60, verdict="review", summary="邊界案例")
    assert "🏷️" not in result
