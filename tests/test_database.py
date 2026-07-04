import pytest
import os
from database import init_db, record_analysis, get_cached_analysis

TEST_DB = "data/test_skipit_cache.db"


@pytest.fixture(autouse=True)
def setup_db():
    init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_get_cached_analysis_returns_none_when_not_analyzed():
    assert get_cached_analysis("https://yt.be/never-seen", TEST_DB) is None


def test_get_cached_analysis_returns_stored_result():
    record_analysis("https://yt.be/abc", "UC1", 3, "廣告推銷影片", ["廣告"], TEST_DB)
    cached = get_cached_analysis("https://yt.be/abc", TEST_DB)
    assert cached["score"] == 3
    assert cached["summary"] == "廣告推銷影片"
    assert cached["tags"] == ["廣告"]
    assert cached["creator_id"] == "UC1"


def test_get_cached_analysis_returns_most_recent():
    record_analysis("https://yt.be/xyz", "UC2", 5, "第一次分析", [], TEST_DB)
    record_analysis("https://yt.be/xyz", "UC2", 8, "第二次分析", [], TEST_DB)
    cached = get_cached_analysis("https://yt.be/xyz", TEST_DB)
    assert cached["score"] == 8
    assert cached["summary"] == "第二次分析"
