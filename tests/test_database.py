import pytest
import os
from database import init_db, get_taste_profile, save_taste_profile, record_analysis, count_consecutive_trash

TEST_DB = "data/test_skipit_v2.db"


@pytest.fixture(autouse=True)
def setup_db():
    init_db(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_get_taste_profile_returns_none_for_new_user():
    assert get_taste_profile("U1", TEST_DB) is None


def test_save_and_get_taste_profile():
    save_taste_profile("U1", "## 討厭的內容\n- 心靈雞湯", TEST_DB)
    assert get_taste_profile("U1", TEST_DB) == "## 討厭的內容\n- 心靈雞湯"


def test_save_taste_profile_overwrites_existing():
    save_taste_profile("U1", "第一版", TEST_DB)
    save_taste_profile("U1", "第二版", TEST_DB)
    assert get_taste_profile("U1", TEST_DB) == "第二版"


def test_count_consecutive_trash_zero_when_no_history():
    assert count_consecutive_trash("U1", "UCabc", TEST_DB) == 0


def test_count_consecutive_trash_counts_streak():
    record_analysis("U1", "https://yt.be/1", "UCabc", "trash", 2, "廢片", [], TEST_DB)
    record_analysis("U1", "https://yt.be/2", "UCabc", "trash", 3, "廢片", [], TEST_DB)
    record_analysis("U1", "https://yt.be/3", "UCabc", "trash", 1, "廢片", [], TEST_DB)
    assert count_consecutive_trash("U1", "UCabc", TEST_DB) == 3


def test_count_consecutive_trash_stops_at_non_trash():
    record_analysis("U1", "https://yt.be/1", "UCabc", "trash", 2, "廢片", [], TEST_DB)
    record_analysis("U1", "https://yt.be/2", "UCabc", "keep", 8, "不錯", [], TEST_DB)
    record_analysis("U1", "https://yt.be/3", "UCabc", "trash", 2, "廢片", [], TEST_DB)
    # 最新兩支的順序：keep 之後又一支 trash -> 只算最新的連續段
    assert count_consecutive_trash("U1", "UCabc", TEST_DB) == 1
