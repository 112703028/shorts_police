import pytest
from unittest.mock import patch
from agents.metadata_agent import run_metadata_agent
from models import SkipItState

BASE_STATE: SkipItState = {
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": None, "creator_name": None,
    "video_path": None, "audio_path": None, "frame_paths": None,
    "metadata_output": None, "vision_output": None, "audio_output": None,
    "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
    "score": None, "summary": None, "tags": None, "preference_updated": False,
}

MOCK_INFO = {
    "title": "超可愛貓咪影片",
    "description": "每天分享貓咪 #cat #cute",
    "uploader": "CatChannel",
    "channel_id": "UCcat123",
    "view_count": 50000,
    "like_count": 1200,
    "duration": 30,
    "tags": ["cat", "cute"],
}

MOCK_GPT = '{"result": "可愛動物影片，娛樂性高", "confidence": 0.6, "tags": []}'


def test_metadata_agent_returns_creator_id():
    with patch("agents.metadata_agent.yt_dlp.YoutubeDL") as mock_ydl, \
         patch("agents.metadata_agent._client") as mock_client:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = MOCK_INFO
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_metadata_agent(BASE_STATE)
    assert result["creator_id"] == "UCcat123"


def test_metadata_agent_returns_creator_name():
    with patch("agents.metadata_agent.yt_dlp.YoutubeDL") as mock_ydl, \
         patch("agents.metadata_agent._client") as mock_client:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = MOCK_INFO
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_metadata_agent(BASE_STATE)
    assert result["creator_name"] == "CatChannel"


def test_metadata_agent_output_format():
    with patch("agents.metadata_agent.yt_dlp.YoutubeDL") as mock_ydl, \
         patch("agents.metadata_agent._client") as mock_client:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = MOCK_INFO
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_metadata_agent(BASE_STATE)
    output = result["metadata_output"]
    assert output["agent"] == "metadata"
    assert isinstance(output["result"], str)
    assert isinstance(output["tags"], list)
    assert 0.0 <= output["confidence"] <= 1.0
