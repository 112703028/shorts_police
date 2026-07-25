import pytest
from unittest.mock import patch
from agents.metadata_agent import run_metadata_agent
from models import AgentState

BASE_STATE: AgentState = {
    "user_id": "U1",
    "url": "https://www.youtube.com/shorts/test123",
    "creator_id": None, "creator_name": None, "video_path": None, "frames": None,
    "transcript": None, "metadata_signals": None, "vision_signals": None,
    "audio_signals": None, "tags": None, "scores": None, "overall_score": None,
    "verdict": None, "summary": None, "taste_profile": "", "user_feedback": None,
    "should_early_stop": False, "skip_audio": False, "needs_reflection": False,
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

MOCK_GPT = '{"signals": []}'


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


def test_metadata_agent_returns_signals_list():
    signals_response = '{"signals": ["按讚率0.3%疑似買流量", "標題使用震驚等煽情用語"]}'
    with patch("agents.metadata_agent.yt_dlp.YoutubeDL") as mock_ydl, \
         patch("agents.metadata_agent._client") as mock_client:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = MOCK_INFO
        mock_client.chat.completions.create.return_value.choices[0].message.content = signals_response
        result = run_metadata_agent(BASE_STATE)
    assert result["metadata_signals"] == ["按讚率0.3%疑似買流量", "標題使用震驚等煽情用語"]


def test_metadata_agent_empty_signals_when_clean():
    with patch("agents.metadata_agent.yt_dlp.YoutubeDL") as mock_ydl, \
         patch("agents.metadata_agent._client") as mock_client:
        mock_ydl.return_value.__enter__.return_value.extract_info.return_value = MOCK_INFO
        mock_client.chat.completions.create.return_value.choices[0].message.content = MOCK_GPT
        result = run_metadata_agent(BASE_STATE)
    assert result["metadata_signals"] == []
