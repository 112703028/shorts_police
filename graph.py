"""
SkipIt Bot 的核心：LangGraph 有向圖 + Orchestrator 動態決策。

圖的結構：
    metadata → check_blacklist ─┬─（黑名單）→ early_stop ────────┐
                                └─（正常）  → download            │
                                              ↓                   │
                                       ◆ 有音軌？                 │
                                    vision ─┬─（無聲）→ scoring   │
                                            └─（有聲）→ audio → scoring
                                                          ↓
                                                     preference → END

四個 agentic 行為：
1. 黑名單早停 — 條件邊，省下下載/Vision/Whisper 的成本
2. 無語音跳過 Audio — download 節點偵測音軌，條件邊路由
3. Vision 低信心重試 — vision 節點內迴圈，重新擷取不同幀
4. 結論矛盾 reflection — scoring prompt 內建，回報 conflict 旗標
"""
import shutil
from pathlib import Path

from langgraph.graph import StateGraph, END

from models import SkipItState
from agents.metadata_agent import run_metadata_agent
from agents.vision_agent import run_vision_agent
from agents.audio_agent import run_audio_agent
from agents.scoring_agent import run_scoring_agent
from agents.preference_agent import run_preference_agent, check_blacklist
from downloader import download_video, has_audio_track
from database import init_db
from config import FRAME_COUNT

init_db()


def _log(emoji: str, node: str, msg: str) -> None:
    # Demo 時投影用的決策過程 log，讓觀眾看到 Orchestrator 每一步在想什麼
    print(f"{emoji} [{node}] {msg}", flush=True)


# ---------- 節點 ----------

def _metadata_node(state: SkipItState) -> dict:
    _log("📥", "Metadata", "抓取影片與頻道資訊...")
    result = run_metadata_agent(state)
    out = result["metadata_output"]
    _log("📥", "Metadata", f"完成 → 頻道: {result['creator_name']}, tags: {out['tags']}")
    return result


def _blacklist_node(state: SkipItState) -> dict:
    result = check_blacklist(state)
    if result["should_early_stop"]:
        _log("🚫", "Orchestrator", "黑名單命中！提早終止，省下下載+Vision+Whisper 成本")
    else:
        _log("✅", "Orchestrator", "非黑名單，繼續完整分析")
    return result


def _early_stop_node(state: SkipItState) -> dict:
    return {
        "score": 1,
        "summary": "頻道已列入黑名單，自動跳過",
        "tags": ["黑名單"],
    }


def _download_node(state: SkipItState) -> dict:
    _log("⬇️", "Download", "下載影片中...")
    video_path = download_video(state["url"])
    skip_audio = not has_audio_track(video_path)
    if skip_audio:
        _log("🔇", "Orchestrator", "偵測到無音軌 → 決定跳過 Audio Agent")
    else:
        _log("🔊", "Orchestrator", "有音軌 → Audio Agent 將執行")
    return {"video_path": str(video_path), "skip_audio": skip_audio}


def _vision_node(state: SkipItState) -> dict:
    _log("👁️", "Vision", f"擷取 {FRAME_COUNT} 幀，送 GPT-4o 分析...")
    result = run_vision_agent(state)
    conf = result["vision_output"]["confidence"]

    # Agentic 行為 3：低信心重試 — 刪掉舊幀，重新擷取不同內容再試一次
    if conf < 0.3 and state.get("vision_retry_count", 0) < 1:
        _log("🔄", "Orchestrator", f"Vision 信心過低 ({conf}) → 決定重新擷取幀再試一次")
        frames_dir = Path(result["video_path"]).parent / f"{Path(result['video_path']).stem}_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        retry_state = {**state, **result, "vision_retry_count": 1}
        result = run_vision_agent(retry_state)
        result["vision_retry_count"] = 1
        conf = result["vision_output"]["confidence"]

    _log("👁️", "Vision", f"完成 → {result['vision_output']['result']} (信心: {conf})")
    return result


def _audio_node(state: SkipItState) -> dict:
    _log("🎧", "Audio", "Whisper 轉錄 + 語調分析中...")
    result = run_audio_agent(state)
    out = result["audio_output"]
    _log("🎧", "Audio", f"完成 → {out['result']} (tags: {out['tags']})")
    return result


def _scoring_node(state: SkipItState) -> dict:
    _log("🧮", "Scoring", "整合三方結果 + 個人品味檔案，計算最終評分...")
    result = run_scoring_agent(state)
    _log("🧮", "Scoring", f"評分: {result['score']}/10 — {result['summary']}")
    return result


def _preference_node(state: SkipItState) -> dict:
    result = run_preference_agent(state)
    _log("💾", "Preference", "已記錄分析結果，更新頻道與標籤學習狀態")
    return result


# ---------- 條件邊的路由函式 ----------

def _route_after_blacklist(state: SkipItState) -> str:
    # Agentic 行為 1：黑名單早停
    return "early_stop" if state["should_early_stop"] else "download"


def _route_after_vision(state: SkipItState) -> str:
    # Agentic 行為 2：無語音跳過 Audio
    return "scoring" if state.get("skip_audio") else "audio"


# ---------- 組裝圖 ----------

def build_graph():
    graph = StateGraph(SkipItState)

    graph.add_node("metadata", _metadata_node)
    graph.add_node("check_blacklist", _blacklist_node)
    graph.add_node("early_stop", _early_stop_node)
    graph.add_node("download", _download_node)
    graph.add_node("vision", _vision_node)
    graph.add_node("audio", _audio_node)
    graph.add_node("scoring", _scoring_node)
    graph.add_node("preference", _preference_node)

    graph.set_entry_point("metadata")
    graph.add_edge("metadata", "check_blacklist")
    graph.add_conditional_edges("check_blacklist", _route_after_blacklist, {
        "early_stop": "early_stop",
        "download": "download",
    })
    graph.add_edge("early_stop", "preference")
    graph.add_edge("download", "vision")
    graph.add_conditional_edges("vision", _route_after_vision, {
        "audio": "audio",
        "scoring": "scoring",
    })
    graph.add_edge("audio", "scoring")
    graph.add_edge("scoring", "preference")
    graph.add_edge("preference", END)

    return graph.compile()


_compiled_graph = build_graph()


def run_pipeline(url: str) -> dict:
    """line_bot.py 的入口：丟入 URL，走完整張圖，回傳最終評分結果。"""
    initial: SkipItState = {
        "url": url,
        "creator_id": None, "creator_name": None,
        "video_path": None, "audio_path": None, "frame_paths": None,
        "metadata_output": None, "vision_output": None, "audio_output": None,
        "should_early_stop": False, "skip_audio": False, "vision_retry_count": 0,
        "score": None, "summary": None, "tags": None, "preference_updated": False,
    }
    final = _compiled_graph.invoke(initial)
    return {
        "score": final.get("score"),
        "summary": final.get("summary"),
        "tags": final.get("tags"),
        "creator_name": final.get("creator_name"),
        "should_early_stop": final.get("should_early_stop", False),
    }


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/shorts/99ObPP9MoBw"
    print(f"\n=== SkipIt Bot 分析: {url} ===\n")
    result = run_pipeline(url)
    print("\n=== 最終結果 ===")
    print(f"評分: {result['score']}/10")
    print(f"摘要: {result['summary']}")
    print(f"標籤: {result['tags']}")
    print(f"頻道: {result['creator_name']}")
