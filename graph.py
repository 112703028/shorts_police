"""
SkipIt Bot v2 的核心：LangGraph 有向圖 + Orchestrator 動態決策。

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
1. 黑名單早停 — 條件邊，查 SQL 表，省下下載/Vision/Whisper 的成本
2. 無語音跳過 Audio — download 節點偵測音軌，條件邊路由
3. Vision/Audio 訊號落差過大 → 二次 reflection — scoring 節點內判斷後重跑一次
4. 連續 3 次 trash 自動封鎖（隱性偏好學習）— preference 節點呼叫 check_implicit_blacklist
"""
from langgraph.graph import StateGraph, END

from models import AgentState
from agents.metadata_agent import run_metadata_agent
from agents.vision_agent import run_vision_agent
from agents.audio_agent import run_audio_agent
from agents.scoring_agent import run_scoring_agent
from agents.preference_agent import check_blacklist, check_implicit_blacklist
from downloader import download_video, has_audio_track
from database import init_db, get_taste_profile, record_analysis

init_db()

# 一邊訊號很多、另一邊完全沒有訊號，視為落差過大，觸發 reflection
_REFLECTION_SIGNAL_GAP = 2


def _log(emoji: str, node: str, msg: str) -> None:
    # Demo 時投影用的決策過程 log，讓觀眾看到 Orchestrator 每一步在想什麼
    print(f"{emoji} [{node}] {msg}", flush=True)


# ---------- 節點 ----------

def _metadata_node(state: AgentState) -> dict:
    _log("📥", "Metadata", "抓取影片與頻道資訊...")
    result = run_metadata_agent(state)
    _log("📥", "Metadata", f"完成 → 頻道: {result['creator_name']}, signals: {result['metadata_signals']}")
    return result


def _blacklist_node(state: AgentState) -> dict:
    result = check_blacklist(state["user_id"], state["creator_id"])
    if result["should_early_stop"]:
        _log("🚫", "Orchestrator", "黑名單命中！提早終止，省下下載+Vision+Whisper 成本")
    else:
        _log("✅", "Orchestrator", "非黑名單，繼續完整分析")
    return result


def _early_stop_node(state: AgentState) -> dict:
    return {
        "scores": {},
        "overall_score": 0,
        "verdict": "trash",
        "summary": "頻道已列入黑名單，自動跳過",
        "tags": ["黑名單"],
    }


def _download_node(state: AgentState) -> dict:
    _log("⬇️", "Download", "下載影片中...")
    video_path = download_video(state["url"])
    skip_audio = not has_audio_track(video_path)
    if skip_audio:
        _log("🔇", "Orchestrator", "偵測到無音軌 → 決定跳過 Audio Agent")
    else:
        _log("🔊", "Orchestrator", "有音軌 → Audio Agent 將執行")
    return {"video_path": str(video_path), "skip_audio": skip_audio}


def _vision_node(state: AgentState) -> dict:
    _log("👁️", "Vision", "每秒擷取一幀，送 GPT-4o 分析...")
    result = run_vision_agent(state)
    _log("👁️", "Vision", f"完成 → signals: {result['vision_signals']}")
    return result


def _audio_node(state: AgentState) -> dict:
    _log("🎧", "Audio", "Whisper 轉錄 + 語意分析中...")
    result = run_audio_agent(state)
    _log("🎧", "Audio", f"完成 → signals: {result['audio_signals']}")
    return result


def _scoring_node(state: AgentState) -> dict:
    _log("🧮", "Scoring", "整合三方訊號 + 個人品味檔案，計算評分...")
    result = run_scoring_agent(state)

    # Agentic 行為 3：Vision/Audio 訊號落差過大 → 二次 reflection
    vision_signals = state.get("vision_signals") or []
    audio_signals = state.get("audio_signals") or []
    signal_gap = abs(len(vision_signals) - len(audio_signals))
    if not state.get("needs_reflection") and signal_gap >= _REFLECTION_SIGNAL_GAP:
        _log("🔄", "Orchestrator",
             f"Vision({len(vision_signals)}) 與 Audio({len(audio_signals)}) 訊號落差過大 → 觸發二次 reflection")
        reflection_state = {**state, **result, "needs_reflection": True}
        result = run_scoring_agent(reflection_state)
        result["needs_reflection"] = True

    _log("🧮", "Scoring", f"評分: {result['overall_score']}/100 ({result['verdict']}) — {result['summary']}")
    return result


def _preference_node(state: AgentState) -> dict:
    record_analysis(
        user_id=state["user_id"],
        url=state["url"],
        creator_id=state.get("creator_id") or "",
        creator_name=state.get("creator_name"),
        verdict=state.get("verdict") or "review",
        overall_score=state.get("overall_score") or 0,
        summary=state.get("summary") or "",
        tags=state.get("tags") or [],
    )
    # Agentic 行為 4：連續 3 次 trash，即使使用者沒表態也自動封鎖
    creator_id = state.get("creator_id")
    if creator_id:
        check_implicit_blacklist(state["user_id"], creator_id)
    _log("💾", "Preference", "已記錄分析結果，更新黑名單學習狀態")
    return {}


# ---------- 條件邊的路由函式 ----------

def _route_after_blacklist(state: AgentState) -> str:
    # Agentic 行為 1：黑名單早停
    return "early_stop" if state["should_early_stop"] else "download"


def _route_after_vision(state: AgentState) -> str:
    # Agentic 行為 2：無語音跳過 Audio
    return "scoring" if state.get("skip_audio") else "audio"


# ---------- 組裝圖 ----------

def build_graph():
    graph = StateGraph(AgentState)

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


def run_pipeline(user_id: str, url: str) -> dict:
    """line_bot.py 的入口：丟入 user_id + URL，走完整張圖，回傳最終判定結果。"""
    initial: AgentState = {
        "user_id": user_id,
        "url": url,
        "creator_id": None, "creator_name": None,
        "video_path": None, "frames": None, "transcript": None,
        "metadata_signals": None, "vision_signals": None, "audio_signals": None,
        "tags": None, "scores": None, "overall_score": None, "verdict": None,
        "summary": None,
        "taste_profile": get_taste_profile(user_id) or "",
        "user_feedback": None,
        "should_early_stop": False, "skip_audio": False, "needs_reflection": False,
    }
    final = _compiled_graph.invoke(initial)
    return {
        "overall_score": final.get("overall_score"),
        "verdict": final.get("verdict"),
        "summary": final.get("summary"),
        "tags": final.get("tags"),
        "creator_id": final.get("creator_id"),
        "creator_name": final.get("creator_name"),
        "should_early_stop": final.get("should_early_stop", False),
    }


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/shorts/99ObPP9MoBw"
    print(f"\n=== SkipIt Bot 分析: {url} ===\n")
    result = run_pipeline("demo_user", url)
    print("\n=== 最終結果 ===")
    print(f"判定: {result['verdict']} ({result['overall_score']}/100)")
    print(f"摘要: {result['summary']}")
    print(f"標籤: {result['tags']}")
    print(f"頻道: {result['creator_name']}")
