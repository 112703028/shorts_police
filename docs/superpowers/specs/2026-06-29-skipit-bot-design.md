# SkipIt Bot — Design Spec
Date: 2026-06-29

## Overview

SkipIt Bot 是一個 Agentic AI 系統，自動判斷 YouTube Shorts 是否為「廢片」並給出評分。使用者在 LINE 群組傳入 YouTube Shorts 連結，Bot 回傳評分（1–10）、五維分析與一句話理由。

---

## Architecture

### System Diagram

```
[LINE 群組]
  → 傳入 YouTube Shorts URL
  → [LINE Bot Webhook] (FastAPI)
  → [Orchestrator Agent]  ← 核心大腦，動態決策
       ├─ [Metadata Agent]    YouTube 標題、描述、頻道、觀看數
       ├─ [Vision Agent]      擷取幀 → GPT-4o 分析畫面
       └─ [Audio Agent]       yt-dlp 音訊 → Whisper 轉錄
  → [Scoring Agent]          整合結果，輸出評分 + 摘要
  → [Preference Agent]       對照黑名單，記錄偏好
  → [LINE Bot 回覆]
```

### Team Division

| 吳廷翰 | Tim |
|---|---|
| Orchestrator + Metadata Agent | Vision Agent + Audio Agent |
| 黑名單邏輯 + 條件邊 + Preference Agent | Scoring Agent |
| LangGraph graph 組裝 | LINE Bot + 整合測試 |

---

## Tech Stack

| 層次 | 工具 |
|---|---|
| LINE Bot | `line-bot-sdk-python` v3 |
| Agent 框架 | LangGraph |
| 影片下載 | yt-dlp |
| Vision 分析 | GPT-4o (OpenAI) |
| 語音轉錄 | OpenAI Whisper API |
| Orchestrator LLM | GPT-4o (OpenAI) |
| 偏好儲存 | SQLite |
| Webhook 伺服器 | FastAPI |
| 本地 tunnel | ngrok |

---

## Orchestrator Dynamic Routing

Orchestrator 不是固定 pipeline，而是根據情況動態決策：

1. **黑名單命中** → 提早終止，直接輸出低分（不執行完整分析）
2. **影片無語音** → 跳過 Audio Agent
3. **Vision 回報「無法判斷」** → 重新擷取不同時間點的幀（最多重試 1 次）
4. **Vision 與 Audio 結論矛盾** → 觸發 Scoring Agent 二次 reflection

這四個條件邊是系統 agentic 行為的核心展示點。

---

## Agent Interfaces

所有 Agent 回傳統一格式：

```python
{
    "agent": str,          # "metadata" | "vision" | "audio" | "scoring"
    "result": str,         # 分析結果文字
    "confidence": float,   # 0.0 - 1.0
    "tags": list[str]      # e.g. ["廣告", "AI生成", "無資訊價值"]
}
```

---

## Data Storage (SQLite)

```sql
CREATE TABLE analysis_history (
    id          INTEGER PRIMARY KEY,
    url         TEXT,
    creator     TEXT,
    score       INTEGER,
    summary     TEXT,
    tags        TEXT,        -- JSON array
    analyzed_at TIMESTAMP
);

CREATE TABLE creator_preferences (
    creator     TEXT PRIMARY KEY,
    status      TEXT,        -- 'blacklist' | 'watchlist' | 'whitelist'
    avg_score   REAL,
    count       INTEGER,
    updated_at  TIMESTAMP
);

CREATE TABLE tag_preferences (
    tag           TEXT PRIMARY KEY,
    dislike_count INTEGER,
    updated_at    TIMESTAMP
);
```

**學習規則：**
- 同一頻道累積 3 次分數 ≤ 4 → 自動升為 `blacklist`
- 1–2 次低分 → 列 `watchlist`（完整分析，但 Orchestrator 知道有前科）
- 某 tag 出現 5 次以上 → Scoring Agent 加重扣分

---

## LINE Bot Output Format

```
📊 評分：3/10
📝 摘要：AI 生成動物影片，無實際資訊價值
🏷️ 標籤：#AI生成 #無資訊 #重複內容
👤 頻道已列入觀察名單
```

---

## Shared Utilities

`downloader.py` — 共用模組，供 Vision Agent 與 Audio Agent 共同使用：
- `download_video(url) -> Path`
- `extract_frames(video_path, n=5) -> list[Path]`
- `extract_audio(video_path) -> Path`

---

## Demo Script

執行三條連結：
1. 確定廢片（AI 生成動物）
2. 正常影片（有實際資訊）
3. 邊界案例（娛樂性高但資訊量低）

展示重點：
- 黑名單命中 → 提早終止（Agentic 關鍵行為 1）
- 影片無語音 → 自動跳過 Audio Agent（Agentic 關鍵行為 2）

---

## Cost Estimate (per video)

- GPT-4o Vision（5 幀）≈ $0.01–0.02
- Whisper（1 分鐘）≈ $0.006
- 每支影片總計 ≈ **$0.02–0.03 USD**
