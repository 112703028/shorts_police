# SkipIt Bot v2 — Design Spec
Date: 2026-07-14
Supersedes: 2026-06-29-skipit-bot-design.md

## Overview

SkipIt Bot 是一個 Agentic AI 系統，在使用者點開 YouTube Shorts 之前自動分析並回傳廢片判定。使用者在 LINE 群組傳入 YouTube Shorts 連結，Bot 回傳判定結果（trash/review/keep）、五維評分、一句話理由，並學習每個使用者的個人品味。

**與 v1 的核心差異**：v1 用結構化 SQLite 表格記錄黑名單/觀察名單；v2 改用**自然語言品味檔案**，讓 LLM 自己判斷使用者回饋屬於偏好改變還是規則細化，更貼近「這個系統會學習」的敘事。v1 評分是單一 1–10 分數；v2 拆成五個維度 + 三分類判定（trash/review/keep），資訊更豐富也更容易做個人化 prompt。

---

## Architecture

### System Diagram

```
[LINE 群組]
  → 傳入 YouTube Shorts URL
  → [LINE Bot Webhook] (FastAPI)
  → [Orchestrator]  ← 核心大腦，讀取 taste_profile，動態決策
       ├─ [Metadata Agent]  yt-dlp：發文頻率、訂閱數異常、標題標點密度、hashtag 相關性
       ├─ [Vision Agent]    ffmpeg 每秒抽一幀 → Claude Vision：AI生成跡象、誇張縮圖、濾鏡、logo、重複度
       └─ [Audio Agent]     ffmpeg 抽音訊 → Whisper 轉錄 → NLP：雞湯關鍵字、恐懼訴求、AI語音特徵、資訊密度
  → [Scoring Agent]   整合 signals + taste_profile → 五維評分 + overall_score + verdict + summary
  → [LINE Bot 回覆]   「❌ 廢片（19分）— 一句話理由」+「這個判定準確嗎？👍 / 👎」
  → （使用者回覆 👍/👎）
  → [Preference Agent]  推理回饋是否與現有規則矛盾，重寫 taste_profile
```

### Team Division

| 吳廷翰 | Tim |
|---|---|
| Orchestrator + Metadata Agent | Vision Agent + Audio Agent |
| taste_profile 讀寫邏輯 + 條件邊設計 + Preference Agent | Scoring Agent |
| LangGraph graph 組裝 | LINE Bot + 整合測試 |

---

## Tech Stack

| 層次 | 工具 |
|---|---|
| LINE Bot | `line-bot-sdk-python` v3 |
| Agent 框架 | LangGraph |
| 影片下載 + metadata | yt-dlp（下載影片本體，也用 `extract_flat` 抓頻道近期影片列表算發文頻率） |
| Vision 分析 | GPT-4o（OpenAI）—— **階段一先用 OpenAI，之後再換成 Claude** |
| 語音轉錄 | OpenAI Whisper API |
| Orchestrator / Scoring / Preference LLM | GPT-4o（OpenAI）—— 之後再換成 Claude |
| 偏好儲存 | SQLite（`taste_profile` 為自然語言純文字欄位） |
| Webhook 伺服器 | FastAPI |
| 本地 tunnel | ngrok |

> **註**：LLM 供應商階段性先用 OpenAI，架構（AgentState、graph 節點、taste_profile 機制）已經按照未來換成 Anthropic Claude 的需求設計，之後只需要把各 agent 內的 `OpenAI(...)` client 換成 `anthropic.Anthropic(...)`，prompt 格式與呼叫方式微調即可，不需要動 graph 結構。

---

## AgentState

```python
class AgentState(TypedDict):
    user_id: str
    url: str
    video_path: str
    frames: list[str]
    transcript: str
    metadata_signals: list[str]
    vision_signals: list[str]
    audio_signals: list[str]
    tags: list[str]
    scores: dict            # {"ai_generated": int, "emotional_manipulation": int, "originality": int, "information_value": int, "visual_quality": int}
    overall_score: int
    verdict: str             # "trash" | "review" | "keep"
    summary: str
    taste_profile: str       # 使用者品味檔案（自然語言全文）
    user_feedback: str       # 使用者回饋原文（👍/👎 或文字說明）
```

---

## LangGraph Nodes

系統共六個節點，Orchestrator 動態決定路由（不是固定 pipeline）：

### 1. `orchestrator`
接收 URL，讀取 `taste_profile`（依 `user_id` 查 SQLite），動態決定哪些 Agent 要執行：
- **帳號在黑名單**（taste_profile 的「封鎖頻道」區塊命中）→ 提早終止，直接輸出 `verdict: "trash"`
- **影片無語音** → 跳過 `audio_agent`
- **Vision 與 Audio 結果矛盾**（例如 Vision 判斷內容有價值但 Audio 偵測到大量雞湯關鍵字）→ 觸發 Scoring Agent 二次 reflection

### 2. `metadata_agent`（平行執行）
用 **yt-dlp**（`skip_download=True` 抓單支影片 metadata，`extract_flat` 抓頻道近期影片列表）：
- 頻道發文頻率（近期影片上傳時間戳計算間隔）
- 訂閱數/觀看數比例異常（買流量嫌疑，`channel_follower_count` 可取得時使用）
- 標題標點符號密度（「😱😱😱」「！！！」型標題黨偵測）
- hashtag 與影片實際內容的相關性
輸出 `metadata_signals: list[str]`

### 3. `vision_agent`（平行執行）
- `yt-dlp` 下載影片
- `ffmpeg` **每秒抽一幀**（不是固定 5 張，密度隨影片長度變動）
- 縮圖 + 關鍵幀送 GPT-4o Vision，偵測：
  - AI 生成跡象（肢體變形、材質過度平滑）
  - 誇張縮圖（與內容不符的縮圖標題黨）
  - 過度濾鏡
  - 品牌 logo / 浮水印
  - 畫面重複度
輸出 `vision_signals: list[str]`

### 4. `audio_agent`（條件執行，只有影片有語音才跑）
- `ffmpeg` 抽取音訊
- Whisper 轉錄
- NLP 分析：
  - 雞湯關鍵字（勵志語錄、心靈雞湯話術）
  - 恐懼訴求（「不看後悔」「小心」型話術）
  - AI 語音特徵（TTS 機器人聲，語調平板、無語助詞）
  - 資訊密度（實質內容 vs 填充廢話比例）
輸出 `audio_signals: list[str]` 和 `transcript: str`

### 5. `scoring_agent`
整合所有 signals，**讀取 `taste_profile` 放進 prompt** 做個人化評分，輸出：
- 五維評分（`scores` dict）：`ai_generated`、`emotional_manipulation`、`originality`、`information_value`、`visual_quality`（每項 0–10）
- `overall_score`：綜合分數
- `verdict`：`"trash"` | `"review"` | `"keep"` 三分類
- `summary`：一句話理由

### 6. `preference_agent`（只在收到使用者回饋時觸發）
接收 `user_feedback`（👍/👎 或文字說明）、這支影片的 `tags`、現有 `taste_profile`，讓 LLM 推理：
- 這個回饋是否與現有規則矛盾？
- 這是「偏好改變」還是「規則需要細化」？
- 應該新增例外規則、修改既有規則，還是都不用動？
最後**重寫整份 `taste_profile`** 存回 SQLite。

---

## `taste_profile` 格式（自然語言，存在 SQLite TEXT 欄位）

依 `user_id` 各自獨立一份。五個區塊：

```
## 討厭的內容
- 心靈雞湯/勵志語錄類影片
- AI 生成的動物/寵物影片

## 例外規則
- [2026-07-10] 使用者說：「這個雖然是雞湯但是我朋友拍的，不要判廢片」
  → 例外：頻道 XXX 的雞湯類內容不算廢片

## 封鎖頻道
- UCxxxxx（連續 3 次低分後自動加入）

## 喜歡的頻道
- UCyyyyy

## 尚未確定
- 開箱影片：使用者對這類影片的回饋不一致，還在觀察
```

每條例外規則附時間戳記和使用者原話，讓 Preference Agent 之後重寫時有完整脈絡可以參考，不會憑空覆蓋掉之前的判斷依據。

---

## LINE Bot 流程

1. 群組有人傳 YouTube Shorts 連結
2. Webhook 觸發 FastAPI `/webhook`
3. Bot 立即回「分析中... ⏳」
4. 背景非同步跑 LangGraph pipeline
5. 完成後用 `push_message` 回群組：
   ```
   ❌ 廢片（19分）— AI 生成動物，無資訊價值

   這個判定準確嗎？回覆 👍 或 👎
   ```
6. 使用者回覆 👍 或 👎（或文字說明）→ 觸發 `preference_agent` 更新該使用者的 `taste_profile`

---

## 檔案結構

```
skipit_bot/
├── config.py
├── models.py               # AgentState TypedDict
├── downloader.py           # yt-dlp 下載 + ffmpeg 截幀/抽音軌（共用）
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── metadata_agent.py   # yt-dlp（單片 metadata + 頻道近期影片列表）
│   ├── vision_agent.py     # GPT-4o Vision（之後換 Claude）
│   ├── audio_agent.py      # Whisper + NLP
│   ├── scoring_agent.py    # 五維評分 + verdict
│   └── preference_agent.py # taste_profile 讀寫
├── graph/
│   └── pipeline.py         # LangGraph 組裝
├── bot/
│   └── line_bot.py         # FastAPI webhook
├── db/
│   └── database.py         # SQLite：taste_profile 表
├── data/                   # SQLite DB（gitignored）
├── tmp/                    # 下載暫存（gitignored）
└── tests/
```

---

## 環境變數（`.env`）

```
OPENAI_API_KEY=sk-...
LINE_CHANNEL_ACCESS_TOKEN=...
LINE_CHANNEL_SECRET=...
```

> `ANTHROPIC_API_KEY` 之後換 Claude 時再加。

---

## Demo Script

同 v1：執行三條連結（確定廢片、正常影片、邊界案例），展示重點：
- 黑名單命中 → 提早終止
- 影片無語音 → 自動跳過 Audio Agent
- 使用者回覆 👍/👎 後，taste_profile 即時更新（可以現場秀 SQLite 內容變化，證明系統真的在學習）
