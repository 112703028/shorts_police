# SkipIt Bot（shorts_police）

在你點開 YouTube Shorts **之前**，先幫你判斷它是不是廢片。

在 LINE 貼上 Shorts 連結，Bot 會下載影片，並從 metadata、畫面、語音三方面分析，再依照**你個人的品味檔案**給出判定：

```
❌ 廢片（23分）— AI 生成寵物配罐頭旁白
🏷️ 貓咪、搞笑
```

你可以回覆 👍/👎 或直接打字說明，Bot 會學習你的喜好，之後的判定就會越來越貼近你的口味。

---

## 系統架構

以 **LangGraph** 組成的多 Agent 有向圖，由 Orchestrator 依條件動態決定路由，而不是走一條固定的 pipeline。

```
[LINE] 貼 Shorts 連結
   → FastAPI Webhook (line_bot.py)
   → metadata → check_blacklist ─┬─（黑名單）→ early_stop ─────────┐
                                 └─（正常）  → download             │
                                                 ↓                  │
                                           vision ─┬─（無聲）→ scoring
                                                   └─（有聲）→ audio → scoring
                                                                     ↓
                                                               preference → END
   → LINE 回覆判定結果
   → 使用者回饋 👍/👎/文字 → Preference Agent 重寫品味檔案
```

### Agents

| Agent | 檔案 | 做什麼 |
|---|---|---|
| Metadata | [agents/metadata_agent.py](agents/metadata_agent.py) | 用 yt-dlp 抓影片與頻道資訊：按讚率、發片頻率、頻道主題一致性、煽情標題、導流連結 |
| Vision | [agents/vision_agent.py](agents/vision_agent.py) | ffmpeg 每秒抽一幀，加上官方縮圖送 GPT-4o，偵測 AI 生成跡象、縮圖不符、濾鏡、浮水印、畫面重複。另用 perceptual hash 計算相鄰幀差異，讓模型根據實際數據判斷 |
| Audio | [agents/audio_agent.py](agents/audio_agent.py) | Whisper 轉錄（自動偵測語言），偵測雞湯、誇大話術、資訊密度低；並結合 [acoustics.py](acoustics.py) 量測的音高與音量起伏來判斷是否為 TTS 機器人聲 |
| Scoring | [agents/scoring_agent.py](agents/scoring_agent.py) | 整合三方訊號與品味檔案，輸出五維評分、一句話理由與主題標籤 |
| Preference | [agents/preference_agent.py](agents/preference_agent.py) | 處理首次問卷與使用者回饋，以自然語言重寫品味檔案，並管理頻道黑名單 |

### 四個 Agentic 行為

1. **黑名單早停**：頻道已被封鎖時，直接判為廢片，省下下載、Vision 和 Whisper 的成本。
2. **無語音跳過 Audio**：先檢查有沒有音軌，再用 volumedetect 檢查是否近乎靜音；沒有聲音就跳過 Audio Agent。
3. **二次 reflection**：Scoring 發現畫面與語音內容兜不起來（疑似拼接或內容農場）時，會帶著不一致的原因重新評分一次。
4. **隱性黑名單**：同一頻道連續 3 次被判為 trash，即使使用者沒有表態也會自動封鎖（門檻由 `IMPLICIT_BLACKLIST_THRESHOLD` 設定）。

### 評分方式

五個維度都是 0–10 分，**分數越高越好**：

| 維度 | 意義 |
|---|---|
| `authenticity` | 真實度（非 AI 生成） |
| `sincerity` | 真誠度（沒有情緒操弄） |
| `originality` | 原創度 |
| `information_value` | 資訊價值 |
| `visual_quality` | 畫面品質 |

`overall_score` 與 `verdict` 由程式算出，**不交給 LLM 自己回報**，確保數字前後一致：

- `overall_score` = 五維平均 × 10（0–100）
- `verdict`：低於 40 為 `trash` ❌ 廢片；40–69 為 `review` ⚠️ 普通；70 以上為 `keep` ✅ 好片

LINE 訊息只顯示總分、理由和標籤，五維細項只印在終端機 log。

### 品味檔案（taste_profile）

每個使用者各有一份**自然語言**的品味檔案，存在 SQLite。Preference Agent 收到回饋時會判斷這是「偏好改變」還是「規則需要細化」，盡量細化而不是直接推翻舊規則（例如「討厭開箱」會細化成「討厭純推銷開箱，接受深度評測」）。檔案最多 15 條規則。

---

## LINE Bot 使用流程

1. **第一次互動**：無論傳什麼，Bot 都會先發問卷：「你覺得什麼樣的 Shorts 算廢片？」可以直接打字描述，也可以回數字選項（例如 `1,3`）。如果觸發問卷的那則訊息帶有連結，回答完問卷後會接著分析那支影片。
2. **貼連結**：Bot 會先回「🔍 分析中」，分析完再推播判定結果。支援 `youtube.com/shorts/...` 與 `youtu.be/...` 兩種網址。
3. **回饋**：直接回覆 👍/👎 或文字。若用 LINE 的「引用回覆」指定某則判定訊息，回饋會精準對應到那支影片；沒有引用時，則視為針對聊天室最新一次的判定。處理完後，Bot 會秀出更新後的品味檔案。
   - 提到「封鎖」、「不想再看到」→ 封鎖該頻道；提到「解封」、「其實還好」→ 解除封鎖。
4. **群組模式**：在群組貼連結時，Metadata、Vision、Audio 只分析一次，但會依 `line_bot.py` 中 `DEMO_USER_IDS` 的每個人，各自套用自己的品味檔案評分，並分別發出判定。群組裡任何人都可以回饋，但只會更新自己的品味檔案。

> 問卷與回饋的等待狀態存在 process 記憶體中，重啟服務後就會清空（目前是 demo 規模）。

---

## 安裝與執行

### 需求

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/)（需在 PATH 中，用於抽幀、抽音訊與音量偵測）
- OpenAI API key（GPT-4o + Whisper）
- LINE Messaging API channel
- ngrok（本地開發時讓 LINE Webhook 連得進來）

### 安裝

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install imagehash            # Vision Agent 會用到，目前還沒列進 requirements.txt
```

### 環境變數（`.env`）

```env
OPENAI_API_KEY=...
LINE_CHANNEL_SECRET=...
LINE_CHANNEL_ACCESS_TOKEN=...

# 選填
YT_DLP_COOKIES_FILE=...          # YouTube 要求登入驗證時使用的 cookies.txt 路徑（優先）
YT_DLP_COOKIES_BROWSER=chrome    # 或直接讀取瀏覽器的 cookies：chrome / safari / firefox
DISABLE_SIGNAL_CACHE=true        # 調整 prompt 時強制重跑分析，不讀取快取
```

其他設定（模型、門檻、併發數）請見 [config.py](config.py)。

### 啟動

```bash
python main.py                   # 初始化 SQLite，並在 :8000 啟動 FastAPI
ngrok http 8000                  # 將 https://<ngrok-domain>/webhook 填到 LINE 後台
```

- 健康檢查：`GET /health`
- macOS demo 可以直接用 [start_demo.sh](start_demo.sh) 一鍵開啟 FastAPI 與 ngrok 兩個 Terminal 視窗。

### 不透過 LINE、直接測試單支影片

```bash
python graph.py "https://www.youtube.com/shorts/<id>"
```

終端機會印出每個節點的決策過程（下載、是否跳過 Audio、reflection、各項評分）。

### 測試

```bash
pytest
```

---

## 專案結構

```
main.py              進入點：init_db + 啟動 uvicorn
line_bot.py          LINE Webhook、問卷、回饋、群組模式
graph.py             LangGraph 組裝、run_pipeline / run_pipeline_multi
models.py            AgentState 定義
agents/              Metadata / Vision / Audio / Scoring / Preference Agents
acoustics.py         音高與音量起伏量測（用於判斷 TTS）
downloader.py        yt-dlp 下載、ffmpeg 抽幀與抽音訊、signal 快取
database.py          SQLite：taste_profiles / analysis_history / blacklist
config.py            設定與環境變數
tests/               pytest 測試
docs/superpowers/    設計規格與實作計畫
data/                SQLite 資料庫（data/skipit.db，不進 git）
tmp/                 下載的影片、截圖、signal 快取（不進 git）
```

### 快取

Metadata、Vision、Audio 的訊號與觀看者無關，因此會以影片為單位快取在 `tmp/`。同一支影片重複分析時不會再呼叫 Whisper 或 GPT-4o，只有 Scoring 會依每個人的品味檔案重新計算。

---

## 技術棧

FastAPI · line-bot-sdk v3 · LangGraph · OpenAI GPT-4o / Whisper · yt-dlp · ffmpeg · librosa · Pillow / imagehash · SQLite

> LLM 目前使用 OpenAI。架構（AgentState、graph 節點、taste_profile）已預留之後換成 Anthropic Claude 的空間，屆時只需替換各 agent 內的 client 與 prompt 格式。


詳細設計請見 [docs/superpowers/specs/2026-07-14-skipit-bot-v2-design.md](docs/superpowers/specs/2026-07-14-skipit-bot-v2-design.md)。
