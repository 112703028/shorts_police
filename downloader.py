import subprocess
import hashlib
import json
import re
from pathlib import Path
import yt_dlp
from config import TMP_DIR, YT_DLP_COOKIES_FILE, YT_DLP_COOKIES_BROWSER, DISABLE_SIGNAL_CACHE


def _video_id(url: str) -> str:
    # 用 URL 的 MD5 前 12 碼當檔名，同一支影片重複分析時可以直接用快取，不用重下載
    return hashlib.md5(url.encode()).hexdigest()[:12]


def base_ydl_opts(extra: dict) -> dict:
    """所有建立 yt_dlp.YoutubeDL 的地方都該用這個，統一帶上登入用的 cookies（如果有設）。
    YouTube 常對匿名抓取要求「請登入驗證」，帶 cookies 讓 yt-dlp 假裝是已登入的瀏覽器在抓，
    降低被擋機率。兩個環境變數都沒設就維持原本匿名抓取（跟改之前行為一致）。"""
    opts = dict(extra)
    if YT_DLP_COOKIES_FILE:
        opts["cookiefile"] = YT_DLP_COOKIES_FILE
    elif YT_DLP_COOKIES_BROWSER:
        opts["cookiesfrombrowser"] = (YT_DLP_COOKIES_BROWSER,)
    return opts


def load_signal_cache(url: str, agent: str) -> dict | None:
    """agent 在打 LLM 前先查有沒有分析過同一支影片（跟看的人是誰無關），
    有就直接回傳上次的結果，不重打 GPT-4o/Whisper API（省錢也省時間）。
    DISABLE_SIGNAL_CACHE=true 時直接跳過讀取（還是會寫入），方便密集調 prompt 時強制每次重跑，
    不用手動一個個刪 tmp/*.json。"""
    if DISABLE_SIGNAL_CACHE:
        return None
    path = Path(TMP_DIR) / f"{_video_id(url)}_{agent}.json"
    if path.exists():
        # 印出來才看得到「省成本」這件事真的發生了，不然跟真的打一次 API 在 log 上長得一樣
        print(f"⚡ [Cache] {agent} 快取命中，跳過 API 呼叫", flush=True)
        return json.loads(path.read_text())
    return None


def save_signal_cache(url: str, agent: str, data: dict) -> None:
    path = Path(TMP_DIR) / f"{_video_id(url)}_{agent}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def download_video(url: str) -> Path:
    out_dir = Path(TMP_DIR)
    out_dir.mkdir(exist_ok=True)
    vid_id = _video_id(url)
    out_path = out_dir / f"{vid_id}.mp4"
    if out_path.exists():
        return out_path
    opts = base_ydl_opts({
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "noprogress": True,
        "merge_output_format": "mp4",
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out_path


def get_thumbnail_url(url: str) -> str | None:
    # 抓 YouTube 官方縮圖網址（不下載影片本體），給 vision_agent 比對「縮圖與內容不符」用
    opts = base_ydl_opts({"skip_download": True, "quiet": True})
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get("thumbnail")


def extract_frames(video_path: Path) -> list[Path]:
    # 每秒抽一幀（不是固定張數），密度隨影片長度變動
    out_dir = video_path.parent / f"{video_path.stem}_frames"
    out_dir.mkdir(exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if existing:
        return existing
    subprocess.run([
        "ffmpeg", "-i", str(video_path), "-vf", "fps=1",
        str(out_dir / "frame_%03d.jpg"), "-y", "-loglevel", "error"
    ], check=True)
    return sorted(out_dir.glob("frame_*.jpg"))


def extract_audio(video_path: Path) -> Path:
    out_path = video_path.with_suffix(".mp3")
    if out_path.exists():
        return out_path
    # -vn 去掉視訊軌，只留音訊轉成 mp3，給 Whisper 轉錄用
    subprocess.run([
        "ffmpeg", "-i", str(video_path), "-vn",
        "-acodec", "libmp3lame", "-q:a", "4", str(out_path), "-y", "-loglevel", "error"
    ], check=True)
    return out_path


def has_audio_track(video_path: Path) -> bool:
    # 只查詢容器裡有沒有 audio stream（技術層面），不代表音軌裡真的有聲音——
    # 很多影片就算沒錄到聲音，編碼器還是會塞一條靜音音軌進去
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True
    )
    return "audio" in result.stdout


_SILENCE_THRESHOLD_DB = -40  # 有實際語音/音樂的影片實測 max_volume 都在 0dB 附近，靜音的在 -90dB 以下，差距很大


def has_meaningful_audio(video_path: Path) -> bool:
    """has_audio_track 只看容器有沒有音軌；這裡進一步用 volumedetect 判斷音軌是不是幾乎全靜音，
    避免「有音軌但實際上沒聲音」的影片還要多花一次 Whisper 呼叫才發現轉不出東西。"""
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True
    )
    match = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    if not match:
        return True  # 解析不出來就保守當作有聲音，交給 Whisper 那層再判斷一次
    return float(match.group(1)) > _SILENCE_THRESHOLD_DB
