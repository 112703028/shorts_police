import subprocess
import hashlib
from pathlib import Path
import yt_dlp
from config import TMP_DIR, FRAME_COUNT


def _video_id(url: str) -> str:
    # 用 URL 的 MD5 前 12 碼當檔名，同一支影片重複分析時可以直接用快取，不用重下載
    return hashlib.md5(url.encode()).hexdigest()[:12]


def download_video(url: str) -> Path:
    out_dir = Path(TMP_DIR)
    out_dir.mkdir(exist_ok=True)
    vid_id = _video_id(url)
    out_path = out_dir / f"{vid_id}.mp4"
    if out_path.exists():
        # 已經下載過，直接回傳快取的檔案
        return out_path
    opts = {
        # 優先抓 mp4 格式的影音軌並合併，抓不到才 fallback 到任何可用格式
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out_path


def extract_frames(video_path: Path, n: int = FRAME_COUNT) -> list[Path]:
    out_dir = video_path.parent / f"{video_path.stem}_frames"
    out_dir.mkdir(exist_ok=True)
    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) == n:
        # 之前已經截過同樣數量的幀，直接重用
        return frames
    # 用 ffprobe 讀取影片總長度（秒）
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip())
    # 把影片均勻切成 n+1 段，取 n 個切點截圖，避免只截到開頭或結尾
    interval = duration / (n + 1)
    frame_paths = []
    for i in range(1, n + 1):
        ts = interval * i
        out_file = out_dir / f"frame_{i:02d}.jpg"
        subprocess.run([
            "ffmpeg", "-ss", str(ts), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(out_file), "-y", "-loglevel", "error"
        ], check=True)
        frame_paths.append(out_file)
    return frame_paths


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
    # 只查詢 audio stream，有查到就代表影片有語音（給 Orchestrator 判斷要不要跳過 Audio Agent）
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True
    )
    return "audio" in result.stdout
