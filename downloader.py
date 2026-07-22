import subprocess
import hashlib
from pathlib import Path
import yt_dlp
from config import TMP_DIR


def _video_id(url: str) -> str:
    # 用 URL 的 MD5 前 12 碼當檔名，同一支影片重複分析時可以直接用快取，不用重下載
    return hashlib.md5(url.encode()).hexdigest()[:12]


def download_video(url: str) -> Path:
    out_dir = Path(TMP_DIR)
    out_dir.mkdir(exist_ok=True)
    vid_id = _video_id(url)
    out_path = out_dir / f"{vid_id}.mp4"
    if out_path.exists():
        return out_path
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "noprogress": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out_path


def get_thumbnail_url(url: str) -> str | None:
    # 抓 YouTube 官方縮圖網址（不下載影片本體），給 vision_agent 比對「縮圖與內容不符」用
    opts = {"skip_download": True, "quiet": True}
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
    # 只查詢 audio stream，有查到就代表影片有語音（給 Orchestrator 判斷要不要跳過 Audio Agent）
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True
    )
    return "audio" in result.stdout
