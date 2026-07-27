"""
聲學特徵分析：從音訊波形算出「語調平不平」的客觀數據，給 Audio Agent 判斷 TTS 用。

分工：ffmpeg 負責把音訊可靠解碼成波形，librosa.pyin 負責準確抓基頻（f0）。
（pyin 靠 numba JIT，process 內第一次呼叫會花幾秒編譯，之後每支影片約 2-3 秒。）

核心兩個特徵，TTS 機器人聲在這兩項都異常平穩：
- 音高變化（以半音計）：自然說話有抑揚頓挫，TTS/朗讀偏平
- 音量起伏（變異係數）：自然說話有輕重，TTS 偏均勻
"""
import subprocess
from pathlib import Path

import numpy as np
import librosa

_SAMPLE_RATE = 16000       # 解碼成 16kHz 單聲道，語音分析足夠
_F0_MIN = 80               # 人聲基頻下限 (Hz)
_F0_MAX = 400              # 人聲基頻上限 (Hz)
_HOP = 512                 # pyin / rms 共用的幀間隔（樣本數），確保兩者幀對得起來
_MIN_VOICED_FRAMES = 15    # 有聲幀太少（音訊過短）就不給結論


def _load_waveform(audio_path: Path) -> np.ndarray:
    """用 ffmpeg 把音訊解成 16kHz 單聲道 float32 波形（-1~1），失敗回空陣列。"""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-i", str(audio_path), "-f", "s16le", "-acodec", "pcm_s16le",
             "-ac", "1", "-ar", str(_SAMPLE_RATE), "-", "-loglevel", "error"],
            capture_output=True, check=True,
        )
    except Exception:
        return np.array([], dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def analyze_prosody(audio_path: Path) -> dict | None:
    """
    回傳聲學特徵 dict，資料不足回 None：
      pitch_semitone_std : 音高變化（半音標準差），越低越平
      energy_cv          : 音量變異係數，越低越均勻
      voiced_ratio       : 有聲幀比例
    """
    wav = _load_waveform(audio_path)
    if wav.size < _SAMPLE_RATE // 2:      # 不到 0.5 秒，沒意義
        return None

    # 1. pyin 抓每幀基頻，voiced_flag 標記哪些幀是有聲的
    f0, voiced_flag, _ = librosa.pyin(
        wav, fmin=_F0_MIN, fmax=_F0_MAX, sr=_SAMPLE_RATE, hop_length=_HOP,
    )
    # 2. 每幀能量（RMS），跟 pyin 用同樣 hop，幀數才對得起來
    rms = librosa.feature.rms(y=wav, hop_length=_HOP)[0]

    n = min(len(f0), len(rms), len(voiced_flag))
    f0, voiced_flag, rms = f0[:n], voiced_flag[:n], rms[:n]

    voiced = voiced_flag & ~np.isnan(f0)
    if int(np.sum(voiced)) < _MIN_VOICED_FRAMES:
        return None

    voiced_f0 = f0[voiced]
    voiced_rms = rms[voiced]

    # 音高轉半音再算標準差（半音是感知上等距的單位，比直接用 Hz 合理）
    semitones = 12.0 * np.log2(voiced_f0 / _F0_MIN)
    pitch_semitone_std = float(np.std(semitones))
    energy_cv = float(np.std(voiced_rms) / np.mean(voiced_rms)) if np.mean(voiced_rms) > 0 else 0.0
    voiced_ratio = float(np.mean(voiced))

    return {
        "pitch_semitone_std": round(pitch_semitone_std, 2),
        "energy_cv": round(energy_cv, 2),
        "voiced_ratio": round(voiced_ratio, 2),
    }


def describe_acoustics(features: dict | None) -> str:
    """把數字轉成給 GPT 看的證據文字，附上自然說話的參考範圍讓模型有基準可比。"""
    if features is None:
        return "（音訊過短或無法解析，無聲學特徵可參考）"
    return (
        f"音高變化：{features['pitch_semitone_std']} 半音"
        f"（自然說話通常 2-4 半音，TTS/朗讀常低於 1.5）\n"
        f"音量起伏：變異係數 {features['energy_cv']}"
        f"（自然說話起伏較大，TTS 偏均勻、通常低於 0.3）\n"
        f"有聲比例：{features['voiced_ratio']}"
    )


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path:
        print("用法: python acoustics.py <音訊檔路徑>")
    else:
        feats = analyze_prosody(path)
        print("特徵:", feats)
        print(describe_acoustics(feats))
