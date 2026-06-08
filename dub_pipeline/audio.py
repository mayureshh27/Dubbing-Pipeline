import subprocess
import shutil
import logging
from pathlib import Path
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.audio")

def _run(cmd: list[str], silent: bool = False):
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL if silent else None,
        stderr=subprocess.DEVNULL if silent else None,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("ffmpeg not found on PATH. Please install ffmpeg.")

def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract 16-kHz mono WAV from video (Whisper-ready)."""
    wav_path = out_dir / "source_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        "-vn", str(wav_path),
    ]
    log.info(f"Extracting mono 16kHz audio → {wav_path.name}")
    _run(cmd)
    return wav_path

def extract_full_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract full-quality stereo audio for mixdown later."""
    wav_path = out_dir / "source_audio_full.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "2", "-ar", "44100",
        "-vn", str(wav_path),
    ]
    _run(cmd, silent=True)
    return wav_path

def _normalize(audio, target_dBFS: float = -20.0):
    diff = target_dBFS - audio.dBFS
    return audio.apply_gain(diff)

def assemble_dubbed_track(
    segments: list[Segment],
    source_wav: Path,
    out_dir: Path,
    cfg: PipelineConfig,
) -> Path:
    log.info("Assembling dubbed audio track …")
    try:
        from pydub import AudioSegment
        from pydub.effects import speedup
    except ImportError:
        raise ImportError("Run: pip install pydub")

    source = AudioSegment.from_wav(str(source_wav))
    total_ms = len(source)
    dubbed = AudioSegment.silent(duration=total_ms)

    for i, seg in enumerate(segments):
        if not seg.tts_wav or not Path(seg.tts_wav).exists():
            continue

        tts_audio = AudioSegment.from_wav(seg.tts_wav)
        seg_start_ms = int(seg.start * 1000)
        seg_end_ms = int(seg.end * 1000)
        slot_ms = seg_end_ms - seg_start_ms

        # Time-stretch TTS to fit original segment duration
        if cfg.stretch_audio and len(tts_audio) > 0:
            ratio = len(tts_audio) / max(slot_ms, 1)
            if ratio > 1.15:
                # TTS is longer than slot → speed it up (max 2x)
                speed_factor = min(ratio, 2.0)
                tts_audio = speedup(tts_audio, playback_speed=speed_factor, chunk_size=50)
            elif ratio < 0.85:
                # TTS is shorter → pad with silence at end
                pad_ms = slot_ms - len(tts_audio)
                tts_audio = tts_audio + AudioSegment.silent(duration=pad_ms)

        # Normalize loudness
        tts_audio = _normalize(tts_audio, target_dBFS=-20.0)

        # Overlay at correct position
        dubbed = dubbed.overlay(tts_audio, position=seg_start_ms)

    dubbed_path = out_dir / "dubbed_track.wav"
    dubbed.export(str(dubbed_path), format="wav")
    log.info(f"Dubbed track saved → {dubbed_path.name}")
    return dubbed_path

def mix_with_bgm(
    dubbed_track: Path,
    original_audio: Path,
    segments: list[Segment],
    out_dir: Path,
    cfg: PipelineConfig,
) -> Path:
    log.info("Mixing dubbed track with background audio …")
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("Run: pip install pydub")

    original = AudioSegment.from_wav(str(original_audio))
    dubbed = AudioSegment.from_wav(str(dubbed_track))

    if original.frame_rate != 44100:
        original = original.set_frame_rate(44100)

    ducked = original
    for seg in segments:
        if not seg.tts_wav:
            continue
        s_ms = int(seg.start * 1000)
        e_ms = int(seg.end * 1000)
        slice_audio = original[s_ms:e_ms].apply_gain(cfg.bgm_volume_db)
        ducked = ducked[:s_ms] + slice_audio + ducked[e_ms:]

    dubbed_44 = dubbed.set_frame_rate(44100).set_channels(2)
    mixed = ducked.overlay(dubbed_44)

    mixed_path = out_dir / "mixed_audio.wav"
    mixed.export(str(mixed_path), format="wav")
    return mixed_path

def mux_to_video(video_path: Path, audio_path: Path, output_path: Path):
    log.info(f"Muxing final output video → {output_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        str(output_path),
    ]
    _run(cmd)
