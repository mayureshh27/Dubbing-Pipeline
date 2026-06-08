import subprocess
import shutil
import logging
import json
import re
from pathlib import Path
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.audio")

def _run(cmd: list[str], silent: bool = False) -> str:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nError: {result.stderr}")
    return result.stdout or result.stderr

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
    _run(cmd, silent=True)
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

def two_pass_loudnorm(input_wav: Path, output_wav: Path):
    """Applies high-quality two-pass loudness normalization to target -16 LUFS."""
    log.info(f"Applying two-pass loudnorm to {input_wav.name} …")
    
    # Pass 1: Measure
    cmd_measure = [
        "ffmpeg", "-i", str(input_wav),
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-"
    ]
    try:
        output = _run(cmd_measure, silent=True)
        # Parse JSON from output
        json_match = re.search(r"\{[\s\S]*\}", output)
        if not json_match:
            raise ValueError("No JSON found in loudnorm pass 1 output.")
        
        stats = json.loads(json_match.group(0))
        
        # Pass 2: Apply
        cmd_apply = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-af", (
                f"loudnorm=linear=true:"
                f"I_measured={stats['input_i']}:"
                f"LRA_measured={stats['input_lra']}:"
                f"TP_measured={stats['input_tp']}:"
                f"threshold_measured={stats['input_thresh']}:"
                f"offset_measured={stats['target_offset']}:"
                f"I=-16:LRA=11:TP=-1.5"
            ),
            "-ar", "44100",
            str(output_wav)
        ]
        _run(cmd_apply, silent=True)
    except Exception as e:
        log.warning(f"Loudnorm failed: {e}. Falling back to simple copy.")
        shutil.copy(input_wav, output_wav)

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
        
        # Time-stretch alignment
        if cfg.stretch_audio and len(tts_audio) > 0:
            ratio = len(tts_audio) / max(slot_ms, 1)
            if ratio > 1.10:
                speed_factor = min(ratio, 2.0)
                tts_audio = speedup(tts_audio, playback_speed=speed_factor, chunk_size=50)
            elif ratio < 0.90:
                pad_ms = slot_ms - len(tts_audio)
                tts_audio = tts_audio + AudioSegment.silent(duration=pad_ms)

        # Overlay at correct position
        dubbed = dubbed.overlay(tts_audio, position=seg_start_ms)

    raw_track = out_dir / "dubbed_track_raw.wav"
    dubbed.export(str(raw_track), format="wav")
    
    # Run two-pass normalization for high quality mastering
    dubbed_path = out_dir / "dubbed_track.wav"
    two_pass_loudnorm(raw_track, dubbed_path)
    
    log.info(f"Dubbed track mastered & saved → {dubbed_path.name}")
    return dubbed_path

def mix_with_bgm(
    dubbed_track: Path,
    original_audio: Path,
    segments: list[Segment],
    out_dir: Path,
    cfg: PipelineConfig,
) -> Path:
    log.info("Mixing dubbed track using dynamic sidechain compression ducking …")
    mixed_path = out_dir / "mixed_audio.wav"
    
    # dynamic ducking: use sidechaincompress filter
    # dubbed_track is input 1 (compressor sidechain key), original_audio is input 0 (ambient background track)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_audio),
        "-i", str(dubbed_track),
        "-filter_complex", "[0:a][1:a]sidechaincompress=threshold=0.10:ratio=5:attack=15:release=250[mixed]",
        "-map", "[mixed]",
        str(mixed_path)
    ]
    try:
        _run(cmd, silent=True)
    except Exception as e:
        log.warning(f"FFmpeg sidechain mix failed: {e}. Falling back to simple overlay.")
        cmd_fallback = [
            "ffmpeg", "-y",
            "-i", str(original_audio),
            "-i", str(dubbed_track),
            "-filter_complex", "amix=inputs=2:duration=first",
            str(mixed_path)
        ]
        _run(cmd_fallback, silent=True)
        
    return mixed_path

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(segments: list[Segment], srt_path: Path, lang: str):
    lines = []
    for idx, seg in enumerate(segments):
        start_t = format_timestamp(seg.start)
        end_t = format_timestamp(seg.end)
        text = seg.text_refined or seg.text_en if lang == "en" else seg.text_ru
        lines.append(f"{idx + 1}")
        lines.append(f"{start_t} --> {end_t}")
        lines.append(text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Subtitles exported → {srt_path.name}")

def mux_to_video(video_path: Path, audio_path: Path, output_path: Path):
    log.info(f"Muxing final output video → {output_path.name}")
    
    # Generate external SRT tracks
    out_dir = audio_path.parent
    # We load segments from refined artifact to build subtitle files
    from .orchestrator import load_artifact
    segments = load_artifact("refined", out_dir) or []
    
    srt_en = out_dir / "english.srt"
    srt_ru = out_dir / "russian.srt"
    write_srt(segments, srt_en, "en")
    write_srt(segments, srt_ru, "ru")
    
    # Mux Video + Audio + 2 subtitle tracks (embedded mov_text)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-i", str(srt_en),
        "-i", str(srt_ru),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-c:s", "mov_text",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map", "2:s:0",
        "-map", "3:s:0",
        "-metadata:s:s:0", "language=eng",
        "-metadata:s:s:0", "title=English Subtitles",
        "-metadata:s:s:1", "language=rus",
        "-metadata:s:s:1", "title=Russian Subtitles",
        str(output_path),
    ]
    _run(cmd, silent=True)
    
    # Also copy external subtitles to output folder
    shutil.copy(srt_en, output_path.parent / (output_path.stem + ".en.srt"))
    shutil.copy(srt_ru, output_path.parent / (output_path.stem + ".ru.srt"))
