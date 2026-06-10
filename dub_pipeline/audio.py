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

def trim_leading_silence(input_wav: Path, output_wav: Path):
    """Trims leading silence using FFmpeg silenceremove filter at -50dB threshold."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_wav),
        "-af", "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0",
        str(output_wav)
    ]
    try:
        _run(cmd, silent=True)
    except Exception as e:
        log.warning(f"Silence trimming failed for {input_wav.name}: {e}. Falling back to copy.")
        shutil.copy(input_wav, output_wav)

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
        
        # Pass 2: Apply EQ, Compand, and Loudnorm
        filter_str = (
            "highpass=f=80,"
            "equalizer=f=3000:width_type=q:width=1.0:g=1.5,"
            "compand=attacks=0.03:decays=0.3:points=-80/-80|-20/-20|-15/-10|0/-3:soft-shelf=6,"
            f"loudnorm=linear=true:"
            f"I_measured={stats['input_i']}:"
            f"LRA_measured={stats['input_lra']}:"
            f"TP_measured={stats['input_tp']}:"
            f"threshold_measured={stats['input_thresh']}:"
            f"offset_measured={stats['target_offset']}:"
            f"I=-16:LRA=11:TP=-1.5"
        )
        cmd_apply = [
            "ffmpeg", "-y", "-i", str(input_wav),
            "-af", filter_str,
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

    current_playhead_ms = 0

    for i, seg in enumerate(segments):
        if i % 100 == 0 or i == len(segments) - 1:
            log.info(f"   … processed {i}/{len(segments)} segments")
        if not seg.tts_wav or not Path(seg.tts_wav).exists():
            continue

        tts_audio = AudioSegment.from_wav(seg.tts_wav)
        
        # Audio Quality Validation
        dur_sec = len(tts_audio) / 1000.0
        if dur_sec <= 0:
            log.warning(f"Segment {seg.id} generated empty audio. Skipping.")
            continue
        if tts_audio.rms < 100:
            log.warning(f"Segment {seg.id} generated silent audio (RMS {tts_audio.rms} < 100). Skipping.")
            continue
        if tts_audio.max >= 32767:
            log.warning(f"Segment {seg.id} audio is clipping (max amplitude {tts_audio.max} >= 32767).")
        if tts_audio.frame_rate not in [16000, 22050, 24000, 44100, 48000]:
            log.warning(f"Segment {seg.id} has unusual sample rate ({tts_audio.frame_rate} Hz).")

        seg_start_ms = int(seg.start * 1000)
        seg_end_ms = int(seg.end * 1000)
        slot_ms = seg_end_ms - seg_start_ms
        
        # Educational timeline gap compression
        if cfg.tts_provider == "kokoro":
            gap_ms = seg_start_ms - current_playhead_ms
            if gap_ms > 400:
                target_start_ms = current_playhead_ms + 200
            else:
                target_start_ms = max(seg_start_ms, current_playhead_ms)
        else:
            target_start_ms = seg_start_ms

        # Time-stretch alignment & Unsafe Ratio Guardrails
        if cfg.stretch_audio and len(tts_audio) > 0:
            ratio = len(tts_audio) / max(slot_ms, 1)
            
            if ratio > 1.35:
                log.warning(f"Segment {seg.id} has unsafe stretch ratio ({ratio:.2f} > 1.35). Skipping to avoid severe distortion.")
                continue
            elif ratio > 1.2:
                log.info(f"Segment {seg.id} stretch ratio ({ratio:.2f}) requires high speedup. Proceeding with caution.")
                
            if ratio > 1.10:
                speed_factor = min(ratio, 2.0)
                tts_audio = speedup(tts_audio, playback_speed=speed_factor, chunk_size=50)
            elif ratio < 0.90:
                if cfg.tts_provider != "kokoro":
                    pad_ms = slot_ms - len(tts_audio)
                    tts_audio = tts_audio + AudioSegment.silent(duration=pad_ms)

        # Overlay at correct position
        dubbed = dubbed.overlay(tts_audio, position=target_start_ms)
        current_playhead_ms = target_start_ms + len(tts_audio)

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
    
    # dynamic ducking: use sidechaincompress filter, then amix both inputs, and apply a hard limiter
    filter_complex = (
        "[0:a][1:a]sidechaincompress=threshold=0.10:ratio=5:attack=5:release=120[ducked]; "
        "[ducked][1:a]amix=inputs=2:duration=first,alimiter=limit=0.95[mixed]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_audio),
        "-i", str(dubbed_track),
        "-filter_complex", filter_complex,
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
            "-filter_complex", "amix=inputs=2:duration=first,alimiter=limit=0.95[mixed]",
            "-map", "[mixed]",
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
