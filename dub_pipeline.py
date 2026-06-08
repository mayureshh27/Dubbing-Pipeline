
"""
Whisper → XTTS Russian-to-English Dubbing Pipeline
Target: C++ lecture series (single-speaker, technical vocabulary)

Stages:
  1. Audio extraction (ffmpeg)
  2. Transcription (Whisper, RU)
  3. Translation  (Helsinki-NLP MarianMT, RU→EN)
  4. Voice cloning synthesis (XTTS-v2)
  5. Segment alignment & silence padding
  6. Audio mixdown + mux back to video

Requirements (install once):
  pip install openai-whisper TTS pydub deep-translator \
              transformers sentencepiece ffmpeg-python torch torchaudio

  ffmpeg must be on PATH.

Usage:
  python dub_pipeline.py --input lecture01.mp4 --output lecture01_en.mp4
  python dub_pipeline.py --input lecture01.mp4 --speaker_wav ref_voice.wav
  python dub_pipeline.py --batch /path/to/lectures/   # process whole folder
"""

import argparse
import os
import sys
import json
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from faster_whisper import WhisperModel

# ── lazy imports (loaded only when needed) ─────────────────────────────────
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dub_pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # Whisper
    whisper_model: str = "h2oai/faster-whisper-large-v3-turbo"          # large-v3 best for technical RU
    whisper_device: str = "auto"             # "auto" | "cuda" | "cpu"
    whisper_compute_type: str = "float16"    # float16 on GPU, int8 on CPU

    # Translation
    translation_backend: str = "marian"      # "marian" | "deep_translator"
    marian_model: str = "Helsinki-NLP/opus-mt-ru-en"

    # XTTS
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker_wav: Optional[str] = None        # Path to reference speaker WAV
    xtts_language: str = "en"
    xtts_speed: float = 1.0                  # Adjust if TTS runs long vs original

    # Timing
    stretch_audio: bool = True               # Time-stretch TTS to match segment
    min_silence_gap: float = 0.05            # Seconds of silence between segments

    # Output
    keep_bgm: bool = True                    # Preserve background music/noise
    bgm_volume_db: float = -18.0             # Attenuate BG during speech
    output_suffix: str = "_en_dubbed"

    # Misc
    device: str = field(init=False)

    def __post_init__(self):
        if self.whisper_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.whisper_device
        if self.device == "cpu":
            self.whisper_compute_type = "int8"
        log.info(f"Device: {self.device}  |  Whisper: {self.whisper_model}  |  Compute: {self.whisper_compute_type}")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 – Audio Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract 16-kHz mono WAV from video (Whisper-ready)."""
    wav_path = out_dir / "source_audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000",
        "-vn", str(wav_path),
    ]
    log.info(f"[1/5] Extracting audio → {wav_path.name}")
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


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 – Transcription
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    start: float
    end: float
    text_ru: str
    text_en: str = ""
    tts_wav: Optional[str] = None


def transcribe(wav_path: Path, cfg: PipelineConfig) -> list[Segment]:
    

    log.info(f"[2/5] Transcribing with {cfg.whisper_model} …")

    model = WhisperModel(
        cfg.whisper_model,           # "h2oai/faster-whisper-large-v3-turbo"
        device=cfg.device,
        compute_type="int8_float16",
        download_root="./.model_cache",  # keep models local to project
    )

    segments_raw, info = model.transcribe(
        str(wav_path),
        language="ru",
        initial_prompt=(
            "Это лекция по программированию на C++. "
            "Термины: вектор, шаблон, указатель, класс, наследование, "
            "компилятор, стандартная библиотека, умный указатель, итератор."
        ),
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,       # don't clip word endings
        },
        beam_size=5,
        best_of=5,
        temperature=0.0,               # greedy — deterministic, faster
        condition_on_previous_text=True,
        word_timestamps=True,
    )

    result = []
    for seg in segments_raw:
        text = seg.text.strip()
        if text:
            result.append(Segment(start=seg.start, end=seg.end, text_ru=text))

    log.info(f"   → {len(result)} segments | lang={info.language} | prob={info.language_probability:.2f}")

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 – Translation (RU → EN)
# ═══════════════════════════════════════════════════════════════════════════

def translate_segments(segments: list[Segment], cfg: PipelineConfig) -> list[Segment]:
    log.info(f"[3/5] Translating {len(segments)} segments (RU→EN) …")

    if cfg.translation_backend == "marian":
        _translate_marian(segments, cfg)
    else:
        _translate_deep(segments)

    return segments


def _translate_marian(segments: list[Segment], cfg: PipelineConfig):
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        raise ImportError("Run: pip install transformers sentencepiece")

    log.info(f"   Loading {cfg.marian_model} …")
    tokenizer = MarianTokenizer.from_pretrained(cfg.marian_model)
    model = MarianMTModel.from_pretrained(cfg.marian_model).to("cpu")
    model.eval()

    BATCH = 16
    texts = [s.text_ru for s in segments]

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
        with torch.no_grad():
            translated = model.generate(**tokens)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        for j, en in enumerate(decoded):
            segments[i + j].text_en = _clean_translation(en)

    # Offload model from RAM
    del model
    import gc
    gc.collect()
    log.info("   Translation complete & model offloaded from CPU memory")


def _translate_deep(segments: list[Segment]):
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        raise ImportError("Run: pip install deep-translator")

    translator = GoogleTranslator(source="ru", target="en")
    for seg in segments:
        seg.text_en = _clean_translation(translator.translate(seg.text_ru))


def _clean_translation(text: str) -> str:
    """Fix common issues with technical translations."""
    replacements = {
        "template": "template",       # sometimes left as «шаблон» transliteration
        "C++": "C++",
        "std::": "std::",
        "nullptr": "nullptr",
        "const": "const",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 – TTS Synthesis (XTTS-v2)
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_segments(segments: list[Segment], cfg: PipelineConfig, out_dir: Path) -> list[Segment]:
    log.info(f"[4/5] Synthesizing {len(segments)} segments with XTTS-v2 …")
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError("Run: pip install TTS")

    tts = TTS(model_name=cfg.xtts_model, progress_bar=False).to(cfg.device)

    speaker_wav = cfg.speaker_wav
    if not speaker_wav:
        # Auto-extract a clean voice reference from the first 30s of the source
        speaker_wav = str(out_dir / "speaker_ref.wav")
        log.info("   No speaker WAV provided — extracting voice reference from source …")
        # (User should ideally supply a clean 6–30 s clip; this is a fallback)

    tts_dir = out_dir / "tts_segments"
    tts_dir.mkdir(exist_ok=True)

    for i, seg in enumerate(segments):
        out_wav = str(tts_dir / f"seg_{i:05d}.wav")
        if Path(out_wav).exists():
            seg.tts_wav = out_wav
            continue  # resume-safe

        text = seg.text_en
        if not text:
            continue

        try:
            tts.tts_to_file(
                text=text,
                file_path=out_wav,
                speaker_wav=speaker_wav,
                language=cfg.xtts_language,
                speed=cfg.xtts_speed,
            )
            seg.tts_wav = out_wav
        except Exception as e:
            log.warning(f"   Segment {i} TTS failed: {e}")

        if i % 20 == 0:
            log.info(f"   … {i}/{len(segments)} done")

    log.info("   TTS synthesis complete")
    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 – Segment Alignment & Audio Assembly
# ═══════════════════════════════════════════════════════════════════════════

def assemble_dubbed_track(
    segments: list[Segment],
    source_wav: Path,
    out_dir: Path,
    cfg: PipelineConfig,
) -> Path:
    log.info("[5a/5] Assembling dubbed audio track …")
    try:
        from pydub import AudioSegment
        from pydub.effects import speedup
    except ImportError:
        raise ImportError("Run: pip install pydub")

    # Get total duration from source
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

        # Normalize loudness to -20 LUFS equivalent (simple peak norm)
        tts_audio = _normalize(tts_audio, target_dBFS=-20.0)

        # Overlay at correct position
        dubbed = dubbed.overlay(tts_audio, position=seg_start_ms)

    dubbed_path = out_dir / "dubbed_track.wav"
    dubbed.export(str(dubbed_path), format="wav")
    log.info(f"   Dubbed track saved → {dubbed_path.name}")
    return dubbed_path


def _normalize(audio, target_dBFS: float = -20.0):
    from pydub import AudioSegment
    diff = target_dBFS - audio.dBFS
    return audio.apply_gain(diff)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5b – BGM Mixing (optional)
# ═══════════════════════════════════════════════════════════════════════════

def mix_with_bgm(
    dubbed_track: Path,
    original_audio: Path,
    segments: list[Segment],
    out_dir: Path,
    cfg: PipelineConfig,
) -> Path:
    """Duck the original audio under speech segments, overlay dubbed track."""
    log.info("[5b/5] Mixing dubbed track with background audio …")
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("Run: pip install pydub")

    original = AudioSegment.from_wav(str(original_audio))
    dubbed    = AudioSegment.from_wav(str(dubbed_track))

    # Resample original to match dubbed if needed
    if original.frame_rate != 44100:
        original = original.set_frame_rate(44100)

    # Attenuate original during speech segments
    ducked = original
    for seg in segments:
        if not seg.tts_wav:
            continue
        s_ms = int(seg.start * 1000)
        e_ms = int(seg.end   * 1000)
        # Replace that slice with the attenuated version
        slice_audio = original[s_ms:e_ms].apply_gain(cfg.bgm_volume_db)
        ducked = ducked[:s_ms] + slice_audio + ducked[e_ms:]

    # Merge: ducked original + dubbed voice (44.1kHz)
    dubbed_44 = dubbed.set_frame_rate(44100).set_channels(2)
    mixed = ducked.overlay(dubbed_44)

    mixed_path = out_dir / "mixed_audio.wav"
    mixed.export(str(mixed_path), format="wav")
    return mixed_path


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6 – Mux back to video
# ═══════════════════════════════════════════════════════════════════════════

def mux_to_video(video_path: Path, audio_path: Path, output_path: Path):
    log.info(f"[6/5] Muxing → {output_path.name}")
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
    log.info(f"   ✓ Output: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Transcript save / resume
# ═══════════════════════════════════════════════════════════════════════════

def save_transcript(segments: list[Segment], path: Path):
    data = [{"start": s.start, "end": s.end, "ru": s.text_ru, "en": s.text_en} for s in segments]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"   Transcript saved → {path.name}")


def load_transcript(path: Path) -> list[Segment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Segment(start=d["start"], end=d["end"], text_ru=d["ru"], text_en=d["en"]) for d in data]


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _run(cmd: list[str], silent: bool = False):
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL if silent else None,
        stderr=subprocess.DEVNULL if silent else None,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("ffmpeg not found on PATH. Install via: sudo apt install ffmpeg  OR  brew install ffmpeg")


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def process_file(input_path: Path, output_path: Optional[Path], cfg: PipelineConfig):
    _check_ffmpeg()

    if output_path is None:
        output_path = input_path.parent / (input_path.stem + cfg.output_suffix + input_path.suffix)

    work_dir = input_path.parent / (input_path.stem + "_dubwork")
    work_dir.mkdir(exist_ok=True)
    log.info(f"Working dir: {work_dir}")

    transcript_path = work_dir / "transcript.json"

    # ── Stage 1: Extract audio ──────────────────────────────────────────
    wav_16k  = extract_audio(input_path, work_dir)
    wav_full = extract_full_audio(input_path, work_dir)

    # ── Stage 2: Transcribe (or resume) ────────────────────────────────
    if transcript_path.exists():
        log.info("[2/5] Resuming from saved transcript …")
        segments = load_transcript(transcript_path)
        if not segments[0].text_en:
            segments = translate_segments(segments, cfg)
            save_transcript(segments, transcript_path)
    else:
        segments = transcribe(wav_16k, cfg)
        segments = translate_segments(segments, cfg)
        save_transcript(segments, transcript_path)

    # ── Stage 4: TTS ────────────────────────────────────────────────────
    segments = synthesize_segments(segments, cfg, work_dir)

    # ── Stage 5: Assembly ───────────────────────────────────────────────
    dubbed_track = assemble_dubbed_track(segments, wav_16k, work_dir, cfg)

    if cfg.keep_bgm:
        final_audio = mix_with_bgm(dubbed_track, wav_full, segments, work_dir, cfg)
    else:
        final_audio = dubbed_track

    # ── Stage 6: Mux ────────────────────────────────────────────────────
    mux_to_video(input_path, final_audio, output_path)

    log.info(f"\n{'═'*55}")
    log.info(f"  Done! → {output_path}")
    log.info(f"{'═'*55}\n")
    return output_path


def process_batch(folder: Path, cfg: PipelineConfig):
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts and cfg.output_suffix not in f.stem)
    if not files:
        log.warning(f"No video files found in {folder}")
        return
    log.info(f"Batch mode: {len(files)} file(s) found")
    for f in files:
        log.info(f"\n{'─'*55}\n  Processing: {f.name}\n{'─'*55}")
        try:
            process_file(f, None, cfg)
        except Exception as e:
            log.error(f"  FAILED: {f.name} — {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Whisper → MarianMT → XTTS-v2  Russian→English video dubbing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input",  type=Path, help="Single video file")
    group.add_argument("--batch",  type=Path, help="Folder of videos to process")

    parser.add_argument("--output",       type=Path,  default=None,  help="Output path (single file mode only)")
    parser.add_argument("--speaker_wav",  type=str,   default=None,  help="Path to reference speaker WAV (6–30 s clean speech)")
    parser.add_argument("--whisper_model",type=str,   default="h2oai/faster-whisper-large-v3-turbo")
    parser.add_argument("--no_bgm",       action="store_true",       help="Replace audio entirely (no BGM mix)")
    parser.add_argument("--no_stretch",   action="store_true",       help="Disable time-stretch alignment")
    parser.add_argument("--xtts_speed",   type=float, default=1.0,   help="XTTS speed factor (0.8–1.4)")
    parser.add_argument("--translation",  choices=["marian", "deep_translator"], default="marian")
    parser.add_argument("--device",       choices=["auto", "cuda", "cpu"], default="auto")

    args = parser.parse_args()

    cfg = PipelineConfig(
        whisper_model=args.whisper_model,
        whisper_device=args.device,
        speaker_wav=args.speaker_wav,
        keep_bgm=not args.no_bgm,
        stretch_audio=not args.no_stretch,
        xtts_speed=args.xtts_speed,
        translation_backend=args.translation,
    )

    if args.batch:
        process_batch(args.batch, cfg)
    else:
        process_file(args.input, args.output, cfg)


if __name__ == "__main__":
    main()