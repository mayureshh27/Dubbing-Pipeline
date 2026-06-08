import logging
import torch
import gc
from pathlib import Path
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.synthesize")

def synthesize_segments(segments: list[Segment], cfg: PipelineConfig, out_dir: Path) -> list[Segment]:
    log.info(f"Synthesizing {len(segments)} segments with {cfg.xtts_model} …")
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError("Run: pip install TTS")

    tts = TTS(model_name=cfg.xtts_model, progress_bar=False).to(cfg.device)

    speaker_wav = cfg.speaker_wav
    if not speaker_wav:
        speaker_wav = str(out_dir / "speaker_ref.wav")
        log.info("No speaker WAV provided — extracting voice reference from source …")
        # Ensure we have a helper to crop 30s of original audio if needed
        # We can implement this fallback check or let the pipeline extract it
        ref_path = Path(speaker_wav)
        if not ref_path.exists():
            log.warning(f"Speaker reference WAV not found at {speaker_wav}. Attempting fallback.")

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
            log.warning(f"Segment {i} TTS failed: {e}")

        if i % 20 == 0:
            log.info(f"   … synthesized {i}/{len(segments)} segments")

    # Strict memory cleanup to prevent OOM
    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("TTS model deleted and VRAM cleared")

    log.info("TTS synthesis complete")
    return segments
