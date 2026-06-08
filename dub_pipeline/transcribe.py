import logging
import torch
import gc
from pathlib import Path
from faster_whisper import WhisperModel
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.transcribe")

def transcribe(wav_path: Path, cfg: PipelineConfig) -> list[Segment]:
    log.info(f"Transcribing with {cfg.whisper_model} …")

    model = WhisperModel(
        cfg.whisper_model,
        device=cfg.device,
        compute_type="int8_float16" if cfg.device == "cuda" else "int8",
        download_root="./.model_cache",
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
            "speech_pad_ms": 200,
        },
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=True,
        word_timestamps=True,
    )

    result = []
    for seg in segments_raw:
        text = seg.text.strip()
        if text:
            result.append(Segment(start=seg.start, end=seg.end, text_ru=text))

    log.info(f"→ Transcribed {len(result)} segments | language={info.language} (prob={info.language_probability:.2f})")

    # Strict memory cleanup to prevent OOM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("Whisper model deleted and VRAM cleared")
        
    return result
