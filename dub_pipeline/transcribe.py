import logging
import hashlib
import re
import torch
import gc
from pathlib import Path
from faster_whisper import WhisperModel
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.transcribe")

def compute_segment_id(start: float, text: str) -> str:
    normalized = re.sub(r'[^\w\s]', '', text.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    key = f"{start:.3f}_{normalized}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

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
            # Map word timestamps if present
            words_list = []
            if seg.words:
                for w in seg.words:
                    words_list.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability
                    })
            # Calculate speech rate and pause density
            duration = seg.end - seg.start
            speech_rate = 0.0
            pause_density = 0.0
            if duration > 0:
                words_count = len(words_list) if words_list else len(text.split())
                speech_rate = (words_count / duration) * 60
                if words_list:
                    gaps = 0.0
                    gaps += max(0.0, words_list[0]["start"] - seg.start)
                    for idx in range(1, len(words_list)):
                        gap = words_list[idx]["start"] - words_list[idx - 1]["end"]
                        if gap > 0:
                            gaps += gap
                    gaps += max(0.0, seg.end - words_list[-1]["end"])
                    pause_density = min(1.0, gaps / duration)

            seg_id = compute_segment_id(seg.start, text)
            result.append(Segment(
                id=seg_id,
                start=seg.start,
                end=seg.end,
                text_ru=text,
                words=words_list,
                speech_rate=speech_rate,
                pause_density=pause_density
            ))

    log.info(f"→ Transcribed {len(result)} segments | language={info.language} (prob={info.language_probability:.2f})")

    # Strict memory cleanup to prevent OOM
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("Whisper model deleted and VRAM cleared")
        
    return result
