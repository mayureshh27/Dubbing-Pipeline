import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import torch

@dataclass
class Segment:
    id: str  # Deterministic hash of time/text
    start: float
    end: float
    text_ru: str
    text_en: str = ""
    text_refined: str = ""
    tts_wav: Optional[str] = None
    words: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
          "id": self.id,
          "start": self.start,
          "end": self.end,
          "ru": self.text_ru,
          "en": self.text_en,
          "refined": self.text_refined,
          "tts_wav": self.tts_wav,
          "words": self.words
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
          id=data["id"],
          start=data["start"],
          end=data["end"],
          text_ru=data["ru"],
          text_en=data.get("en", ""),
          text_refined=data.get("refined", ""),
          tts_wav=data.get("tts_wav"),
          words=data.get("words", [])
        )

@dataclass
class PipelineConfig:
    # Whisper
    whisper_model: str = "h2oai/faster-whisper-large-v3-turbo"
    whisper_device: str = "auto"
    whisper_compute_type: str = "float16"

    # Translation
    translation_backend: str = "marian"
    marian_model: str = "Helsinki-NLP/opus-mt-ru-en"

    # XTTS / TTS Registry
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_provider: str = "xtts"
    speaker_wav: Optional[str] = None
    xtts_language: str = "en"
    xtts_speed: float = 1.0

    # Timing
    stretch_audio: bool = True
    min_silence_gap: float = 0.05

    # Output
    keep_bgm: bool = True
    bgm_volume_db: float = -18.0
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
