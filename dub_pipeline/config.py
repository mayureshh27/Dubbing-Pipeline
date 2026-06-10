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
    speech_rate: float = 0.0
    pause_density: float = 0.0

    def to_dict(self):
        return {
          "id": self.id,
          "start": self.start,
          "end": self.end,
          "ru": self.text_ru,
          "en": self.text_en,
          "refined": self.text_refined,
          "tts_wav": self.tts_wav,
          "words": self.words,
          "speech_rate": self.speech_rate,
          "pause_density": self.pause_density
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
          words=data.get("words", []),
          speech_rate=data.get("speech_rate", 0.0),
          pause_density=data.get("pause_density", 0.0)
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

    # Profile & Fallbacks
    profile: str = "fast"  # fast, high_quality, educational, authentic
    qwen_model_path: str = "models/qwen/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    translategemma_model_path: str = "models/translategemma/google.translategemma-4b-it.Q4_K_M.gguf"

    # Misc
    device: str = field(init=False)

    def __post_init__(self):
        if self.whisper_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.whisper_device
        if self.device == "cpu":
            self.whisper_compute_type = "int8"

        # Profile defaults mapping
        if self.profile == "fast":
            self.tts_provider = "kokoro"
        elif self.profile == "educational":
            self.tts_provider = "kokoro"
        elif self.profile == "authentic":
            self.tts_provider = "f5"
        elif self.profile == "high_quality":
            self.tts_provider = "xtts"
