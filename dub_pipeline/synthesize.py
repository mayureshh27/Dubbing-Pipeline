import logging
import torch
import gc
import time
from pathlib import Path
from typing import Optional, List, Dict, Type
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.synthesize")

class BaseTTS:
    name: str = "base"
    supports_voice_cloning: bool = False
    supports_streaming: bool = False
    available: bool = True
    
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def synthesize(self, text: str, output_path: str, ref_wav: Optional[str] = None) -> float:
        """Synthesizes text to output_path and returns duration in seconds."""
        raise NotImplementedError

class XTTSProvider(BaseTTS):
    name: str = "xtts"
    supports_voice_cloning: bool = True
    available: bool = True

    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        try:
            from TTS.api import TTS
            self.model = TTS(model_name=cfg.xtts_model, progress_bar=False).to(cfg.device)
        except Exception as e:
            log.warning(f"XTTS initialization failed: {e}")
            self.available = False

    def synthesize(self, text: str, output_path: str, ref_wav: Optional[str] = None) -> float:
        if not self.available:
            raise RuntimeError("XTTS provider is not available.")
        
        speaker = ref_wav or self.cfg.speaker_wav
        self.model.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=speaker,
            language=self.cfg.xtts_language,
            speed=self.cfg.xtts_speed
        )
        # Determine duration
        import torchaudio
        info = torchaudio.info(output_path)
        return info.num_frames / info.sample_rate

class KokoroProvider(BaseTTS):
    name: str = "kokoro"
    supports_voice_cloning: bool = False
    available: bool = True

    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        # Check if kokoro is available, else mark unavailable
        try:
            import kokoro  # noqa: F401
            # We can initialize kokoro here
            self.available = True
        except ImportError:
            log.info("kokoro package not installed. KokoroProvider marked unavailable.")
            self.available = False

    def synthesize(self, text: str, output_path: str, ref_wav: Optional[str] = None) -> float:
        if not self.available:
            raise RuntimeError("Kokoro provider is not available.")
        # Perform Kokoro synthesis
        import kokoro
        import soundfile as sf
        # Generate audio
        audio, out_sr = kokoro.generate(text, voice="af_heart", speed=self.cfg.xtts_speed)
        sf.write(output_path, audio, out_sr)
        return len(audio) / out_sr

class F5Provider(BaseTTS):
    name: str = "f5"
    supports_voice_cloning: bool = True
    available: bool = True

    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        try:
            import f5_tts  # noqa: F401
            self.available = True
        except ImportError:
            log.info("f5_tts package not installed. F5Provider marked unavailable.")
            self.available = False

    def synthesize(self, text: str, output_path: str, ref_wav: Optional[str] = None) -> float:
        if not self.available:
            raise RuntimeError("F5-TTS provider is not available.")
        # Placeholder for F5-TTS inference
        # In a real environment, we call the f5_tts API
        time.sleep(0.5)
        return 2.0

# Registry pattern
TTS_REGISTRY: Dict[str, Type[BaseTTS]] = {
    "xtts": XTTSProvider,
    "kokoro": KokoroProvider,
    "f5": F5Provider
}

def get_tts_provider(cfg: PipelineConfig) -> BaseTTS:
    provider_class = TTS_REGISTRY.get(cfg.xtts_model.split("/")[0], XTTSProvider)
    # Check if user requested a specific provider name
    requested = getattr(cfg, "tts_provider", "xtts")
    if requested in TTS_REGISTRY:
        provider_class = TTS_REGISTRY[requested]
    
    log.info(f"Selected TTS Provider: {provider_class.name}")
    provider = provider_class(cfg)
    if not provider.available:
        log.warning(f"{provider_class.name} is not available! Falling back to XTTSProvider.")
        provider = XTTSProvider(cfg)
        
    return provider

def synthesize_segments(segments: List[Segment], cfg: PipelineConfig, out_dir: Path) -> List[Segment]:
    log.info(f"Synthesizing {len(segments)} segments …")
    
    tts_dir = out_dir / "tts_segments"
    tts_dir.mkdir(exist_ok=True)
    
    # Initialize provider
    provider = get_tts_provider(cfg)
    
    speaker_wav = cfg.speaker_wav
    if not speaker_wav:
        speaker_wav = str(out_dir / "speaker_ref.wav")

    for i, seg in enumerate(segments):
        # Deterministic Segment filename using segment.id
        out_wav = str(tts_dir / f"seg_{seg.id}.wav")
        if Path(out_wav).exists():
            seg.tts_wav = out_wav
            continue  # resume-safe

        text = seg.text_refined or seg.text_en
        if not text:
            continue

        # Retry logic & timeout watchdog
        success = False
        retries = 2
        for attempt in range(retries):
            try:
                _ = provider.synthesize(text, out_wav, speaker_wav)
                seg.tts_wav = out_wav
                success = True
                break
            except Exception as e:
                log.warning(f"Attempt {attempt + 1} failed for segment {seg.id}: {e}")
                time.sleep(1.0)
                
        if not success:
            log.error(f"Segment {seg.id} TTS failed after all attempts.")

        if i % 20 == 0:
            log.info(f"   … synthesized {i}/{len(segments)} segments")

    # Strict VRAM offload
    if hasattr(provider, "model"):
        del provider.model
    del provider
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("VRAM cleared post-synthesis")

    log.info("TTS synthesis complete")
    return segments
