import logging
import torch
import gc
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
        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code='a', device=cfg.device)
            self.available = True
        except Exception as e:
            log.warning(f"Kokoro initialization failed: {e}")
            self.available = False

    def synthesize(self, text: str, output_path: str, ref_wav: Optional[str] = None) -> float:
        if not self.available:
            raise RuntimeError("Kokoro provider is not available.")
        import soundfile as sf
        import numpy as np
        generator = self.pipeline(text, voice="af_heart", speed=self.cfg.xtts_speed, split_pattern=None)
        audio_segments = []
        for gs, ps, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_segments.append(audio)
        if not audio_segments:
            raise RuntimeError("Kokoro generated no audio.")
        combined_audio = np.concatenate(audio_segments)
        sf.write(output_path, combined_audio, 24000)
        return len(combined_audio) / 24000

class F5Provider(BaseTTS):
    name: str = "f5"
    supports_voice_cloning: bool = True
    available: bool = True

    def __init__(self, cfg: PipelineConfig):
        super().__init__(cfg)
        try:
            from f5_tts.api import F5TTS
            self.model = F5TTS(device=cfg.device, hf_cache_dir="./.model_cache")
            self.available = True
        except Exception as e:
            log.warning(f"F5-TTS initialization failed: {e}")
            self.available = False

    def synthesize(self, text: str, output_path: str, ref_wav: Optional[str] = None) -> float:
        if not self.available:
            raise RuntimeError("F5-TTS provider is not available.")
        from pathlib import Path
        ref_file = ref_wav or self.cfg.speaker_wav
        if not ref_file or not Path(ref_file).exists():
            raise FileNotFoundError(f"Reference WAV not found: {ref_file}")
            
        ref_text = self.model.transcribe(ref_file)
        
        self.model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=text,
            speed=self.cfg.xtts_speed,
            file_wave=output_path
        )
        
        import torchaudio
        info = torchaudio.info(output_path)
        return info.num_frames / info.sample_rate

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
        # Attempt 1: Standard synthesis
        try:
            _ = provider.synthesize(text, out_wav, speaker_wav)
            seg.tts_wav = out_wav
            success = True
        except Exception as e:
            log.warning(f"Attempt 1 failed for segment {seg.id} with provider {provider.name}: {e}. Trying Attempt 2 (text splitting) …")
            
        # Attempt 2: Split text into smaller sentences and merge
        if not success:
            try:
                import re
                from pydub import AudioSegment
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                if len(sentences) > 1:
                    log.info(f"Splitting segment {seg.id} into {len(sentences)} sub-sentences for retry …")
                    sub_wavs = []
                    for idx, sent in enumerate(sentences):
                        sub_out_wav = str(tts_dir / f"seg_{seg.id}_sub_{idx}.wav")
                        provider.synthesize(sent, sub_out_wav, speaker_wav)
                        sub_wavs.append(sub_out_wav)
                    
                    # Merge audio
                    combined = AudioSegment.empty()
                    for sw in sub_wavs:
                        combined += AudioSegment.from_wav(sw)
                    combined.export(out_wav, format="wav")
                    seg.tts_wav = out_wav
                    success = True
                    log.info(f"Successfully synthesized and merged sub-sentences for segment {seg.id}")
                else:
                    log.warning(f"Text too short to split for segment {seg.id}. Skipping Attempt 2.")
            except Exception as e:
                log.warning(f"Attempt 2 (split and merge) failed for segment {seg.id}: {e}")
                
        # Attempt 3: Fallback to KokoroProvider
        if not success and provider.name != "kokoro":
            log.warning(f"Attempt 2 failed. Falling back to Kokoro for segment {seg.id} …")
            try:
                kokoro_provider = KokoroProvider(cfg)
                if kokoro_provider.available:
                    kokoro_provider.synthesize(text, out_wav, speaker_wav)
                    seg.tts_wav = out_wav
                    success = True
                    log.info(f"Successfully synthesized segment {seg.id} using Kokoro fallback")
                else:
                    log.error("Kokoro provider is not available for fallback.")
            except Exception as e:
                log.error(f"Kokoro fallback failed for segment {seg.id}: {e}")
                
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
