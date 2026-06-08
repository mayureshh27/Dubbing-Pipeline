import argparse
import logging
from pathlib import Path
import torch

# Monkey patch torch.load to default weights_only=False for compatibility with Coqui TTS in PyTorch 2.6+
orig_load = torch.load
def new_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = new_load

from dub_pipeline.config import PipelineConfig
from dub_pipeline.orchestrator import process_file, process_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dub_pipeline.main")

def main():
    parser = argparse.ArgumentParser(
        description="Modular Whisper → MarianMT → XTTS-v2 Russian→English video dubbing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single video file")
    group.add_argument("--batch", type=Path, help="Folder of videos to process")

    parser.add_argument("--output", type=Path, default=None, help="Output path (single file mode only)")
    parser.add_argument("--speaker_wav", type=str, default=None, help="Path to reference speaker WAV (6–30 s clean speech)")
    parser.add_argument("--whisper_model", type=str, default="h2oai/faster-whisper-large-v3-turbo")
    parser.add_argument("--no_bgm", action="store_true", help="Replace audio entirely (no BGM mix)")
    parser.add_argument("--no_stretch", action="store_true", help="Disable time-stretch alignment")
    parser.add_argument("--xtts_speed", type=float, default=1.0, help="XTTS speed factor (0.8–1.4)")
    parser.add_argument("--translation", choices=["marian", "deep_translator"], default="marian")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")

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
