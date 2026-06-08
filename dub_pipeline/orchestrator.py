import json
import logging
from pathlib import Path
from typing import Optional
from .config import PipelineConfig, Segment
from .audio import check_ffmpeg, extract_audio, extract_full_audio, assemble_dubbed_track, mix_with_bgm, mux_to_video
from .transcribe import transcribe
from .translate import translate_segments
from .synthesize import synthesize_segments

log = logging.getLogger("dub_pipeline.orchestrator")

class CheckpointManager:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.checkpoint_path = work_dir / "checkpoint.json"

    def save(self, stage: int, segments: list[Segment], **extra):
        data = {
            "stage": stage,
            "segments": [s.to_dict() for s in segments],
            **extra
        }
        self.checkpoint_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"Checkpoint saved: Stage {stage} -> {self.checkpoint_path.name}")

    def load(self) -> Optional[dict]:
        if not self.checkpoint_path.exists():
            return None
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            data["segments"] = [Segment.from_dict(s) for s in data["segments"]]
            return data
        except Exception as e:
            log.warning(f"Failed to load checkpoint: {e}")
            return None

def process_file(input_path: Path, output_path: Optional[Path], cfg: PipelineConfig):
    check_ffmpeg()

    if output_path is None:
        output_path = input_path.parent / (input_path.stem + cfg.output_suffix + input_path.suffix)

    work_dir = input_path.parent / (input_path.stem + "_dubwork")
    work_dir.mkdir(exist_ok=True)
    log.info(f"Working directory: {work_dir}")

    cp = CheckpointManager(work_dir)
    state = cp.load()

    segments = []
    current_stage = 0
    wav_16k = work_dir / "source_audio.wav"
    wav_full = work_dir / "source_audio_full.wav"

    transcript_path = work_dir / "transcript.json"

    if state:
        current_stage = state.get("stage", 0)
        segments = state.get("segments", [])
        if current_stage == 6:
            log.info(f"Pipeline is already fully completed! Output video is at {output_path}")
            return output_path
        log.info(f"Loaded checkpoint. Resuming from Stage {current_stage + 1}/6")
    elif transcript_path.exists():
        try:
            import json
            data = json.loads(transcript_path.read_text(encoding="utf-8"))
            segments = [Segment(start=d["start"], end=d["end"], text_ru=d["ru"], text_en=d["en"]) for d in data]
            if segments and segments[0].text_en:
                current_stage = 3
                log.info("Found completed transcript.json from previous run. Resuming from Stage 4/6 (Voice Synthesis).")
            else:
                current_stage = 2
                log.info("Found Russian transcript from previous run. Resuming from Stage 3/6 (Translation).")
        except Exception as e:
            log.warning(f"Failed to import old transcript.json: {e}")

    # ── Stage 1: Extract Audio ──────────────────────────────────────────
    if current_stage < 1:
        log.info("[1/6] Extracting audio tracks …")
        wav_16k = extract_audio(input_path, work_dir)
        wav_full = extract_full_audio(input_path, work_dir)
        current_stage = 1
        cp.save(current_stage, segments)

    # Ensure speaker reference WAV exists if not custom provided
    speaker_ref = work_dir / "speaker_ref.wav"
    if not cfg.speaker_wav and not speaker_ref.exists():
        log.info("Extracting default 30-second speaker voice reference …")
        # Ensure we have the 16k WAV to extract from
        if not wav_16k.exists():
            wav_16k = extract_audio(input_path, work_dir)
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_16k),
            "-ss", "0", "-t", "30",
            "-acodec", "copy", str(speaker_ref)
        ]
        from .audio import _run
        _run(cmd, silent=True)

    # ── Stage 2: Transcribe ─────────────────────────────────────────────
    if current_stage < 2:
        log.info("[2/6] Running speech transcription …")
        segments = transcribe(wav_16k, cfg)
        current_stage = 2
        cp.save(current_stage, segments)

    # ── Stage 3: Translate ──────────────────────────────────────────────
    if current_stage < 3:
        log.info("[3/6] Running translation (RU → EN) …")
        segments = translate_segments(segments, cfg)
        current_stage = 3
        cp.save(current_stage, segments)

    # ── Stage 4: Synthesize ─────────────────────────────────────────────
    if current_stage < 4:
        log.info("[4/6] Running voice synthesis (Coqui TTS) …")
        segments = synthesize_segments(segments, cfg, work_dir)
        current_stage = 4
        cp.save(current_stage, segments)

    # ── Stage 5: Assembly ───────────────────────────────────────────────
    if current_stage < 5:
        log.info("[5/6] Assembling dubbed audio track …")
        dubbed_track = assemble_dubbed_track(segments, wav_16k, work_dir, cfg)
        if cfg.keep_bgm:
            final_audio = mix_with_bgm(dubbed_track, wav_full, segments, work_dir, cfg)
        else:
            final_audio = dubbed_track
        current_stage = 5
        cp.save(current_stage, segments, final_audio_path=str(final_audio))
    else:
        final_audio_str = state.get("final_audio_path") if state else None
        if final_audio_str:
            final_audio = Path(final_audio_str)
        else:
            final_audio = work_dir / "mixed_audio.wav" if cfg.keep_bgm else work_dir / "dubbed_track.wav"

    # ── Stage 6: Mux ────────────────────────────────────────────────────
    if current_stage < 6:
        log.info("[6/6] Muxing final audio back to video …")
        mux_to_video(input_path, final_audio, output_path)
        current_stage = 6
        cp.save(current_stage, segments)

    log.info(f" Dubbing complete! Saved output to {output_path}")
    return output_path

def process_batch(folder: Path, cfg: PipelineConfig):
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts and cfg.output_suffix not in f.stem)
    if not files:
        log.warning(f"No video files found in {folder}")
        return
    log.info(f"Batch mode: {len(files)} file(s) found")
    for f in files:
        log.info(f"\nProcessing: {f.name}\n")
        try:
            process_file(f, None, cfg)
        except Exception as e:
            log.error(f"Failed to process: {f.name} — {e}")
