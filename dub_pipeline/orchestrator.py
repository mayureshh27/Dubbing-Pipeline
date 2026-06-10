import json
import logging
import torch
from pathlib import Path
from typing import Optional, List
from .config import PipelineConfig, Segment
from .audio import check_ffmpeg, extract_audio, extract_full_audio, assemble_dubbed_track, mix_with_bgm, mux_to_video
from .transcribe import transcribe
from .segment import semantic_segmentation
from .translate import translate_segments
from .synthesize import synthesize_segments

log = logging.getLogger("dub_pipeline.orchestrator")

class ManifestManager:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.manifest_path = work_dir / "manifest.json"

    def load(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                log.warning(f"Failed to load manifest: {e}")
        return {
            "schema_version": "1.0",
            "pipeline_version": "2.0.0",
            "state": {
                "stage": "extract",
                "last_successful_segment_index": 0,
                "last_segment_id": "",
                "status": "idle"
            },
            "metrics": {
                "total_duration_mismatch": 0.0,
                "average_stretch_ratio": 1.0,
                "failed_segments_count": 0
            }
        }

    def save(self, stage: str, status: str = "running", last_idx: int = 0, last_id: str = "", metrics: Optional[dict] = None):
        manifest = self.load()
        manifest["state"]["stage"] = stage
        manifest["state"]["status"] = status
        manifest["state"]["last_successful_segment_index"] = last_idx
        manifest["state"]["last_segment_id"] = last_id
        if metrics:
            manifest["metrics"].update(metrics)
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"Manifest updated: stage={stage}, status={status}")

def save_artifact(name: str, segments: List[Segment], work_dir: Path):
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    file_path = artifacts_dir / f"{name}.json"
    data = {
        "schema_version": "1.0",
        "segments": [s.to_dict() for s in segments]
    }
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"Artifact saved: {file_path.name}")

def load_artifact(name: str, work_dir: Path) -> Optional[List[Segment]]:
    file_path = work_dir / "artifacts" / f"{name}.json"
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return [Segment.from_dict(s) for s in data["segments"]]
    except Exception as e:
        log.warning(f"Failed to load artifact {name}: {e}")
        return None

def process_file(input_path: Path, output_path: Optional[Path], cfg: PipelineConfig):
    check_ffmpeg()

    if output_path is None:
        output_path = input_path.parent / (input_path.stem + cfg.output_suffix + input_path.suffix)

    work_dir = input_path.parent / (input_path.stem + "_dubwork")
    work_dir.mkdir(exist_ok=True)
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    log.info(f"Working directory: {work_dir}")

    mm = ManifestManager(work_dir)
    manifest = mm.load()
    
    current_stage = manifest["state"]["stage"]
    segments = []
    
    wav_16k = work_dir / "source_audio.wav"
    wav_full = work_dir / "source_audio_full.wav"

    # Try loading from the furthest valid artifact
    for art_name in ["refined", "translated", "segmented", "transcript"]:
        loaded = load_artifact(art_name, work_dir)
        if loaded:
            segments = loaded
            log.info(f"Restored {len(segments)} segments from artifact '{art_name}'")
            break

    import time
    metrics = {
        "stt_seconds": 0.0,
        "translation_seconds": 0.0,
        "tts_seconds": 0.0,
        "alignment_seconds": 0.0,
        "peak_vram_mb": 0.0
    }

    # ── Stage 1: Extract Audio ──────────────────────────────────────────
    if current_stage == "extract" or not wav_16k.exists():
        log.info("[1/8] Extracting audio tracks …")
        wav_16k = extract_audio(input_path, work_dir)
        wav_full = extract_full_audio(input_path, work_dir)
        current_stage = "transcribe"
        mm.save(current_stage, "completed")

    # Ensure speaker reference WAV exists if not custom provided
    speaker_ref = work_dir / "speaker_ref.wav"
    if not cfg.speaker_wav and not speaker_ref.exists():
        log.info("Extracting default 30-second speaker voice reference …")
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
    if current_stage == "transcribe":
        log.info("[2/8] Running speech transcription …")
        start_t = time.time()
        segments = transcribe(wav_16k, cfg)
        metrics["stt_seconds"] = time.time() - start_t
        save_artifact("transcript", segments, work_dir)
        current_stage = "segment"
        mm.save(current_stage, "completed")

    # ── Stage 3: Semantic Segmentation ──────────────────────────────────
    if current_stage == "segment":
        log.info("[3/8] Running semantic segmentation …")
        if not segments:
            segments = load_artifact("transcript", work_dir) or []
        segments = semantic_segmentation(segments, cfg)
        save_artifact("segmented", segments, work_dir)
        current_stage = "translate"
        mm.save(current_stage, "completed")

    # ── Stage 4: Translate ──────────────────────────────────────────────
    if current_stage == "translate":
        log.info("[4/8] Running translation (RU → EN) …")
        if not segments:
            segments = load_artifact("segmented", work_dir) or []
        start_t = time.time()
        segments = translate_segments(segments, cfg, work_dir=work_dir)
        metrics["translation_seconds"] = time.time() - start_t
        save_artifact("translated", segments, work_dir)
        current_stage = "refine"
        mm.save(current_stage, "completed")

    # ── Stage 5: Refinement ─────────────────────────────────────────────
    if current_stage == "refine":
        log.info("[5/8] Running translation refinement (pass-through for now) …")
        if not segments:
            segments = load_artifact("translated", work_dir) or []
        for s in segments:
            if not s.text_refined:
                s.text_refined = s.text_en
        save_artifact("refined", segments, work_dir)
        current_stage = "synthesize"
        mm.save(current_stage, "completed")

    # ── Stage 6: Synthesize ─────────────────────────────────────────────
    if current_stage == "synthesize":
        log.info("[6/8] Running voice synthesis …")
        if not segments:
            segments = load_artifact("refined", work_dir) or []
        
        for s in segments:
            if not s.text_refined:
                s.text_refined = s.text_en
                
        start_t = time.time()
        segments = synthesize_segments(segments, cfg, work_dir)
        metrics["tts_seconds"] = time.time() - start_t
        
        last_idx = 0
        last_id = ""
        for idx, s in enumerate(segments):
            if s.tts_wav and Path(s.tts_wav).exists():
                last_idx = idx
                last_id = s.id
                
        save_artifact("refined", segments, work_dir)
        current_stage = "assemble"
        mm.save(current_stage, "completed", last_idx, last_id)

    # ── Stage 7: Assembly ───────────────────────────────────────────────
    if current_stage == "assemble":
        log.info("[7/8] Assembling dubbed audio track …")
        if not segments:
            segments = load_artifact("refined", work_dir) or []
        start_t = time.time()
        dubbed_track = assemble_dubbed_track(segments, wav_16k, work_dir, cfg)
        metrics["alignment_seconds"] = time.time() - start_t
        if cfg.keep_bgm:
            final_audio = mix_with_bgm(dubbed_track, wav_full, segments, work_dir, cfg)
        else:
            final_audio = dubbed_track
        current_stage = "mux"
        mm.save(current_stage, "completed", metrics={"final_audio_path": str(final_audio)})

    # ── Stage 8: Mux ────────────────────────────────────────────────────
    if current_stage == "mux":
        log.info("[8/8] Muxing final audio back to video …")
        manifest = mm.load()
        final_audio_str = manifest["metrics"].get("final_audio_path")
        if final_audio_str:
            final_audio = Path(final_audio_str)
        else:
            final_audio = work_dir / "mixed_audio.wav" if cfg.keep_bgm else work_dir / "dubbed_track.wav"
        
        mux_to_video(input_path, final_audio, output_path)
        current_stage = "done"
        
        # Track Peak VRAM
        if torch.cuda.is_available():
            metrics["peak_vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
        mm.save(current_stage, "completed", metrics=metrics)

    # Clean up temporary/sub-sentence audio files
    try:
        log.info("Performing post-run temporary file cleanup …")
        tts_dir = work_dir / "tts_segments"
        if tts_dir.exists():
            for sub_file in tts_dir.glob("seg_*_sub_*.wav"):
                sub_file.unlink()
        raw_track = work_dir / "dubbed_track_raw.wav"
        if raw_track.exists():
            raw_track.unlink()
    except Exception as e:
        log.warning(f"Post-run cleanup failed: {e}")

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
