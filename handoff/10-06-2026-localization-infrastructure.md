# Handoff: AI Lecture Localization Infrastructure Hardening
**Date**: 10-06-2026  
**Topic**: hardended-multimodal-localization-infrastructure

This document provides a detailed summary of the architecture upgrades, bug fixes, and configuration additions implemented in the AI Lecture Localization System.

---

## 🛠️ Achievements & Implementation Summary

### 1. Robust Translation Hierarchy & Stickiness
* **Hierarchy**: Implemented Google Translate $\rightarrow$ Qwen-3B local fallback (via `llama-cpp-python`) $\rightarrow$ MarianMT emergency local backup.
* **Provider Stickiness**: If Google Translate fails 3 times consecutively (e.g., due to rate-limiting), the pipeline locks the translator to the Qwen fallback for the remainder of that lecture, maintaining stylistic and terminology consistency.
* **Anti-Spam Rate Limiting**: Added a random sleep (`0.05` to `0.15` seconds) between Google Translate requests to significantly reduce the risk of IP blocks.
* **Segment-Level Cache Resume**: Configured the translation cache to save and flush to disk every **10 segments** so that interrupted runs can be resumed without losing progress.

### 2. Upgraded Plugin TTS & Fallbacks
* **Kokoro-82M**: Fully implemented `KokoroProvider` using the official `KPipeline` ONNX interface.
* **F5-TTS Voice Cloning**: Fully integrated the flow-matching `F5Provider` with automatic reference text transcription.
* **TTS Retries**: Added Attempt 1 (default synthesis) $\rightarrow$ Attempt 2 (split into sub-sentences and merge) $\rightarrow$ Attempt 3 (fallback to Kokoro) to guarantee synthesis completion.

### 3. Advanced Audio Mastering & Quality Checks
* **Mastering Chain**: Built a chained FFmpeg signal process in `audio.py` (`highpass=80Hz` $\rightarrow$ `equalizer=3kHz` presenza boost $\rightarrow$ dynamic compression `compand` $\rightarrow$ two-pass `loudnorm` normalization).
* **Audio Quality Checks**: Added validations to skip silent, empty, or corrupted audio segments, and log warnings for clipped audio.
* **Unsafe Stretch Guardrails**: Skip or warn if a segment stretch ratio is dangerously high (e.g. skipping if $>1.35$, warning if $>1.2$).

### 4. Stage Profiling & VRAM Monitoring
* Wrapped all pipeline stages in timers and logged durations (`stt_seconds`, `translation_seconds`, etc.).
* Tracked peak VRAM usage via `torch.cuda.max_memory_allocated()`.
* Logged these metrics directly to `manifest.json` on completion.

---

## 🚀 How to Run the Pipeline

Run the pipeline from the project root using the synced virtual environment:

```powershell
uv run python main.py --input lecture01.mp4 --no_bgm
```

* **Offline Qwen Requirement**: To use Qwen translation offline, download `Qwen2.5-3B-Instruct-Q4_K_M.gguf` and place it in the `models/qwen/` folder.
