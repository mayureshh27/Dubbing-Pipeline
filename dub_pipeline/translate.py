import logging
import gc
import re
import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import torch
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.translate")

CACHE_FILE = Path("artifacts/translation_cache.json")

TERMINOLOGY_ENFORCEMENTS = {
    r"\b(traversal object|pointer traversal)\b": "iterator",
    r"\b(templates? class|pattern class)\b": "class template",
    r"\b(templates? function|pattern function)\b": "function template",
    r"\b(transfer semantics|move semantics?)\b": "move semantics",
    r"\b(smart pointer)\b": "smart pointer",
    r"\b(unique pointer)\b": "unique_ptr",
    r"\b(shared pointer)\b": "shared_ptr",
    r"\b(weak pointer)\b": "weak_ptr",
    r"\b(standard library)\b": "standard library",
    r"\b(std vector|vector std)\b": "std::vector",
}

def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Failed to load translation cache: {e}")
    return {}

def save_cache(cache: dict):
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"Failed to save translation cache: {e}")

def get_normalized_hash(text: str) -> str:
    normalized = re.sub(r'[^\w\s]', '', text.lower())
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def estimate_spoken_duration(text: str) -> float:
    """Weighted timing model estimating spoken duration in seconds."""
    words = text.split()
    duration = 0.0
    for w in words:
        if any(c in w for c in ["::", "->", "<", ">", "_", "std::"]):
            duration += 0.9
        elif w.lower() in ["raii", "nullptr", "vector", "template", "const"]:
            duration += 0.7
        else:
            duration += 0.32
    duration += text.count(",") * 0.25
    duration += text.count(".") * 0.4
    duration += text.count("?") * 0.4
    duration += text.count("!") * 0.4
    return max(0.8, duration)

def apply_terminology_dictionary(text: str) -> str:
    if not text:
        return ""
    for pattern, replacement in TERMINOLOGY_ENFORCEMENTS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text

def translate_segments(segments: List[Segment], cfg: PipelineConfig, work_dir: Optional[Path] = None) -> List[Segment]:
    log.info(f"Translating {len(segments)} segments (RU→EN) …")
    cache = load_cache()
    
    needed_segments = []
    
    # 1. Check Cache
    for seg in segments:
        h = get_normalized_hash(seg.text_ru)
        if h in cache:
            seg.text_en = cache[h]["translated_text"]
            log.debug(f"Cache hit for segment {seg.id}")
        else:
            needed_segments.append((seg, h))
            
    # 2. Translate Needed Segments using Fallback Hierarchy with Provider Stickiness
    if needed_segments:
        import time
        import random
        
        # Determine initial provider based on config
        backend = getattr(cfg, "translation_backend", "marian")
        if backend == "deep_translator":
            current_provider = "google"
        elif backend == "qwen":
            current_provider = "qwen"
        elif backend == "translategemma":
            current_provider = "translategemma"
        else:
            current_provider = "marian"

        google_failures = 0
        qwen_translator = None
        gemma_translator = None
        marian_translator = None
        
        log.info(f"Translating {len(needed_segments)} new segments (starting provider: {current_provider}) …")
        
        for idx, (seg, h) in enumerate(needed_segments):
            translated_text = ""
            success = False
            
            # --- Try TranslateGemma ---
            if current_provider == "translategemma":
                try:
                    if gemma_translator is None:
                        from .translategemma_fallback import TranslateGemmaTranslator
                        gemma_translator = TranslateGemmaTranslator(cfg.translategemma_model_path, device=cfg.device)
                    translated_text = gemma_translator.translate(seg.text_ru).strip()
                    success = True
                except Exception as e:
                    log.warning(f"TranslateGemma translation failed on segment {seg.id}: {e}. Falling back permanently to Qwen.")
                    current_provider = "qwen"

            # --- Try Google ---
            if not success and current_provider == "google":
                try:
                    from deep_translator import GoogleTranslator
                    translator = GoogleTranslator(source="ru", target="en")
                    translated_text = translator.translate(seg.text_ru).strip()
                    success = True
                    # Rate limiting protection sleep
                    time.sleep(random.uniform(0.05, 0.15))
                except Exception as e:
                    google_failures += 1
                    log.warning(f"Google Translate failed ({google_failures}/3) on segment {seg.id}: {e}")
                    if google_failures >= 3:
                        log.warning("Google Translate failed 3 times consecutively. Switching provider permanently to Qwen fallback for this run.")
                        current_provider = "qwen"
            
            # --- Try Qwen Fallback ---
            if not success and current_provider == "qwen":
                try:
                    if qwen_translator is None:
                        from .qwen_fallback import QwenFallbackTranslator
                        qwen_translator = QwenFallbackTranslator(cfg.qwen_model_path, device=cfg.device)
                    translated_text = qwen_translator.translate(seg.text_ru).strip()
                    success = True
                except Exception as e:
                    log.warning(f"Qwen fallback translation failed on segment {seg.id}: {e}. Switching permanently to MarianMT disaster recovery.")
                    current_provider = "marian"
                    
            # --- Try MarianMT Disaster Recovery ---
            if not success and current_provider == "marian":
                try:
                    if marian_translator is None:
                        marian_translator = MarianFallbackTranslator(cfg.marian_model)
                    translated_text = marian_translator.translate(seg.text_ru)
                    success = True
                except Exception as e:
                    log.error(f"Catastrophic failure: MarianMT failed on segment {seg.id}: {e}")
                    translated_text = seg.text_ru  # Fallback to original text if everything fails
                    success = True
            
            seg.text_en = translated_text
            
            # Update cache and save periodically
            cache[h] = {
                "source": seg.text_ru,
                "provider": current_provider,
                "translated_text": seg.text_en,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "refined": False
            }
            
            if idx % 10 == 0 or idx == len(needed_segments) - 1:
                save_cache(cache)
                log.info(f"   … translated {idx}/{len(needed_segments)} segments (current provider: {current_provider})")
                
        if qwen_translator is not None:
            qwen_translator.unload()
        if gemma_translator is not None:
            gemma_translator.unload()
        if marian_translator is not None:
            marian_translator.unload()

    # 3. Post-process terminology
    for seg in segments:
        seg.text_en = apply_terminology_dictionary(seg.text_en)

    # 4. Selective Refinement (Gemini Flash with Qwen local fallback)
    refine_segments_if_needed(segments, cfg)

    # 5. Log translation metrics
    if work_dir:
        try:
            metrics_file = work_dir / "artifacts" / "metrics.json"
            google_count = 0
            qwen_count = 0
            marian_count = 0
            for seg in segments:
                h = get_normalized_hash(seg.text_ru)
                entry = cache.get(h, {})
                prov = entry.get("provider", "marian")
                if prov == "google":
                    google_count += 1
                elif prov == "qwen":
                    qwen_count += 1
                else:
                    marian_count += 1
            
            metrics = {}
            if metrics_file.exists():
                try:
                    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            metrics["translation_provider_used"] = {
                "google": google_count,
                "qwen": qwen_count,
                "marian": marian_count
            }
            metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"Failed to save translation metrics: {e}")

    return segments

def refine_segments_if_needed(segments: List[Segment], cfg: PipelineConfig):
    api_key = os.environ.get("GEMINI_API_KEY")
    qwen_translator = None
    gemma_translator = None
    
    import urllib.request
    import json

    log.info("Running selective timing-based refinement …")
    for i, seg in enumerate(segments):
        if i % 100 == 0 or i == len(segments) - 1:
            log.info(f"   … checked/refined {i}/{len(segments)} segments")
        original_duration = seg.end - seg.start
        predicted_duration = estimate_spoken_duration(seg.text_en)
        
        # If predicted spoken duration exceeds budget by more than 20%
        if predicted_duration > (original_duration * 1.2):
            log.info(f"Segment {seg.id} exceeds budget (estimated {predicted_duration:.2f}s vs original {original_duration:.2f}s). Refining …")
            
            gemini_success = False
            if api_key:
                prompt = (
                    f"You are a professional educational localization refiner.\n"
                    f"Rewrite this translated English C++ lecture segment to be more concise and shorter so it fits in a {original_duration:.2f} second speaking window.\n"
                    f"Original Russian: {seg.text_ru}\n"
                    f"Literal English: {seg.text_en}\n"
                    f"Constraints:\n"
                    f"- Do not alter code/technical semantics.\n"
                    f"- Preserve terminology: iterator, template, pointer, std::vector.\n"
                    f"- Output ONLY the refined English text with no quotes, commentary, or explanation."
                )
                
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
                    req_data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.1}
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(req_data).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=10) as response:
                        res = json.loads(response.read().decode("utf-8"))
                        refined_text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if refined_text:
                            seg.text_refined = refined_text
                            log.info(f"   Refined (Gemini) {seg.id}: '{seg.text_en}' -> '{seg.text_refined}'")
                            gemini_success = True
                            continue
                except Exception as ex:
                    log.warning(f"Refinement Gemini API error on segment {seg.id}: {ex}")

            if not gemini_success:
                if getattr(cfg, "translation_backend", "") == "translategemma":
                    log.info("Trying TranslateGemma local fallback for timing-aware refinement …")
                    try:
                        if gemma_translator is None:
                            from .translategemma_fallback import TranslateGemmaTranslator
                            gemma_translator = TranslateGemmaTranslator(cfg.translategemma_model_path, device=cfg.device)
                        refined_text = gemma_translator.rewrite_for_timing(seg.text_en, original_duration)
                        if refined_text:
                            seg.text_refined = refined_text
                            log.info(f"   Refined (TranslateGemma) {seg.id}: '{seg.text_en}' -> '{seg.text_refined}'")
                            continue
                    except Exception as ex:
                        log.warning(f"TranslateGemma local refinement failed on segment {seg.id}: {ex}")

                log.info("Trying Qwen local fallback for timing-aware refinement …")
                try:
                    if qwen_translator is None:
                        from .qwen_fallback import QwenFallbackTranslator
                        qwen_translator = QwenFallbackTranslator(cfg.qwen_model_path, device=cfg.device)
                    refined_text = qwen_translator.rewrite_for_timing(seg.text_en, original_duration)
                    if refined_text:
                        seg.text_refined = refined_text
                        log.info(f"   Refined (Qwen) {seg.id}: '{seg.text_en}' -> '{seg.text_refined}'")
                        continue
                except Exception as ex:
                    log.warning(f"Qwen local refinement failed on segment {seg.id}: {ex}")
                
        seg.text_refined = seg.text_en

    if qwen_translator is not None:
        qwen_translator.unload()
    if gemma_translator is not None:
        gemma_translator.unload()

class MarianFallbackTranslator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None:
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise ImportError("Run: pip install transformers sentencepiece")
        log.info(f"Loading {self.model_name} on CPU for memory safety …")
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name).to("cpu")
        self.model.eval()

    def unload(self):
        if self.model is not None:
            log.info("Unloading Marian model …")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()

    def translate(self, text: str) -> str:
        self.load()
        import torch
        tokens = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
        with torch.no_grad():
            translated = self.model.generate(**tokens)
        decoded = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return decoded[0].strip()

def _translate_marian(segments: List[Segment], cfg: PipelineConfig):
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        raise ImportError("Run: pip install transformers sentencepiece")

    log.info(f"Loading {cfg.marian_model} on CPU for memory safety …")
    tokenizer = MarianTokenizer.from_pretrained(cfg.marian_model)
    model = MarianMTModel.from_pretrained(cfg.marian_model).to("cpu")
    model.eval()

    BATCH = 16
    texts = [s.text_ru for s in segments]

    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cpu")
        with torch.no_grad():
            translated = model.generate(**tokens)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        for j, en in enumerate(decoded):
            segments[i + j].text_en = en.strip()

    del model
    gc.collect()
