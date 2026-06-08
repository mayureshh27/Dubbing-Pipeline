import logging
import gc
import re
import os
import json
import hashlib
from pathlib import Path
from typing import List
from datetime import datetime
import torch
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.translate")

CACHE_FILE = Path("translation_cache.json")

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

def translate_segments(segments: List[Segment], cfg: PipelineConfig) -> List[Segment]:
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
            
    # 2. Translate Needed Segments using Fallback Hierarchy
    if needed_segments:
        # Try Google Translate first
        google_success = False
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source="ru", target="en")
            log.info(f"Translating {len(needed_segments)} new segments via Google Translate …")
            for seg, h in needed_segments:
                translation = translator.translate(seg.text_ru)
                seg.text_en = translation.strip()
                # Update Cache
                cache[h] = {
                    "source": seg.text_ru,
                    "provider": "google",
                    "translated_text": seg.text_en,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "refined": False
                }
            google_success = True
        except Exception as e:
            log.warning(f"Google Translate failed: {e}. Falling back to MarianMT …")
            
        if not google_success:
            # Fallback to local MarianMT
            _translate_marian([pair[0] for pair in needed_segments], cfg)
            # Update cache for MarianMT translations
            for seg, h in needed_segments:
                cache[h] = {
                    "source": seg.text_ru,
                    "provider": "marian",
                    "translated_text": seg.text_en,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "refined": False
                }
                
        save_cache(cache)

    # 3. Post-process terminology
    for seg in segments:
        seg.text_en = apply_terminology_dictionary(seg.text_en)

    # 4. Selective Refinement (Gemini Flash / OpenAI / Mock Refiner)
    refine_segments_if_needed(segments)

    return segments

def refine_segments_if_needed(segments: List[Segment]):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.info("No GEMINI_API_KEY found. Skipping timing-based refinement step.")
        for seg in segments:
            seg.text_refined = seg.text_en
        return

    import urllib.request
    import json

    log.info("Running selective timing-based refinement via Gemini API …")
    for i, seg in enumerate(segments):
        original_duration = seg.end - seg.start
        predicted_duration = estimate_spoken_duration(seg.text_en)
        
        # If predicted spoken duration exceeds budget by more than 20%
        if predicted_duration > (original_duration * 1.2):
            log.info(f"Segment {seg.id} exceeds budget (estimated {predicted_duration:.2f}s vs original {original_duration:.2f}s). Refining …")
            
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
                        log.info(f"   Refined {seg.id}: '{seg.text_en}' -> '{seg.text_refined}'")
                        continue
            except Exception as ex:
                log.warning(f"Refinement API error on segment {seg.id}: {ex}")
                
        seg.text_refined = seg.text_en

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
