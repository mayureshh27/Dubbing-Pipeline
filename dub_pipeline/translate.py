import logging
import gc
import torch
from .config import PipelineConfig, Segment

log = logging.getLogger("dub_pipeline.translate")

def translate_segments(segments: list[Segment], cfg: PipelineConfig) -> list[Segment]:
    log.info(f"Translating {len(segments)} segments (RU→EN) …")

    if cfg.translation_backend == "marian":
        _translate_marian(segments, cfg)
    else:
        _translate_deep(segments)

    return segments

def _translate_marian(segments: list[Segment], cfg: PipelineConfig):
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
            segments[i + j].text_en = _clean_translation(en)

        if (i // BATCH) % 10 == 0:
            log.info(f"   … translated batch {i // BATCH}/{len(texts) // BATCH}")

    # Offload model from CPU RAM
    del model
    gc.collect()
    log.info("Translation complete & translation model offloaded from memory")

def _translate_deep(segments: list[Segment]):
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        raise ImportError("Run: pip install deep-translator")

    log.info("Translating using Google Translator API …")
    translator = GoogleTranslator(source="ru", target="en")
    for i, seg in enumerate(segments):
        seg.text_en = _clean_translation(translator.translate(seg.text_ru))
        if i % 100 == 0:
            log.info(f"   … translated {i}/{len(segments)} segments")

def _clean_translation(text: str) -> str:
    """Fix common issues with technical translations."""
    replacements = {
        "template": "template",
        "C++": "C++",
        "std::": "std::",
        "nullptr": "nullptr",
        "const": "const",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text.strip()
