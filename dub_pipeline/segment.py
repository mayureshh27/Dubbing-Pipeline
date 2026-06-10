import re
import logging
from typing import List
from .config import Segment
from .transcribe import compute_segment_id

log = logging.getLogger("dub_pipeline.segment")

def is_code_heavy(text: str) -> bool:
    code_indicators = [
        r'std::', r'vector', r'template', r'raii', r'const\s', r'class\s', r'struct\s',
        r'::', r'->', r'operator', r'namespace', r'void\s', r'int\s', r'double\s'
    ]
    # Check indicators or camelCase / snake_case
    if any(re.search(ind, text, re.IGNORECASE) for ind in code_indicators):
        return True
    if re.search(r'[a-z]+_[a-z]+|[a-z]+[A-Z]', text):
        return True
    return False

def split_segment_by_words(seg: Segment, max_words: int, max_duration: float) -> List[Segment]:
    """Splits a single segment into smaller pieces at word level, prioritizing pauses."""
    if not seg.words:
        words = seg.text_ru.split()
        if len(words) <= max_words:
            return [seg]
        num_words = len(words)
        duration = seg.end - seg.start
        estimated_words = []
        current_time = seg.start
        word_dur = duration / num_words if num_words > 0 else 0.0
        for w in words:
            estimated_words.append({
                "word": w + " ",
                "start": current_time,
                "end": current_time + word_dur,
                "probability": 1.0
            })
            current_time += word_dur
        seg = Segment(
            id=seg.id,
            start=seg.start,
            end=seg.end,
            text_ru=seg.text_ru,
            words=estimated_words,
            speech_rate=seg.speech_rate,
            pause_density=seg.pause_density
        )

    if len(seg.words) <= max_words:
        return [seg]

    sub_segments = []
    current_words = []
    current_start = seg.start
    
    for i, w in enumerate(seg.words):
        current_words.append(w)
        # Check if we should split
        word_count = len(current_words)
        duration = w["end"] - current_start
        
        # Lookahead for pause
        has_pause = False
        if i < len(seg.words) - 1:
            pause_len = seg.words[i + 1]["start"] - w["end"]
            if pause_len > 0.4:  # significant pause
                has_pause = True
        
        # Split conditions
        should_split = (word_count >= max_words) or (duration >= max_duration) or (has_pause and word_count >= max_words // 2)
        
        # Ensure we don't split if it's the last word
        if should_split and i < len(seg.words) - 1:
            text_part = "".join([x["word"] for x in current_words]).strip()
            if text_part:
                sub_segments.append(Segment(
                    id=compute_segment_id(current_start, text_part),
                    start=current_start,
                    end=w["end"],
                    text_ru=text_part,
                    words=current_words.copy()
                ))
            current_words = []
            current_start = seg.words[i + 1]["start"]
            
    # Add remaining words
    if current_words:
        text_part = "".join([x["word"] for x in current_words]).strip()
        if text_part:
            sub_segments.append(Segment(
                id=compute_segment_id(current_start, text_part),
                start=current_start,
                end=seg.end,
                text_ru=text_part,
                words=current_words
            ))
            
    return sub_segments

def semantic_segmentation(raw_segments: List[Segment]) -> List[Segment]:
    log.info("Running code-aware semantic segmentation …")
    optimized = []
    
    for seg in raw_segments:
        # Code indicator check
        code_heavy = is_code_heavy(seg.text_ru)
        
        # Standard limits vs strict code limits
        if code_heavy:
            max_words = 15
            max_duration = 6.0
        else:
            max_words = 28
            max_duration = 10.0
            
        sub_segs = split_segment_by_words(seg, max_words, max_duration)
        optimized.extend(sub_segs)
        
    log.info(f"→ Resegmented {len(raw_segments)} raw segments into {len(optimized)} semantic segments")
    return optimized
