"""Simple regex-based prefilter for guidance statements.

Reads `pointerEvents/contents.jsonl`, looks for guidance-related patterns,
and writes `extractor_lib/candidate_guidance.jsonl` with match metadata. 

Usage:
    python extractor_lib/regex_filter.py
"""

import re
import json
from pathlib import Path

PATTERNS = [
    # explicit forward-looking verbs
    r"\b(expect|anticipate|forecast|project|estimate|guide|outlook|plan|target|intend)\b",
    # temporal tokens (quarters, fiscal years)
    r"\b(Q[1-4]|FY\b|fiscal year|next quarter|next year|this quarter|upcoming quarter|quarter)\b",
    # metric words
    r"\b(revenue|earnings|eps|margin|growth|sales|income|operating|guidance|outlook)\b",
    # dollar amounts or units (e.g. $45.3B, 3.2 billion)
    r"\b\$\s?\d{1,3}(?:[\,\.]\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s?(?:billion|million|bn|m|B|M)\b",
    # range indicators (e.g. "$83-87", "between 10 and 12")
    r"\b(between|to|and|range|\d+\s?-\s?\d+|from\s+\$?\d+\s+to\s+\$?\d+)\b",
    # proximity: expect/forecast within 120 chars of a number or $ amount
    r"(?s)(expect|anticipate|forecast|project|estimate|guide).{0,120}?\$?\d",
]

# Weights to make some patterns more important than others (same order as PATTERNS)
WEIGHTS = [2.0, 1.0, 0.5, 1.0, 1.5, 4.0]

COMPILED = [re.compile(p, flags=re.IGNORECASE) for p in PATTERNS]


def score_text(text: str) -> dict:
    """Return matched pattern indexes, weighted score and normalized confidence.

    Confidence is sum(matched_weights)/sum(all_weights) so patterns like
    numeric amounts and proximity matches carry more weight.
    """
    matches = []
    score = 0.0
    for i, rx in enumerate(COMPILED):
        if rx.search(text):
            matches.append(i)
            score += WEIGHTS[i]
    max_score = sum(WEIGHTS)
    confidence = score / max_score if max_score else 0.0
    return {"matched_patterns": matches, "match_score": score, "confidence": confidence, "match_count": len(matches)}


def extract_candidates(input_path: Path, output_path: Path, min_confidence: float = 0.2):
    total = 0
    kept = 0
    out_f = output_path.open("w", encoding="utf-8")

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = obj.get("extracted_text") or obj.get("text") or obj.get("content") or obj.get("html") or ""
            if not text:
                continue

            meta = score_text(text)
            if meta["confidence"] >= min_confidence:
                kept += 1
                candidate = {
                    "uid": obj.get("uid"),
                    "source_id": obj.get("source_id"),
                    "title": obj.get("title"),
                    "source_url": obj.get("source_url") or obj.get("link") or obj.get("url"),
                    "match_count": meta.get("match_count"),
                    "matched_patterns": meta.get("matched_patterns"),
                    "match_score": meta.get("match_score"),
                    "confidence": meta.get("confidence"),
                    "extracted_text_preview": text[:200].replace("\n", " "),
                }
                out_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    out_f.close()
    return {"total": total, "kept": kept}


if __name__ == "__main__":

    input_path = Path("C:/Users/aless/Github/FinanceProject/pointerEvents/contents.jsonl")
    output_path = Path("C:/Users/aless/Github/FinanceProject/extractor_lib/candidate_guidance.jsonl")

    stats = extract_candidates(input_path, output_path, min_confidence=0.5)
    print(f"Processed {stats['total']} records, kept {stats['kept']} candidates")
