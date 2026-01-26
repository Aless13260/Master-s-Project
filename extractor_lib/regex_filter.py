"""Simple regex-based prefilter for guidance statements.

Reads `ingestion_json/contents.jsonl`, looks for forward-looking patterns,
and writes `extractor_lib/candidate_guidance.jsonl` with entries that match.

Simple logic: If text contains forward-looking verbs OR a proximity match
(forward verb near a number), it passes through.

Usage:
    python extractor_lib/regex_filter.py
"""

import re
import json
from pathlib import Path

# Single pattern - forward-looking verbs. That's it.
PATTERN = re.compile(
    r"\b(expect|anticipate|forecast|outlook|guidance|project|intend|target)\b",
    re.IGNORECASE
)


def has_guidance_patterns(text: str) -> bool:
    """Return True if text contains forward-looking language."""
    return bool(PATTERN.search(text))


def extract_candidates(input_paths: list[Path], output_path: Path):
    """Filter contents to only those with guidance-like language."""
    total = 0
    kept = 0
    out_f = output_path.open("w", encoding="utf-8")

    for input_path in input_paths:
        if not input_path.exists():
            print(f"Skipping missing input file: {input_path}")
            continue
            
        print(f"Scanning {input_path}...")
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

                text = obj.get("extracted_text") or obj.get("text") or obj.get("content") or ""
                if not text:
                    continue

                if has_guidance_patterns(text):
                    kept += 1
                    candidate = {
                        "uid": obj.get("uid"),
                        "source_id": obj.get("source_id"),
                        "title": obj.get("title"),
                        "link": obj.get("link") or obj.get("source_url") or obj.get("url"),
                        "published_at": obj.get("published_at"),
                        "extracted_text": text,
                    }
                    out_f.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    out_f.close()
    return {"total": total, "kept": kept}


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_paths = [
        base_dir / "ingestion_json" / "contents.jsonl",
    ]
    output_path = base_dir / "extractor_lib" / "candidate_guidance.jsonl"

    stats = extract_candidates(input_paths, output_path)
    print(f"Processed {stats['total']} records, kept {stats['kept']} candidates")
