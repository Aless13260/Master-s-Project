import json
import sys
import argparse
from pathlib import Path
import time

# Add project root to path to import extractor_lib
sys.path.append(str(Path(__file__).parent.parent))

from extractor_lib.LLM_extractor import LLMExtractor

GT_DIR = Path("evaluation") / "ground_truth"
CONTENTS_PATHS = [
    Path("ingestion_json") / "contents_8k.jsonl",
    Path("ingestion_json") / "contents_IR.jsonl"
]
OUTPUT_FILE_STANDARD = Path("evaluation") / "extracted_on_gt.jsonl"
OUTPUT_FILE_AGENTIC = Path("evaluation") / "extracted_on_gt_agentic.jsonl"

def load_full_contents():
    """Load full text content from ingestion_json/contents_*.jsonl"""
    contents = {}
    for path in CONTENTS_PATHS:
        if path.exists():
            print(f"Loading full contents from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            uid = record.get("uid")
                            text = record.get("extracted_text")
                            if uid and text:
                                contents[uid] = text
                        except Exception:
                            continue
    return contents

def load_gt_records():
    records = []
    for file_path in GT_DIR.glob("*.jsonl"):
        print(f"Loading records from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records

def main():
    parser = argparse.ArgumentParser(description="Run extraction on ground truth data.")
    parser.add_argument("--agentic", action="store_true", help="Enable agentic review (two-stage extraction).")
    args = parser.parse_args()

    records = load_gt_records()
    full_contents = load_full_contents()
    print(f"Loaded {len(records)} records.")
    
    # Determine output file based on mode
    output_file = OUTPUT_FILE_AGENTIC if args.agentic else OUTPUT_FILE_STANDARD
    print(f"Mode: {'Agentic' if args.agentic else 'Standard'}")
    print(f"Output: {output_file}")

    # Check for existing progress
    existing_uids = set()
    if output_file.exists():
        print(f"Checking existing output in {output_file}...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "uid" in rec:
                            existing_uids.add(rec["uid"])
                    except:
                        pass
        print(f"Found {len(existing_uids)} already processed documents.")

    # Initialize extractor
    # Using deepseek as default provider, adjust if needed
    extractor = LLMExtractor(provider="deepseek", temperature=0.0)
    
    # Open in append mode to resume
    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, record in enumerate(records):
            uid = record.get("uid")
            
            if uid in existing_uids:
                print(f"Skipping {i+1}/{len(records)}: {uid} (already processed)")
                continue

            # Prefer full text from contents.jsonl, fallback to snippet in GT file
            text = full_contents.get(uid) or record.get("full_text_snippet")
            
            if not uid or not text:
                print(f"Skipping record {i}: missing uid or text")
                continue
                
            print(f"Processing {i+1}/{len(records)}: {uid} ({len(text)} chars)...")
            
            try:
                # Extract guidance based on selected mode
                if args.agentic:
                    guidance_items = extractor.extract_with_agentic_review(text)
                else:
                    guidance_items = extractor.extract_from_text(text)
                
                # Convert Pydantic models to dicts
                guidance_dicts = [item.model_dump() for item in guidance_items]
                
                # Write to output in the format expected by evaluation script (one record per UID)
                output_record = {
                    "uid": uid,
                    "guidance": guidance_dicts
                }
                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                out_f.flush()
                    
            except Exception as e:
                print(f"  Error processing {uid}: {e}")
                
            # Rate limiting / politeness
            time.sleep(0.5)

if __name__ == "__main__":
    main()
