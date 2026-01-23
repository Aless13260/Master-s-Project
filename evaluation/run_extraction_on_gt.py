import json
import sys
import argparse
from pathlib import Path
import time

# Add project root to path to import extractor_lib
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from extractor_lib.LLM_extractor import LLMExtractor

GT_DIR = PROJECT_ROOT / "evaluation" / "ground_truth"
CONTENTS_PATHS = [
    PROJECT_ROOT / "ingestion_json" / "contents.jsonl"
]
OUTPUT_FILE_STANDARD = PROJECT_ROOT / "evaluation" / "extracted_on_gt.jsonl"
OUTPUT_FILE_ENHANCED = PROJECT_ROOT / "evaluation" / "extracted_on_gt_enhanced.jsonl"

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
    parser.add_argument("--enhanced", action="store_true", help="Enable enhanced mode extraction (agentic period normalization).")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file, skipping processed UIDs.")
    parser.add_argument("--provider", default="deepseek", help="LLM provider (deepseek, openai, github)")
    parser.add_argument("--model", default=None, help="Specific model name (e.g., gpt-5, deepseek-chat)")
    parser.add_argument("--output", default=None, help="Custom output file name (relative to evaluation/)")
    args = parser.parse_args()

    records = load_gt_records()
    full_contents = load_full_contents()
    print(f"Loaded {len(records)} records.")
    
    # Determine output file based on mode or custom path
    if args.output:
        output_file = Path(__file__).parent / args.output
    else:
        output_file = OUTPUT_FILE_ENHANCED if args.enhanced else OUTPUT_FILE_STANDARD
    print(f"Mode: {'Enhanced' if args.enhanced else 'Standard'}")
    print(f"Output: {output_file}")

    # Check for existing progress
    existing_uids = set()
    if args.resume and output_file.exists():
        print(f"Checking existing output in {output_file}...")
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            if "uid" in rec:
                                existing_uids.add(rec["uid"])
                        except json.JSONDecodeError:
                            print(f"Warning: Malformed JSON on line {line_num+1} of {output_file}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")
            
        print(f"Found {len(existing_uids)} already processed documents.")
    elif not args.resume and output_file.exists():
        print(f"Starting from scratch. Existing file {output_file} will be overwritten.")

    # Initialize extractor
    extractor = LLMExtractor(provider=args.provider, model=args.model, temperature=0.0)
    print(f"Using provider: {args.provider}, model: {args.model or 'default'}")
    
    # Open in append mode if resuming, else write mode
    mode = "a" if args.resume else "w"
    with open(output_file, mode, encoding="utf-8") as out_f:
        for i, record in enumerate(records):
            uid = record.get("uid")
            
            if uid in existing_uids:
                print(f"Skipping {i+1}/{len(records)}: {uid} (already processed)")
                continue

            # Determine text source
            text = full_contents.get(uid)
            source_type = "full_content"
            
            if not text:
                # Check if it's synthetic or just missing
                is_synthetic = "synthetic" in record.get("source_id", "") or record.get("manual_verification_status") == "synthetic"
                text = record.get("full_text_snippet")
                
                if is_synthetic and text:
                    source_type = "synthetic_full"
                else:
                    source_type = "snippet_fallback"

            if not uid or not text:
                print(f"Skipping record {i}: missing uid or text")
                continue
                
            print(f"Processing {i+1}/{len(records)}: {uid} ({len(text)} chars, source: {source_type})...")
            
            try:
                # Build metadata for better extraction context
                # For synthetic docs, use document-level fields
                # For real docs, these may come from the record or be inferred
                published_at = record.get('published_at', '')
                company_name = record.get('company_name', '')
                
                # Synthetic docs may have company in metadata
                if not company_name and record.get('synthetic_metadata'):
                    company_info = record.get('synthetic_metadata', {}).get('company', {})
                    company_name = company_info.get('name', '')
                
                metadata = {
                    'source_url': record.get('source_url'),
                    'published_at': published_at,
                    'source_id': record.get('source_id'),
                    'company_name': company_name,  # Pass company for period normalization
                }
                
                # Extract guidance based on selected mode
                if args.enhanced:
                    guidance_items = extractor.extract_from_text(text, metadata, use_agentic_normalization=True)
                else:
                    guidance_items = extractor.extract_from_text(text, metadata, use_agentic_normalization=False)
                
                # Collect all items for this document
                guidance_list = []
                for item in guidance_items:
                    guidance_list.append(item.model_dump())
                
                # Write one record per document
                output_record = {
                    "uid": uid,
                    "guidance": guidance_list
                }
                out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                out_f.flush()
                    
            except Exception as e:
                print(f"  Error processing {uid}: {e}")
                
            # Rate limiting / politeness
            time.sleep(0.5)

if __name__ == "__main__":
    main()
