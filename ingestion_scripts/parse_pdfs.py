import json
import time
import sys
from pathlib import Path
from io import BytesIO
import requests
from pypdf import PdfReader

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "ingestion_json" / "contents.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "ingestion_json" / "pdf_contents.jsonl"
UA = "AgenticFinanceResearchBot/0.1 (contact: aless13260@gmail.com)"

def fetch_pdf_text(url: str) -> str | None:
    """Downloads a PDF and extracts text using pypdf."""
    headers = {"User-Agent": UA}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        with BytesIO(response.content) as f:
            reader = PdfReader(f)
            text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
            return "\n".join(text)
    except Exception as e:
        print(f"[ERROR] Failed to parse PDF {url}: {e}")
        return None

def main():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        return

    print(f"Scanning {INPUT_PATH} for skipped PDFs...")
    
    # Track what we've already processed in the output file to avoid duplicates
    seen_uids = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    seen_uids.add(data.get("uid"))
                except:
                    pass

    processed_count = 0
    
    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "a", encoding="utf-8") as outfile:
        
        for line in infile:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # We only care about records that were skipped because they were PDFs
            if record.get("fetch_status") == "skipped_pdf":
                uid = record.get("uid")
                
                if uid in seen_uids:
                    continue

                url = record.get("link")
                print(f"Processing PDF: {url}")
                
                text = fetch_pdf_text(url)
                
                if text and len(text) > 100:
                    # Update the record
                    record["extracted_text"] = text
                    record["fetch_status"] = "ok_pdf"
                    
                    # Write to output
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    outfile.flush()
                    processed_count += 1
                    print(f"  -> Extracted {len(text)} chars")
                else:
                    print("  -> Failed or empty")
                
                # Be polite
                time.sleep(1)

    print(f"Done. Processed {processed_count} PDFs. Results in {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
