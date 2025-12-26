import json
import time
import sys
from pathlib import Path
from io import BytesIO
import requests
from pypdf import PdfReader
import hashlib
import argparse

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "ingestion_json" / "contents.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "ingestion_json" / "pdf_contents.jsonl"
UA = "AgenticFinanceResearchBot/0.1 (contact: aless13260@gmail.com)"

def _request_headers_for_url(url: str) -> dict[str, str]:
    headers = {
        "User-Agent": UA,
        "Accept": "application/pdf,*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    if "sec.gov" in url.lower():
        headers["User-Agent"] = "AgenticFinanceResearchBot/0.1 (Academic Research; contact: aless13260@gmail.com)"
        headers["Host"] = "www.sec.gov"
    return headers


def _looks_like_pdf_bytes(content: bytes) -> bool:
    return content.lstrip().startswith(b"%PDF-")


def fetch_pdf_text(url: str) -> str | None:
    """Downloads a PDF and extracts text using pypdf."""
    try:
        response = requests.get(url, headers=_request_headers_for_url(url), timeout=30, allow_redirects=True)
        response.raise_for_status()

        if not _looks_like_pdf_bytes(response.content):
            # Many "PDF" links actually return HTML (blocked pages, login, etc.).
            return None
        
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

def main(input_path: Path, output_path: Path, delay_s: float = 1.0):
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    print(f"Scanning {input_path} for PDFs (skipped_pdf and embedded pdf_links)...")
    
    # Track what we've already processed in the output file to avoid duplicates
    seen_uids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    seen_uids.add(data.get("uid"))
                except:
                    pass

    processed_count = 0
    
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "a", encoding="utf-8") as outfile:
        
        for line in infile:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            parent_uid = record.get("uid")
            parent_link = record.get("original_link") or record.get("link")

            pdf_urls: list[str] = []
            if record.get("fetch_status") == "skipped_pdf":
                if record.get("link"):
                    pdf_urls.append(record["link"])
            else:
                for u in (record.get("pdf_links") or []):
                    if isinstance(u, str) and u:
                        pdf_urls.append(u)

            # De-dupe per record
            dedup = []
            seen_local = set()
            for u in pdf_urls:
                if u in seen_local:
                    continue
                seen_local.add(u)
                dedup.append(u)
            pdf_urls = dedup

            for pdf_url in pdf_urls:
                # Create a stable UID for each PDF derived from the parent UID + URL.
                base = f"{parent_uid}|{pdf_url}" if parent_uid else pdf_url
                pdf_uid = hashlib.sha1(base.encode("utf-8")).hexdigest()

                if pdf_uid in seen_uids:
                    continue

                print(f"Processing PDF: {pdf_url}")
                text = fetch_pdf_text(pdf_url)

                out_record = {
                    "uid": pdf_uid,
                    "parent_uid": parent_uid,
                    "source_id": record.get("source_id"),
                    "link": pdf_url,
                    "parent_link": parent_link,
                    "title": record.get("title"),
                    "published_at": record.get("published_at"),
                    "discovered_at": record.get("discovered_at"),
                    "fetched_at": record.get("fetched_at"),
                    "fetch_status": "pdf_failed",
                    "extracted_text": None,
                }

                if text and len(text) > 100:
                    out_record["extracted_text"] = text
                    out_record["fetch_status"] = "ok_pdf"
                    processed_count += 1
                    print(f"  -> Extracted {len(text)} chars")
                else:
                    print("  -> Failed or empty")

                outfile.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                outfile.flush()
                seen_uids.add(pdf_uid)

                # Be polite
                time.sleep(delay_s)

    print(f"Done. Processed {processed_count} PDFs. Results in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDFs discovered during ingestion.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_PATH), help="Input contents JSONL path.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output PDF contents JSONL path.")
    parser.add_argument("--delay", type=float, default=1.0, help="Polite delay between PDF fetches.")
    args = parser.parse_args()

    main(Path(args.input), Path(args.out), delay_s=float(args.delay))
