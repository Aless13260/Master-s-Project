"""
Web parser specifically for Non-8-K items (RSS feeds, Press Releases, etc.).
Based on web_parse_trafilatura.py but with inverted filtering logic:
- SKIPS all 8-K items.
- PROCESSES everything else (ignoring guidance_candidate flags).
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from typing import Iterable, Dict, Any
import re
import requests
import trafilatura
import datetime as dt
import zoneinfo
import hashlib
import tempfile
import os
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import argparse

# Import helper functions from the main script to avoid code duplication
# We need to make sure the directory is in path
sys.path.append(str(Path(__file__).parent))
try:
    from web_parse_trafilatura import (
        read_jsonl, looks_like_pdf, fetch_url, extract_with_trafilatura,
        find_exhibit_from_index, extract_pdf_links_from_html,
        UA, MIN_CONTENT_LENGTH, PROJECT_ROOT, CACHE_PATH, TZ
    )
    # Override paths for IR stream
    POINTERS_PATH = PROJECT_ROOT / "ingestion_json" / "pointers_IR.json"
    OUT_PATH = PROJECT_ROOT / "ingestion_json" / "contents_IR.jsonl"
except ImportError:
    # Fallback if import fails (e.g. if running from different cwd), though sys.path.append should fix it.
    # For robustness in this specific task, I will redefine the core logic here if import fails, 
    # but to keep it clean I'll assume the file exists as I see it in the workspace.
    pass

def main(
    pointers_path: Path = POINTERS_PATH,
    out_path: Path = OUT_PATH,
    delay: float = 0.15,
    no_cache: bool = False,
    cache_path: Path | None = None,
    reprocess: bool = False,
    limit: int | None = None,
) -> int:
    seen = set()
    # choose cache path
    chosen_cache = Path(cache_path) if cache_path else CACHE_PATH
    # load cache mapping of html_hash -> content_hash and content_hash -> metadata
    cache: Dict[str, Any] = {"html_to_content": {}, "content_meta": {}}
    if not no_cache and chosen_cache.exists():
        try:
            with chosen_cache.open("r", encoding="utf-8") as cf:
                cache = json.load(cf)
        except Exception:
            cache = {"html_to_content": {}, "content_meta": {}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load already processed UIDs to avoid re-processing
    if not reprocess and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    uid = obj.get("uid")
                    if uid:
                        seen.add(uid)
                except Exception:
                    continue

    total = 0
    added = 0
    skipped_8k = 0
    
    print(f"Starting Non-8-K extraction...")
    print(f"Reading pointers from {pointers_path}")
    print(f"Writing to {out_path}")

    with out_path.open("a", encoding="utf-8") as out:
        for p in read_jsonl(pointers_path):
            if limit is not None and added >= limit:
                break
            total += 1
            uid = p.get("uid")
            if not uid:
                continue
            
            # --- FILTERING LOGIC CHANGED HERE ---
            title = (p.get("title") or "").lower()
            source_id = (p.get("source_id") or "").lower()
            
            # 1. Skip if it looks like an 8-K
            if "8-k" in title or "8k" in source_id:
                skipped_8k += 1
                continue
                
            # 2. Skip if already processed (unless reprocess=True)
            if uid in seen:
                continue
            
            # 3. Process everything else!
            print(f"[Processing] {uid} | {p.get('title')}")
            
            link = p.get("link")
            if not link:
                continue

            item: Dict[str, Any] = {
                "uid": uid,
                "source_id": p.get("source_id"),
                "link": link,
                "original_link": link,
                "title": p.get("title"),
                "published_at": p.get("published_at"),
                "discovered_at": p.get("discovered_at"),
                "fetched_at": dt.datetime.now(TZ).isoformat(),
                "fetch_status": "skipped",
                "extracted_text": None,
                "pdf_links": []
            }

            if looks_like_pdf(link):
                item["fetch_status"] = "skipped_pdf"
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                added += 1
                seen.add(uid)
                continue

            resp = fetch_url(link)
            if not resp:
                item["fetch_status"] = "fetch_failed"
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                added += 1
                seen.add(uid)
                continue

            ct = resp.headers.get("Content-Type", "")
            if "pdf" in ct.lower():
                item["fetch_status"] = "skipped_pdf"
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                added += 1
                seen.add(uid)
                continue

            # Compute SHA256 of raw response content (bytes). Use this to skip identical HTML quickly.
            html_bytes = resp.content
            try:
                html_hash = hashlib.sha256(html_bytes).hexdigest()
            except Exception:
                html_hash = None

            # If we've seen this exact HTML before, map to existing content if available and skip re-extraction
            mapped_content_hash = None
            if not no_cache and html_hash and html_hash in cache.get("html_to_content", {}):
                mapped_content_hash = cache["html_to_content"][html_hash]
                meta = cache.get("content_meta", {}).get(mapped_content_hash)
                if meta:
                    # Write a duplicate record pointing to existing content
                    item["fetch_status"] = "duplicate_html_skipped"
                    item["extracted_text"] = None
                    item["duplicate_of"] = meta.get("uid")
                    item["content_hash"] = mapped_content_hash
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    added += 1
                    seen.add(uid)
                    # persist cache and continue (only if caching enabled)
                    if not no_cache:
                        try:
                            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=chosen_cache.parent) as tf:
                                json.dump(cache, tf, ensure_ascii=False)
                                tmpname = tf.name
                            os.replace(tmpname, chosen_cache)
                        except Exception:
                            pass
                    # polite short delay
                    time.sleep(delay)
                    continue

            html = resp.text

            # Track any discovered PDF links in the resolved content.
            discovered_pdf_links: list[str] = []

            # If this is an SEC EDGAR index page (8-K, 10-Q, etc.), try to find and follow the Exhibit 99 link
            # (Less likely for non-8K, but possible for other filings)
            if "sec.gov" in link.lower() and "Archives/edgar" in link:
                exhibit_url = find_exhibit_from_index(html, link)
                
                if exhibit_url:
                    print(f"[SEC] Found exhibit: {exhibit_url}")
                    time.sleep(0.3)
                    exhibit_resp = fetch_url(exhibit_url)
                    if exhibit_resp:
                        if "pdf" not in exhibit_resp.headers.get("Content-Type", "").lower():
                            html = exhibit_resp.text
                            link = exhibit_url
                            item["link"] = exhibit_url
                            item["resolved_exhibit_url"] = exhibit_url
                            discovered_pdf_links = extract_pdf_links_from_html(html, exhibit_url)
                        else:
                            item["link"] = exhibit_url
                            item["resolved_exhibit_url"] = exhibit_url
                            item["fetch_status"] = "skipped_pdf"
                            item["pdf_links"] = [exhibit_url]
                            out.write(json.dumps(item, ensure_ascii=False) + "\n")
                            added += 1
                            seen.add(uid)
                            time.sleep(delay)
                            continue

            if not discovered_pdf_links:
                discovered_pdf_links = extract_pdf_links_from_html(html, link)
            if discovered_pdf_links:
                item["pdf_links"] = discovered_pdf_links

            # Text extraction main body
            text = extract_with_trafilatura(html, link)
            
            if text and len(text) < MIN_CONTENT_LENGTH:
                if item.get("pdf_links"):
                    print(f"[INFO] Content short ({len(text)} chars) but PDFs found: {uid}")
                    item["fetch_status"] = "pdf_links_found"
                    item["extracted_text"] = text
                else:
                    print(f"[SKIP] Content too short ({len(text)} chars): {uid}")
                    item["fetch_status"] = "skipped_too_short"
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    added += 1
                    seen.add(uid)
                    continue

            if text:
                if item.get("fetch_status") not in ("pdf_links_found",):
                    item["fetch_status"] = "ok"
                item["extracted_text"] = text
            else:
                item["fetch_status"] = "extract_failed"
            
            content_hash = None
            if text:
                try:
                    normalized = " ".join(text.split())
                    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                except Exception:
                    content_hash = None

            if content_hash:
                if not no_cache and content_hash in cache.get("content_meta", {}):
                    meta = cache["content_meta"][content_hash]
                    item["fetch_status"] = "duplicate_content_skipped"
                    item["extracted_text"] = None
                    item["duplicate_of"] = meta.get("uid")
                    item["content_hash"] = content_hash
                else:
                    if not no_cache:
                        cache.setdefault("content_meta", {})[content_hash] = {
                            "uid": uid,
                            "title": item.get("title"),
                            "source_id": item.get("source_id"),
                            "fetched_at": item.get("fetched_at")
                        }
                    item["content_hash"] = content_hash

                if html_hash and not no_cache:
                    cache.setdefault("html_to_content", {})[html_hash] = content_hash

            if not no_cache:
                try:
                    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=chosen_cache.parent) as tf:
                        json.dump(cache, tf, ensure_ascii=False)
                        tmpname = tf.name
                    os.replace(tmpname, chosen_cache)
                except Exception:
                    pass
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            added += 1
            seen.add(uid)
            time.sleep(delay)

    print(f"Processed {total} pointers.")
    print(f"Skipped {skipped_8k} 8-K items.")
    print(f"Wrote {added} new content records -> {out_path}")
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and extract text from NON-8-K pointer URLs.")
    parser.add_argument("--pointers", type=str, default=str(POINTERS_PATH), help="Path to pointers.json (JSONL).")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output JSONL path for extracted content.")
    parser.add_argument("--delay", type=float, default=0.15, help="Polite delay between requests.")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Disable fetch cache.")
    parser.add_argument("--cache-path", type=str, default=str(CACHE_PATH), help="Cache JSON path.")
    parser.add_argument("--reprocess", action="store_true", default=False, help="Do not skip already-seen UIDs in the output.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after writing N records.")
    args = parser.parse_args()

    main(
        pointers_path=Path(args.pointers),
        out_path=Path(args.out),
        delay=float(args.delay),
        no_cache=bool(args.no_cache),
        cache_path=Path(args.cache_path) if args.cache_path else None,
        reprocess=bool(args.reprocess),
        limit=args.limit,
    )
