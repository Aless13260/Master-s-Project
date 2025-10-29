"""
Simple web parser using trafilatura to extract main text from URLs collected in pointer events.
Reads:  pointerEvents/pointers.json  (JSONL of pointer events)
Writes: pointerEvents/contents.jsonl (JSONL of extracted contents)

Notes:
- Skips links that clearly point to PDFs (by extension or content-type header).
- Uses a polite User-Agent and timeouts.
- Does not attempt heavy PDF parsing here (you mentioned handling PDFs later).
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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TZ = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")
UA = "AgenticFinanceResearchBot/0.1 (contact: aless13260@gmail.com)"

POINTERS_PATH = Path("pointerEvents") / "pointers.json"
OUT_PATH = Path("pointerEvents") / "contents.jsonl"
CACHE_PATH = Path("pointerEvents") / "fetch_cache.json"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue


def looks_like_pdf(url: str) -> bool:
    return url.lower().split("?")[0].endswith(".pdf")


def fetch_url(url: str, timeout: int = 15) -> requests.Response | None:
    headers = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    # SEC.gov requires lower request rates and proper identification
    # They allow up to 10 req/sec but we'll be more conservative
    if "sec.gov" in url.lower():
        headers["User-Agent"] = "AgenticFinanceResearchBot/0.1 (Academic Research; contact: aless13260@gmail.com)"
        headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        headers["Accept-Encoding"] = "gzip, deflate"
        headers["Host"] = "www.sec.gov"
    
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        # Check if SEC blocked us
        if "sec.gov" in url.lower() and "automated tool" in r.text.lower():
            print(f"[WARN] SEC.gov blocking automated access to: {url}")
            return None
        return r
    except Exception as e:
        print(f"[WARN] Fetch failed for {url}: {e}")
        return None


def extract_with_trafilatura(html: str, url: str) -> str | None:
    try:
        result = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        # For SEC exhibits, if trafilatura fails, try BeautifulSoup fallback
        if not result and "sec.gov" in url.lower():
            soup = BeautifulSoup(html, "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text
            text = soup.get_text(separator="\n", strip=True)
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
            return text if len(text) > 100 else None
        return result
    except Exception:
        return None

def find_exhibit_from_index(html: str, base_url: str) -> str | None:
    """Parse a filing index page and try to find the first exhibit URL that looks like a press release or EX-99."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # look for <a> tags with filenames or link text matching our patterns
        patterns = re.compile(r"(ex-?99|exhibit\s*99|press[\s_-]*release|earnings|results)", re.I)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            txt = (a.get_text(" ") or "").strip()
            if patterns.search(href) or patterns.search(txt):
                # build absolute URL
                return urljoin(base_url, href)
    except Exception:
        return None
    return None

def find_sublinks_in_text(extracted_text: str, base_url: str) -> list[Dict[str, Any]]:
    """Find URLs in extracted text and return them as sublink dictionaries.
    
    Args:
        extracted_text: The main extracted text content
        base_url: Base URL to resolve relative links
        
    Returns:
        List of sublink dictionaries with url, extracted_text, fetch_status, similarity_to_main fields
    """
    if not extracted_text:
        return []
    
    # Regex pattern to find URLs in text
    url_pattern = re.compile(
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+',
        re.IGNORECASE
    )
    
    urls = url_pattern.findall(extracted_text)
    sublinks = []
    
    for url in urls:
        # Clean up the URL (remove trailing punctuation)
        url = re.sub(r'[.,;:!?)]+$', '', url)
        
        # Add protocol if missing for www links
        if url.lower().startswith('www.'):
            url = 'https://' + url
            
        # Skip if it looks like a PDF
        if looks_like_pdf(url):
            continue
            
        # Create sublink dictionary
        sublink = {
            "url": url,
            "extracted_text": None,
            "fetch_status": "pending",
            "similarity_to_main": 0.0
        }
        sublinks.append(sublink)
    
    return sublinks

def compute_similarity(text1: str, text2: str) -> float:
    """Compute TF-IDF cosine similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)
    except Exception:
        return 0.0

def main(pointers_path: Path = POINTERS_PATH, out_path: Path = OUT_PATH, only_candidates: bool = True, mark_skipped: bool = False, delay: float = 1, no_cache: bool = False, cache_path: Path | None = None) -> int:
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
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    uid = obj.get("uid")
                    if uid:
                        seen.add(uid)
                except Exception:
                    continue

    # guidance keywords used to decide whether to follow a pointer
    guidance_patterns = ("8-k", "8k", "press release", "press-release", "earnings", "results", "guidance", "outlook")

    total = 0
    added = 0
    with out_path.open("a", encoding="utf-8") as out:
        for p in read_jsonl(pointers_path):
            total += 1
            uid = p.get("uid")
            if not uid or uid in seen:
                continue
            link = p.get("link")
            if not link:
                continue
            # decide whether this pointer is a candidate by title/summary (backwards compatible)
            is_candidate = False
            if p.get("guidance_candidate") is not None:
                is_candidate = bool(p.get("guidance_candidate"))
            else:
                title = (p.get("title") or "").lower()
                summary = (p.get("summary_hint") or "").lower()
                if any(k in title or k in summary for k in guidance_patterns):
                    is_candidate = True

            if only_candidates and not is_candidate:
                # mark as seen so we don't keep reprocessing uninteresting pointers
                seen.add(uid)
                if mark_skipped:
                    item: Dict[str, Any] = {
                        "uid": uid,
                        "source_id": p.get("source_id"),
                        "link": link,
                        "title": p.get("title"),
                        "published_at": p.get("published_at"),
                        "discovered_at": p.get("discovered_at"),
                        "fetched_at": dt.datetime.now(TZ).isoformat(),
                        "fetch_status": "skipped_not_candidate",
                        "extracted_text": None
                    }
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    added += 1
                else:
                    # quiet skip
                    print(f"[SKIP] not a candidate: {uid} - {p.get('title')}")
                continue
            item: Dict[str, Any] = {
                "uid": uid,
                "source_id": p.get("source_id"),
                "link": link,
                "title": p.get("title"),
                "published_at": p.get("published_at"),
                "discovered_at": p.get("discovered_at"),
                "fetched_at": dt.datetime.now(TZ).isoformat(),
                "fetch_status": "skipped",
                "extracted_text": None,
                "sublinks": []
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

            # If this is an SEC EDGAR index page (8-K, 10-Q, etc.), try to find and follow the Exhibit 99 link
            if "sec.gov" in link.lower() and "Archives/edgar" in link:
                exhibit_url = find_exhibit_from_index(html, link)
                if exhibit_url:
                    print(f"[SEC] Found exhibit: {exhibit_url}")
                    time.sleep(0.3)  # SEC asks for <10 req/sec, we do ~3/sec to stay well under limit
                    exhibit_resp = fetch_url(exhibit_url)
                    if exhibit_resp:
                        # Skip if it's a PDF
                        if "pdf" not in exhibit_resp.headers.get("Content-Type", "").lower():
                            html = exhibit_resp.text
                            link = exhibit_url  # Update link to point to the actual exhibit
                            print(f"[SEC] Using exhibit content from {exhibit_url}")
                        else:
                            print(f"[SEC] Exhibit is PDF, skipping: {exhibit_url}")
                    else:
                        print(f"[SEC] Failed to fetch exhibit: {exhibit_url}")

            # Text extraction main body
            text = extract_with_trafilatura(html, link)
            if text:
                item["fetch_status"] = "ok"
                item["extracted_text"] = text
            else:
                item["fetch_status"] = "extract_failed"
            
            # Find sublinks in the extracted text
            item["sublinks"].extend(find_sublinks_in_text(text or "", link))

            # Add sublink extraction (if any)
            if item["sublinks"]:
                for sublink in item["sublinks"]:
                    # Fetch the sublink content
                    sub_resp = fetch_url(sublink["url"])
                    if sub_resp:
                        subtext = extract_with_trafilatura(sub_resp.text, sublink["url"])
                    else:
                        subtext = None
                    sublink["extracted_text"] = subtext
                    sublink["fetch_status"] = "ok" if subtext else "extract_failed"
                    sublink["similarity_to_main"] = compute_similarity(text or "", subtext or "")
            # Compute a content hash of the normalized extracted text and update cache
            content_hash = None
            if text:
                try:
                    # Normalize whitespace and trim
                    normalized = " ".join(text.split())
                    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                except Exception:
                    content_hash = None

            if content_hash:
                # If we have seen this content before, avoid storing duplicate extracted text
                if not no_cache and content_hash in cache.get("content_meta", {}):
                    meta = cache["content_meta"][content_hash]
                    item["fetch_status"] = "duplicate_content_skipped"
                    item["extracted_text"] = None
                    item["duplicate_of"] = meta.get("uid")
                    item["content_hash"] = content_hash
                else:
                    # store metadata for this content so future identical pages can be skipped
                    if not no_cache:
                        cache.setdefault("content_meta", {})[content_hash] = {
                            "uid": uid,
                            "title": item.get("title"),
                            "source_id": item.get("source_id"),
                            "fetched_at": item.get("fetched_at")
                        }
                    item["content_hash"] = content_hash

                # Map html_hash -> content_hash for quicker detection next time
                if html_hash and not no_cache:
                    cache.setdefault("html_to_content", {})[html_hash] = content_hash

            # persist cache after each processed pointer (best-effort)
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
            # polite short delay
            time.sleep(delay)

    print(f"Processed {total} pointers, wrote {added} content records -> {out_path}")
    return added


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else POINTERS_PATH
    # You can set a custom delay here, e.g., delay=0.05 for faster processing
    main(path, delay=0.2)
