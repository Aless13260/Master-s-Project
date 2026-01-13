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
import argparse

TZ = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")
UA = "AgenticFinanceResearchBot/0.1 (contact: aless13260@gmail.com)"
MIN_CONTENT_LENGTH = 300  # Skip content shorter than this (likely image-only or empty)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POINTERS_PATH = PROJECT_ROOT / "ingestion_json" / "pointers.json"
OUT_PATH = PROJECT_ROOT / "ingestion_json" / "contents.jsonl"
CACHE_PATH = PROJECT_ROOT / "ingestion_json" / "fetch_cache.json"


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

def _strip_ix_viewer_wrapper(url: str) -> str:
    """Strip SEC's /ix?doc= inline XBRL viewer wrapper to get the direct document URL.
    
    e.g., https://www.sec.gov/ix?doc=/Archives/edgar/data/789019/000095017024132722/msft-ex99_1.htm
    becomes https://www.sec.gov/Archives/edgar/data/789019/000095017024132722/msft-ex99_1.htm
    """
    if "/ix?doc=" in url:
        # Extract the path from the doc= parameter
        match = re.search(r"/ix\?doc=([^&]+)", url)
        if match:
            doc_path = match.group(1)
            # Build absolute URL on sec.gov
            return f"https://www.sec.gov{doc_path}"
    return url


def find_exhibit_from_index(html: str, base_url: str) -> str | None:
    """Parse a filing index page and find the EX-99 exhibit URL.
    
    Returns:
        exhibit_url if EX-99 found, None otherwise
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        # 1) Parse the SEC "Document Format Files" table where the Type column is explicit.
        # Typical columns: Seq | Description | Document | Type | Size
        type_patterns = re.compile(r"^\s*EX-?99(\.|-|\s|$)", re.I)
        press_patterns = re.compile(r"(press[\s_-]*release|earnings|results)", re.I)

        best_exhibit: tuple[int, str] | None = None

        for table in soup.find_all("table"):
            summary = (table.get("summary") or "").strip().lower()
            if "document format files" not in summary and "documents" not in summary:
                continue
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) < 4:
                    continue

                # Heuristic: Document link is usually in the 3rd column.
                doc_td = tds[2]
                link = doc_td.find("a", href=True)
                if not link:
                    continue
                href = link["href"]
                abs_url = urljoin(base_url, href)
                # Strip /ix?doc= wrapper if present
                abs_url = _strip_ix_viewer_wrapper(abs_url)

                type_txt = (tds[3].get_text(" ") or "").strip()
                desc_txt = (tds[1].get_text(" ") or "").strip() if len(tds) > 1 else ""

                # Score EX-99 exhibits
                score = 0
                if type_patterns.search(type_txt):
                    # Prefer EX-99.* explicitly.
                    score += 100
                    # Slight preference for EX-99.1 variants.
                    if re.search(r"EX-?99\.(0?1|1)\b", type_txt, flags=re.I):
                        score += 10
                if press_patterns.search(desc_txt) or press_patterns.search(href):
                    score += 5

                if score > 0:
                    if best_exhibit is None or score > best_exhibit[0]:
                        best_exhibit = (score, abs_url)

        if best_exhibit is not None:
            return best_exhibit[1]

        # 2) Fallback: look for <a> tags with filenames or link text matching our patterns.
        patterns = re.compile(r"(ex-?99|exhibit\s*99|press[\s_-]*release|earnings|results)", re.I)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            txt = (a.get_text(" ") or "").strip()
            if patterns.search(href) or patterns.search(txt):
                result = urljoin(base_url, href)
                return _strip_ix_viewer_wrapper(result)
        
        return None
    except Exception:
        return None


def extract_pdf_links_from_html(html: str, base_url: str) -> list[str]:
    """Extract absolute PDF URLs from an HTML page.

    This is primarily used for SEC exhibit pages (e.g., EX-99.*) that sometimes
    embed or link to a PDF rather than containing meaningful HTML text.
    """
    pdf_links: list[str] = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[str] = []

        for a in soup.find_all("a", href=True):
            candidates.append(a["href"])

        for tag in soup.find_all(["embed", "object", "iframe"], src=True):
            candidates.append(tag["src"])

        for tag in soup.find_all("object", data=True):
            candidates.append(tag["data"])

        for raw in candidates:
            if not raw:
                continue
            url = urljoin(base_url, raw)
            url_lc = url.lower()
            if url_lc.split("?")[0].endswith(".pdf"):
                pdf_links.append(url)
                continue
            # Some links don't end with .pdf but are served as PDF downloads.
            if "pdf" in url_lc and ("/pdf" in url_lc or "format=pdf" in url_lc or "type=pdf" in url_lc):
                pdf_links.append(url)

    except Exception:
        return []

    # De-dupe, preserve order
    seen = set()
    uniq: list[str] = []
    for u in pdf_links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def main(
    pointers_path: Path = POINTERS_PATH,
    out_path: Path = OUT_PATH,
    only_candidates: bool = True,
    mark_skipped: bool = False,
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

    # guidance keywords used to decide whether to follow a pointer
    guidance_patterns = ("8-k", "8k", "press release", "press-release", "earnings", "results", "guidance", "outlook", "results")

    total = 0
    added = 0
    with out_path.open("a", encoding="utf-8") as out:
        for p in read_jsonl(pointers_path):
            if limit is not None and added >= limit:
                break
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
            if not no_cache and html_hash and html_hash in cache.get("html_to_content", {}):
                content_hash = cache["html_to_content"][html_hash]
                meta = cache.get("content_meta", {}).get(content_hash)
                if meta:
                    # Write a duplicate record pointing to existing content
                    item["fetch_status"] = "duplicate_html_skipped"
                    item["extracted_text"] = None
                    item["duplicate_of"] = meta.get("uid")
                    item["content_hash"] = content_hash
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    added += 1
                    seen.add(uid)
                    time.sleep(delay)
                    continue

            html = resp.text

            # If this is an SEC EDGAR index page (8-K, 10-Q, etc.), try to find and follow the Exhibit 99 link
            discovered_pdf_links: list[str] = []
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
                            item["link"] = exhibit_url
                            item["resolved_exhibit_url"] = exhibit_url
                            print(f"[SEC] Using exhibit content from {exhibit_url}")
                            discovered_pdf_links = extract_pdf_links_from_html(html, exhibit_url)
                        else:
                            # Exhibit itself is a PDF; record it so parse_pdfs can fetch it.
                            print(f"[SEC] Exhibit is PDF: {exhibit_url}")
                            item["link"] = exhibit_url
                            item["resolved_exhibit_url"] = exhibit_url
                            item["fetch_status"] = "skipped_pdf"
                            item["pdf_links"] = [exhibit_url]
                            out.write(json.dumps(item, ensure_ascii=False) + "\n")
                            added += 1
                            seen.add(uid)
                            time.sleep(delay)
                            continue
                    else:
                        print(f"[SEC] Failed to fetch exhibit: {exhibit_url}")

            # If the resolved page itself includes PDFs, keep them around for later PDF parsing.
            if not discovered_pdf_links:
                discovered_pdf_links = extract_pdf_links_from_html(html, link)
            if discovered_pdf_links:
                item["pdf_links"] = discovered_pdf_links

            # Text extraction main body
            text = extract_with_trafilatura(html, link)
            
            # Check for minimum content length to skip image-only or empty pages
            if text and len(text) < MIN_CONTENT_LENGTH:
                # If we discovered PDFs, keep the record so the PDF stage can pick it up.
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
                if item["fetch_status"] != "pdf_links_found":
                    item["fetch_status"] = "ok"
                item["extracted_text"] = text
            else:
                item["fetch_status"] = "extract_failed"
            
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

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            added += 1
            seen.add(uid)
            time.sleep(delay)

    # Persist cache once after processing
    if not no_cache:
        try:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=chosen_cache.parent) as tf:
                json.dump(cache, tf, ensure_ascii=False)
                tmpname = tf.name
            os.replace(tmpname, chosen_cache)
        except Exception:
            pass

    print(f"Processed {total} pointers, wrote {added} content records -> {out_path}")
    return added


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and extract text from pointer URLs using trafilatura (SEC-aware).")
    parser.add_argument("--pointers", type=str, default=str(POINTERS_PATH), help="Path to pointers.json (JSONL).")
    parser.add_argument("--out", type=str, default=str(OUT_PATH), help="Output JSONL path for extracted content.")
    parser.add_argument("--only-candidates", action="store_true", default=True, help="Only process guidance candidates.")
    parser.add_argument("--all", dest="only_candidates", action="store_false", help="Process all pointers.")
    parser.add_argument("--mark-skipped", action="store_true", default=False, help="Write skipped_not_candidate records.")
    parser.add_argument("--delay", type=float, default=0.15, help="Polite delay between requests.")
    parser.add_argument("--no-cache", action="store_true", default=False, help="Disable fetch cache.")
    parser.add_argument("--cache-path", type=str, default=str(CACHE_PATH), help="Cache JSON path.")
    parser.add_argument("--reprocess", action="store_true", default=False, help="Do not skip already-seen UIDs in the output.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after writing N records.")
    args = parser.parse_args()

    main(
        pointers_path=Path(args.pointers),
        out_path=Path(args.out),
        only_candidates=args.only_candidates,
        mark_skipped=args.mark_skipped,
        delay=args.delay,
        no_cache=args.no_cache,
        cache_path=Path(args.cache_path) if args.cache_path else None,
        reprocess=args.reprocess,
        limit=args.limit,
    )
