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
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import requests
import trafilatura
import datetime as dt
import zoneinfo

TZ = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")
UA = "AgenticFinanceResearchBot/0.1 (contact: aless13260@gmail.com)"

POINTERS_PATH = Path("pointerEvents") / "pointers.json"
OUT_PATH = Path("pointerEvents") / "contents.jsonl"


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
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        return None


def extract_with_trafilatura(html: str, url: str) -> str | None:
    try:
        result = trafilatura.extract(html, url=url, include_comments=False, include_tables=False)
        return result
    except Exception:
        return None


def find_exhibit_from_index(html: str, base_url: str) -> str | None:
    """Parse a filing index page and try to find the first exhibit URL that looks like a press release or EX-99."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # look for <a> tags with filenames or link text matching our patterns
        patterns = re.compile(r"(ex-?99|exhibit\s*99|press[\s_-]*release|earnings|results|primary|ex-?101)", re.I)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            txt = (a.get_text(" ") or "").strip()
            if patterns.search(href) or patterns.search(txt):
                # build absolute URL
                return urljoin(base_url, href)

    except Exception:
        return None
    return None


def main(pointers_path: Path = POINTERS_PATH, out_path: Path = OUT_PATH, only_candidates: bool = True, mark_skipped: bool = False) -> int:
    seen = set()
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
                        "extracted_text": None,
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

            html = resp.text

            # If we fetched a filing index page (typical EDGAR landing ending with -index.htm),
            # attempt to find an exhibit (EX-99 / press release / primary document) and follow it.
            if link.lower().endswith("-index.htm") or "-index.htm" in link.lower():
                exhibit_url = find_exhibit_from_index(html, resp.url)
                if exhibit_url and exhibit_url != resp.url:
                    # try fetching the exhibit
                    ex_resp = fetch_url(exhibit_url)
                    if ex_resp and "pdf" not in (ex_resp.headers.get("Content-Type", "") or ""):
                        html = ex_resp.text
                        link = exhibit_url

            text = extract_with_trafilatura(html, link)
            if text:
                item["fetch_status"] = "ok"
                item["extracted_text"] = text
            else:
                item["fetch_status"] = "extract_failed"

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            added += 1
            seen.add(uid)
            # polite short delay
            time.sleep(0.2)

    print(f"Processed {total} pointers, wrote {added} content records → {out_path}")
    return added


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else POINTERS_PATH
    main(path)
