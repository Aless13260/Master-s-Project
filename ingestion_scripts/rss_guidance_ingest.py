import sys, json, feedparser, yaml, datetime as dt, zoneinfo, hashlib, argparse
from pathlib import Path
from typing import Iterator, Dict, Any, List, Set
import requests
TZ = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ────────────────────────────── Config ──────────────────────────────
def load_feeds(path: str = str(PROJECT_ROOT / "sources.yaml")) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [f for f in data.get("feeds", []) if f.get("id") and f.get("url")]

 
UA = "AgenticFinanceResearchBot/0.1 (contact: aless13260@gmail.com)"

def fetch_feed(url: str) -> bytes | None:
    """Fetch RSS feed bytes with polite headers & retries."""
    try:
        headers = {
            "User-Agent": UA,
            "Accept": "application/rss+xml, application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"[WARN] Could not fetch {url}: {e}")
        return None

# ────────────────────────────── Pointer event ───────────────────────
def make_uid(parts: List[str]) -> str:
    s = "\x1f".join(p or "" for p in parts)  # unit-separator join
    return hashlib.sha256(s.encode()).hexdigest()[:40]

def entry_to_pointer(source_id: str, e: Dict[str, Any], guidance_candidate: bool = False) -> Dict[str, Any]:
    title     = e.get("title") or ""
    link      = e.get("link") or e.get("id") or ""
    published = e.get("published") or e.get("updated") or None
    uid       = make_uid([source_id, title, link, published or ""])
    return {
        "uid": uid,
        "source_id": source_id,
        "title": title,
        "link": link,
        "published_at": published,                 # when publisher says it went live
        "discovered_at": dt.datetime.now(TZ).isoformat(),  # when WE saw it
        "status": "pending",                       # to be consumed by the fetcher
        "guidance_candidate": guidance_candidate,
        # optional tiny hint payloads (kept small):
        "summary_hint": e.get("summary") or e.get("description") or None,
        "categories": [t.get("term") for t in e.get("tags", [])] if e.get("tags") else None,
    }

def iter_pointer_events(sources_path: str = "./sources.yaml", allow_all: bool = False) -> Iterator[Dict[str, Any]]:
    for src in load_feeds(sources_path):
        sid, url = src["id"], src["url"]
        print(f"[DEBUG] Fetching feed: {sid} -> {url}")
        content = fetch_feed(url)
        if not content:
            continue
        parsed = feedparser.parse(content)
        # Only keep entries likely to contain guidance language or press releases (unless allow_all)
        guidance_patterns = ("8-k", "8k", "press release", "press-release", "earnings", "results", "guidance", "outlook")
        # Common filing types we want to ignore (ownership/insider filings, etc.)
        skip_types = ("3", "4", "5", "13g", "13d", "144", "schedule 13g", "schedule 13d", "form 3", "form 4", "form 5", "ownership", "statement of changes", "statement of changes in beneficial ownership")
        # High-value 8-K items that contain earnings/guidance
        valuable_8k_items = ("item 2.02", "item 7.01", "item 8.01")

        for e in (parsed.entries or []):
            title_raw = (e.get("title") or "")
            title = title_raw.lower()
            summary = (e.get("summary") or e.get("description") or "").lower()
            link = e.get("link") or e.get("id")
            if not link:
                continue

            # quick filing-type check: many SEC titles start with the form type (e.g., "4 - ...")
            filing_token = ""
            if "-" in title_raw:
                filing_token = title_raw.split("-", 1)[0].strip().lower()
            else:
                # fallback: first word
                filing_token = title.split()[0] if title.split() else ""

            # Check if this is an 8-K filing from a feed we added
            is_8k_feed = sid.endswith("_8k")
            is_8k_filing = "8-k" in title.lower() or filing_token == "8-k"
            
            # Check if this 8-K has valuable items (earnings, guidance, material events)
            has_valuable_item = any(item in summary for item in valuable_8k_items)

            # Determine if this should be marked as a guidance candidate
            is_guidance_candidate = False
            
            if not allow_all:
                if filing_token in skip_types or any(st in title or st in summary for st in skip_types):
                    # definitely skip ownership/insider filings and other noisy types
                    continue

                # Auto-pass ALL 8-K filings from dedicated 8-K feeds
                if is_8k_feed:
                    is_guidance_candidate = True
                # Or if it's an 8-K with valuable items
                elif is_8k_filing and has_valuable_item:
                    is_guidance_candidate = True
                # Otherwise require at least one positive guidance-like pattern
                elif any(p in title or p in summary for p in guidance_patterns):
                    is_guidance_candidate = True
                else:
                    # Skip entries that don't match any criteria
                    continue
            else:
                # allow_all mode: accept everything but still mark 8-Ks as guidance candidates
                is_guidance_candidate = is_8k_feed or is_8k_filing

            yield entry_to_pointer(sid, e, guidance_candidate=is_guidance_candidate)

# ────────────────────────────── Main (produce a queue) ──────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="?", default=str(PROJECT_ROOT / "sources.yaml"), help="Path to sources.yaml")
    parser.add_argument("--allow-all", action="store_true", help="Do not filter entries by guidance keywords or skip types (use for analysis)")
    args = parser.parse_args()
    sources_path = args.sources
    
    POINTERS_8K = PROJECT_ROOT / "ingestion_json" / "pointers_8k.json"
    POINTERS_IR = PROJECT_ROOT / "ingestion_json" / "pointers_IR.json"

    seen: Set[str] = set()

    # Load existing pointer records from BOTH files
    for path in [POINTERS_8K, POINTERS_IR]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        ln = line.strip()
                        if not ln:
                            continue
                        try:
                            obj = json.loads(ln)
                            uid = obj.get("uid")
                            if uid:
                                seen.add(uid)
                        except Exception:
                            continue
            except Exception as e:
                print(f"[WARN] Error reading {path}: {e}")

    # iterate feeds and collect new events
    new_events_8k = []
    new_events_ir = []
    
    print("Fetching feeds...")
    for ev in iter_pointer_events(sources_path, allow_all=args.allow_all):
        if ev["uid"] in seen:
            continue
        seen.add(ev["uid"])
        
        # Determine destination based on content
        title = (ev.get("title") or "").lower()
        source_id = (ev.get("source_id") or "").lower()
        categories = [c.lower() for c in ev.get("categories") or []]
        
        is_8k = "8-k" in title or "8k" in source_id or any("8-k" in c for c in categories)
        
        if is_8k:
            new_events_8k.append(ev)
        else:
            new_events_ir.append(ev)

    # Append to files
    if new_events_8k:
        with open(POINTERS_8K, "a", encoding="utf-8") as out:
            for ev in new_events_8k:
                out.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print(f"Wrote {len(new_events_8k)} new 8-K events → {POINTERS_8K}")
        
    if new_events_ir:
        with open(POINTERS_IR, "a", encoding="utf-8") as out:
            for ev in new_events_ir:
                out.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print(f"Wrote {len(new_events_ir)} new IR events → {POINTERS_IR}")
        
    if not new_events_8k and not new_events_ir:
        print("No new events found.")

