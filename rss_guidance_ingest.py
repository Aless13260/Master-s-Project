import sys, json, feedparser, yaml, datetime as dt, zoneinfo, hashlib, argparse
from typing import Iterator, Dict, Any, List, Set
import requests
TZ = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")

# ────────────────────────────── Config ──────────────────────────────
def load_feeds(path: str = "./sources.yaml") -> List[Dict[str, Any]]:
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

            if not allow_all:
                if filing_token in skip_types or any(st in title or st in summary for st in skip_types):
                    # definitely skip ownership/insider filings and other noisy types
                    continue

                # now require at least one positive guidance-like pattern
                if not any(p in title or p in summary for p in guidance_patterns):
                    continue

            yield entry_to_pointer(sid, e)

# ────────────────────────────── Main (produce a queue) ──────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="?", default="./sources.yaml", help="Path to sources.yaml")
    parser.add_argument("--allow-all", action="store_true", help="Do not filter entries by guidance keywords or skip types (use for analysis)")
    args = parser.parse_args()
    sources_path = args.sources
    out_path     = r"C:\Users\aless\Github\FinanceProject\pointerEvents\pointers.json"

    seen: Set[str] = set()

    # Load existing pointer records (if any)
    existing_records: List[Dict[str, Any]] = []
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                    existing_records.append(obj)
                except Exception:
                    continue
    except FileNotFoundError:
        existing_records = []

    # Detect sources removed from sources.yaml and prune their records
    current_source_ids = {s["id"] for s in load_feeds(sources_path)}
    existing_source_ids = {r.get("source_id") for r in existing_records if r.get("source_id")}
    deleted_sources = existing_source_ids - current_source_ids
    if deleted_sources:
        print(f"[INFO] Detected deleted sources: {deleted_sources}. Pruning corresponding records.")
        # Rewrite pointers file without events from deleted sources
        remaining = [r for r in existing_records if r.get("source_id") not in deleted_sources]
        with open(out_path, "w", encoding="utf-8") as f:
            for r in remaining:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Also prune contents.jsonl (if present) for matching UIDs
        contents_path = out_path.replace("pointers.json", "contents.jsonl")
        try:
            kept_lines: List[str] = []
            with open(contents_path, "r", encoding="utf-8") as cf:
                for ln in cf:
                    ln_strip = ln.strip()
                    if not ln_strip:
                        continue
                    try:
                        obj = json.loads(ln_strip)
                        if obj.get("source_id") in deleted_sources:
                            continue
                        kept_lines.append(ln_strip)
                    except Exception:
                        # keep unparsable lines
                        kept_lines.append(ln_strip)
            with open(contents_path, "w", encoding="utf-8") as cf:
                for ln in kept_lines:
                    cf.write(ln + "\n")
        except FileNotFoundError:
            pass

        existing_records = remaining

    # Build seen set from remaining existing records to avoid duplicates
    for r in existing_records:
        uid = r.get("uid")
        if uid:
            seen.add(uid)

    # iterate feeds and collect new events into memory first, then append to file
    new_events = []
    for ev in iter_pointer_events(sources_path, allow_all=args.allow_all):
        if ev["uid"] in seen:
            continue
        seen.add(ev["uid"])
        new_events.append(ev)

    # append to a JSONL queue file in one go
    new_count = 0
    if new_events:
        with open(out_path, "a", encoding="utf-8") as out:
            for ev in new_events:
                out.write(json.dumps(ev, ensure_ascii=False) + "\n")
                new_count += 1
    print(f"Wrote {new_count} new pointer events → {out_path}")

