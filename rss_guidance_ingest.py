# Ingestion from RSS feeds
# rss_guidance_ingest.py

import feedparser
import yaml
import hashlib
import datetime as dt
import zoneinfo
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

# ────────────────────────────── Utilities ──────────────────────────────
def make_uid(parts: List[str]) -> str:
    """Create a stable short hash for deduplication."""
    s = "|".join(p or "" for p in parts)
    return hashlib.sha256(s.encode()).hexdigest()[:40]

def now_iso() -> str:
    tz = zoneinfo.ZoneInfo(os.getenv("TIMEZONE", "Asia/Kuala_Lumpur"))
    return dt.datetime.now(tz=tz).isoformat()

# ────────────────────────────── Core logic ──────────────────────────────
def load_feeds(path: str = "./sources.yaml") -> List[Dict[str, Any]]:
    """Read list of RSS feeds from sources.yaml"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("feeds", [])

def entry_to_guidance(source_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one RSS entry → GuidanceEvent dict"""
    title = entry.get("title", "")
    link = entry.get("link") or entry.get("id")
    summary = entry.get("summary") or entry.get("description", "")
    published = entry.get("published") or entry.get("updated")

    uid = make_uid([source_id, title, link or "", published or ""])
    return {
        "uid": uid,
        "source_id": source_id,
        "title": title,
        "link": link,
        "summary": summary,
        "published_at": published,
        "status": "pending",
        "fetched_at": None,
        "discovered_at": now_iso(),
    }

def ingest_guidance(sources_path: str = "./sources.yaml") -> List[Dict[str, Any]]:
    """Parse all configured RSS feeds into guidance events"""
    feeds = load_feeds(sources_path)
    all_events = []

    for src in feeds:
        sid = src["id"]
        url = src["url"]
        print(f"Fetching {sid} → {url}")
        parsed = feedparser.parse(url)
        entries = parsed.entries or []
        for e in entries:
            all_events.append(entry_to_guidance(sid, e))
    return all_events

# ────────────────────────────── Main runner ──────────────────────────────
if __name__ == "__main__":
    load_dotenv()
    events = ingest_guidance()
    print(f"\nGenerated {len(events)} guidance events.")
    print("Sample:", events[0] if events else "None")