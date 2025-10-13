import sys, json, feedparser, yaml, datetime as dt, zoneinfo, hashlib
from typing import Iterator, Dict, Any, List, Set

TZ = zoneinfo.ZoneInfo("Asia/Kuala_Lumpur")

# ────────────────────────────── Config ──────────────────────────────
def load_feeds(path: str = "./sources.yaml") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [f for f in data.get("feeds", []) if f.get("id") and f.get("url")]

# ────────────────────────────── Pointer event ───────────────────────
def make_uid(parts: List[str]) -> str:
    s = "\x1f".join(p or "" for p in parts)  # unit-separator join
    return hashlib.sha256(s.encode()).hexdigest()[:40]

def entry_to_pointer(source_id: str, e: Dict[str, Any]) -> Dict[str, Any]:
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
        # optional tiny hint payloads (kept small):
        "summary_hint": e.get("summary") or e.get("description") or None,
        "categories": [t.get("term") for t in e.get("tags", [])] if e.get("tags") else None,
    }

def iter_pointer_events(sources_path: str = "./sources.yaml") -> Iterator[Dict[str, Any]]:
    for src in load_feeds(sources_path):
        sid, url = src["id"], src["url"]
        parsed = feedparser.parse(url)
        for e in (parsed.entries or []):
            link = e.get("link") or e.get("id")
            if not link:
                continue
            yield entry_to_pointer(sid, e)

# ────────────────────────────── Main (produce a queue) ──────────────
if __name__ == "__main__":
    sources_path = sys.argv[1] if len(sys.argv) > 1 else "./sources.yaml"
    out_path     = r"C:\Users\aless\Github\FinanceProject\pointerEvents\pointers.json"

    seen: Set[str] = set()  # in-run dedupe; 
    if out_path:
        # append to a JSONL queue file
        with open(out_path, "a", encoding="utf-8") as out:
            for ev in iter_pointer_events(sources_path):
                if ev["uid"] in seen: 
                    continue
                seen.add(ev["uid"])
                out.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print(f"Wrote {len(seen)} pointer events → {out_path}")
    else:
        # print to stdout (pipe into the next stage if you like)
        for ev in iter_pointer_events(sources_path):
            if ev["uid"] in seen: 
                continue
            seen.add(ev["uid"])
            print(json.dumps(ev, ensure_ascii=False))
