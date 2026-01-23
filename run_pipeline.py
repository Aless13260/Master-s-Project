import subprocess
import sys
import argparse
from pathlib import Path

def run_step(command, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD: {command}")
    print(f"{'='*60}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Step failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the complete Finance Guidance Extraction Pipeline")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip RSS ingestion and HTML parsing")
    parser.add_argument("--skip-filter", action="store_true", help="Skip regex filtering")
    parser.add_argument("--skip-extract", action="store_true", help="Skip LLM extraction")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced mode for extraction (agentic period normalization)")
    parser.add_argument("--max-items", type=int, help="Limit number of items for LLM extraction (testing)")
    parser.add_argument("--refresh-db", action="store_true", help="Drop and recreate SQLite tables")
    
    args = parser.parse_args()
    
    python_exe = sys.executable

    # 1. Ingestion (RSS -> HTML)
    if not args.skip_ingest:
        # A. RSS Ingest
        run_step(f'"{python_exe}" ingestion_scripts/rss_guidance_ingest.py', "Ingesting RSS Feeds")
        
        # B. HTML Parse
        run_step(f'"{python_exe}" ingestion_scripts/web_parse_trafilatura.py', "Parsing HTML Content")

    # 2. Filtering (Content -> Candidates)
    if not args.skip_filter:
        run_step(f'"{python_exe}" extractor_lib/regex_filter.py', "Filtering Candidates (Regex)")

    # 3. Extraction (Candidates -> JSONL)
    if not args.skip_extract:
        cmd = f'"{python_exe}" extractor_lib/LLM_extractor.py'
        if args.enhanced:
            cmd += " --enhanced"
        if args.max_items:
            cmd += f" --max-items {args.max_items}"
        
        run_step(cmd, "Running LLM Extraction")

    # 4. Storage (JSONL -> SQLite)
    cmd = f'"{python_exe}" migrate_to_sqlite.py'
    if args.refresh_db:
        cmd += " --refresh"
    if args.enhanced:
        cmd += " --enhanced"
    run_step(cmd, "Migrating to SQLite Database")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
