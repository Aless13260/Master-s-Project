import sqlite3
import json
from pathlib import Path
import sys
from uuid import uuid4
import argparse
import hashlib

# Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "finance_data.db"
CONTENTS_PATH = BASE_DIR / "ingestion_json" / "contents.jsonl"
CANDIDATES_PATH = BASE_DIR / "extractor_lib" / "candidate_guidance.jsonl"
GUIDANCE_PATH = BASE_DIR / "extractor_lib" / "extracted_guidance.jsonl"
REASONING_GUIDANCE_PATH = BASE_DIR / "extractor_lib" / "extracted_guidance_reasoning.jsonl"

def drop_tables(conn):
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS guidance")
    c.execute("DROP TABLE IF EXISTS candidates")
    c.execute("DROP TABLE IF EXISTS contents")
    conn.commit()
    print("Dropped existing tables.")

def create_tables(conn):
    c = conn.cursor()
    
    # 1. Contents Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS contents (
            uid TEXT PRIMARY KEY,
            source_id TEXT,
            title TEXT,
            link TEXT,
            published_at TEXT,
            fetched_at TEXT,
            fetch_status TEXT,
            extracted_text TEXT,
            sublinks JSON
        )
    ''')

    # 2. Candidates Table (Filtered items)
    c.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            uid TEXT PRIMARY KEY,
            source_id TEXT,
            title TEXT,
            match_score REAL,
            confidence REAL,
            matched_patterns TEXT, -- Stored as JSON string
            extracted_text_preview TEXT,
            FOREIGN KEY(uid) REFERENCES contents(uid)
        )
    ''')

    # 3. Guidance Table (Final extracted data)
    # We flatten the 'guidance' object here for easier SQL querying
    c.execute('''
        CREATE TABLE IF NOT EXISTS guidance (
            guid TEXT PRIMARY KEY,
            content_uid TEXT,
            source_id TEXT,
            company TEXT,
            guidance_type TEXT,
            metric_name TEXT,
            reporting_period TEXT,
            current_value REAL,
            unit TEXT,
            guided_range_low REAL,
            guided_range_high REAL,
            is_revision BOOLEAN,
            revision_direction TEXT,
            qualitative_direction TEXT,
            rationales TEXT,
            statement_text TEXT,
            source_type TEXT,
            extracted_at TEXT,
            extraction_method TEXT,
            processing_duration_seconds REAL,
            was_updated_by_agent BOOLEAN,
            FOREIGN KEY(content_uid) REFERENCES contents(uid)
        )
    ''')
    conn.commit()
    print("Tables created successfully.")

def migrate_contents(conn):
    if not CONTENTS_PATH.exists():
        print(f"Skipping contents (not found): {CONTENTS_PATH}")
        return

    c = conn.cursor()
    count = 0
    with open(CONTENTS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                c.execute('''
                    INSERT OR REPLACE INTO contents (
                        uid, source_id, title, link, published_at, fetched_at, 
                        fetch_status, extracted_text, sublinks
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('uid'),
                    data.get('source_id'),
                    data.get('title'),
                    data.get('link'),
                    data.get('published_at'),
                    data.get('fetched_at'),
                    data.get('fetch_status'),
                    data.get('extracted_text'),
                    json.dumps(data.get('sublinks', [])) # Store list as JSON string
                ))
                count += 1
            except Exception as e:
                print(f"Error inserting content: {e}")
    
    conn.commit()
    print(f"Migrated {count} content records.")

def migrate_candidates(conn):
    if not CANDIDATES_PATH.exists():
        print(f"Skipping candidates (not found): {CANDIDATES_PATH}")
        return

    c = conn.cursor()
    count = 0
    with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                c.execute('''
                    INSERT OR REPLACE INTO candidates (
                        uid, source_id, title, match_score, confidence, 
                        matched_patterns, extracted_text_preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.get('uid'),
                    data.get('source_id'),
                    data.get('title'),
                    data.get('match_score'),
                    data.get('confidence'),
                    json.dumps(data.get('matched_patterns', [])),
                    data.get('extracted_text_preview')
                ))
                count += 1
            except Exception as e:
                print(f"Error inserting candidate: {e}")
    
    conn.commit()
    print(f"Migrated {count} candidate records.")

def generate_deterministic_id(data_dict):
    """Generate a consistent ID based on the content of the guidance."""
    # Create a unique string signature for this item
    # We use content_uid + metric + value + statement to ensure uniqueness
    # Also include extraction_method to differentiate between standard and reasoning versions of the same item
    unique_str = f"{data_dict.get('content_uid')}|{data_dict.get('metric_name')}|{data_dict.get('guidance_type')}|{data_dict.get('statement_text')}|{data_dict.get('extraction_method')}"
    return hashlib.md5(unique_str.encode('utf-8')).hexdigest()

def migrate_guidance(conn, file_path):
    if not file_path.exists():
        print(f"Skipping guidance (not found): {file_path}")
        return

    c = conn.cursor()
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                row = json.loads(line)
                # The actual guidance data is nested under "guidance" key
                g = row.get('guidance', {})
                
                # Use deterministic ID to prevent duplicates on re-runs
                # This fixes both the "guid_1" collision AND allows safe appending
                row_uid = row.get('uid')
                extraction_method = g.get('extraction_method', 'standard')
                
                guid_id = generate_deterministic_id({
                    'content_uid': row_uid,
                    'metric_name': g.get('metric_name'),
                    'guidance_type': g.get('guidance_type'),
                    'statement_text': g.get('statement_text'),
                    'extraction_method': extraction_method
                })

                c.execute('''
                    INSERT OR REPLACE INTO guidance (
                        guid, content_uid, source_id, company, guidance_type, metric_name,
                        reporting_period, current_value, unit, 
                        guided_range_low, guided_range_high, 
                        is_revision, revision_direction, qualitative_direction, rationales,
                        statement_text, source_type, extracted_at,
                        extraction_method, processing_duration_seconds, was_updated_by_agent
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    guid_id,
                    row_uid,  # content_uid comes from the top level 'uid'
                    row.get('source_id'),
                    g.get('company'),
                    g.get('guidance_type'),
                    g.get('metric_name'),
                    g.get('reporting_period'),
                    g.get('current_value'),
                    g.get('unit'),
                    g.get('guided_range_low'),
                    g.get('guided_range_high'),
                    g.get('is_revision'),
                    g.get('revision_direction'),
                    g.get('qualitative_direction'),
                    g.get('rationales'),
                    g.get('statement_text'),
                    g.get('source_type'),
                    g.get('extracted_at'),
                    g.get('extraction_method', 'standard'),  # Default to 'standard' if missing
                    g.get('processing_duration_seconds'),
                    g.get('was_updated_by_agent', False)
                ))
                count += 1
            except Exception as e:
                print(f"Error inserting guidance: {e}")
    
    conn.commit()
    print(f"Migrated {count} guidance records from {file_path.name}.")
def main():
    parser = argparse.ArgumentParser(description="Migrate JSONL data to SQLite")
    parser.add_argument("--refresh", action="store_true", help="Drop existing tables and start fresh")
    args = parser.parse_args()

    print(f"Creating/Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        if args.refresh:
            print("Refreshing database (dropping tables)...")
            drop_tables(conn)

        create_tables(conn)
        migrate_contents(conn)
        migrate_candidates(conn)
        migrate_guidance(conn, GUIDANCE_PATH)
        migrate_guidance(conn, REASONING_GUIDANCE_PATH)
        print("\nMigration Complete!")
        print(f"You can now open '{DB_PATH.name}' with any SQLite viewer.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
