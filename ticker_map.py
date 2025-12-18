
import yaml
from pathlib import Path

def load_ticker_map(sources_path="sources.yaml"):
    """
    Loads sources.yaml and creates a mapping from source_id (e.g., 'msft_8k') 
    to a clean company name (e.g., 'Microsoft').
    """
    mapping = {}
    try:
        with open(sources_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            
        for feed in data.get("feeds", []):
            sid = feed.get("id")
            if not sid:
                continue
                
            # Heuristic: Extract ticker from ID (e.g., 'msft_8k' -> 'MSFT')
            # You can also add a 'company_name' field to your YAML in the future for better accuracy
            parts = sid.split('_')
            ticker = parts[0].upper()
            
            # Basic manual overrides for common ones if needed, 
            # otherwise default to Ticker
            company_name = ticker
            
            # If you want to map tickers to full names, you could add a dictionary here
            # e.g. {'MSFT': 'Microsoft', 'AAPL': 'Apple', ...}
            # For now, we'll just use the Ticker as the Company Name which is standard in finance
            
            mapping[sid] = company_name
            
    except Exception as e:
        print(f"[WARN] Failed to load ticker map: {e}")
        
    return mapping
