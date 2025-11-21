
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from extractor_lib.LLM_extractor import LLMExtractor

def test_compound_extraction():
    # Mock text with compound guidance
    text = """
    Fiscal 2025 Outlook
    
    For the full fiscal year 2025, the Company expects total revenue to be in the range of $10.5 billion to $10.7 billion, representing growth of 5% to 7%. 
    We also anticipate GAAP earnings per share of $3.50 to $3.60 and an operating margin of approximately 22%.
    Capital expenditures are expected to be around $500 million.
    """
    
    print("Testing extraction on compound text...")
    
    # Initialize extractor (using real LLM call)
    extractor = LLMExtractor(provider="github", model="gpt-4o-mini")
    
    # Run extraction
    items = extractor.extract_from_text(text)
    
    print(f"\nExtracted {len(items)} items:")
    for item in items:
        print(f"- {item.guidance_type} ({item.metric_name}): {item.guided_value or item.guided_range_low}-{item.guided_range_high} {item.current_unit}")

if __name__ == "__main__":
    test_compound_extraction()
