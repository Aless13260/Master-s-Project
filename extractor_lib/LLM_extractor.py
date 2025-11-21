"""
LLM-based guidance extractor using structured output.
Extracts financial guidance from documents using the Guidance schema.

Main script to run LLM extraction on filtered candidate guidance.
Reads: extractor_lib/candidate_guidance.jsonl (filtered candidates)
       pointerEvents/contents.jsonl (full text content)
Writes: extractor_lib/extracted_guidance.jsonl
"""

import json
import sys
import re
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from llama_index.core.llms import ChatMessage
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field

# Import your LLM setup and guidance schema
sys.path.append(str(Path(__file__).parent.parent))
from llm_setup import setup_llm
from extractor_lib.guidance_schema import Guidance


# File paths
CANDIDATES_PATH = Path("extractor_lib") / "candidate_guidance.jsonl"
CONTENTS_PATH = Path("pointerEvents") / "contents.jsonl"
OUTPUT_PATH = Path("extractor_lib") / "extracted_guidance.jsonl"

# Guidance keywords for smart text extraction (from regex_filter)
GUIDANCE_KEYWORDS = [
    r'\b(expect|expects|expected|expecting|anticipat\w*|forecast\w*|project\w*|guidance)\b',
    r'\b(outlook|target\w*|goal\w*|aim\w*)\b',
]



# Use the existing Guidance schema for extraction
GuidanceExtraction = Guidance


class MultiGuidanceExtraction(BaseModel):
    """Container for multiple guidance items in a document."""
    guidance_items: List[GuidanceExtraction] = Field(
        description="List of all financial guidance items found in the document"
    )


class LLMExtractor:
    """Extract financial guidance using an LLM with structured output."""
    
    def __init__(self, provider: str = "github", model: str = None, temperature: float = 0.0, max_retries: int = 3):
        """
        Initialize the extractor.
        
        Args:
            provider: LLM provider ("github", "openai", etc.)
            model: Specific model to use (or None for defaults, uses gpt-4o-mini)
            temperature: 0.0 for deterministic extraction
            max_retries: Number of retry attempts for failed LLM calls
        """
        # store config so we can reconfigure on errors (e.g., rate limits)
        self.provider = provider
        self.model = model or "gpt-4o-mini"
        self.temperature = temperature
        self.llm = setup_llm(provider=self.provider, model=self.model, temperature=self.temperature)
        self.max_retries = max_retries

        # System prompt for extraction
        # NOTE: the field names below match the Guidance pydantic model in
        # extractor_lib.guidance_schema so the LLM's structured output can be
        # parsed directly into that model. We intentionally omit 'ticker' to
        # keep the model simpler; ticker can be mapped later from other sources.
        self.system_prompt = """
        You are a financial analyst specialized in extracting forward-looking financial guidance from corporate filings, earnings releases, and press statements.

        CRITICAL: Only extract FORWARD-LOOKING guidance. DO NOT extract historical results or past performance.

        EXAMPLES OF FORWARD-LOOKING (extract these):
        ✓ "expects revenue to be $50-52 billion for Q2"
        ✓ "guidance for full year EPS of $5.00 to $5.50"
        ✓ "anticipates operating margin of approximately 30%"
        ✓ "targeting 10% growth in FY2026"

        HANDLING MULTIPLE ITEMS (CRITICAL):
        If a single sentence contains multiple metrics (e.g., "Revenue of $10B and EPS of $2.00"), you MUST create separate JSON objects for each metric.
        - Object 1: Revenue, $10B
        - Object 2: EPS, $2.00
        Do NOT combine them. Do NOT skip items. Extract EVERY distinct forward-looking metric found.

        EXAMPLES OF HISTORICAL (DO NOT extract these):
        ✗ "revenue was $281.7 billion and increased 15%"
        ✗ "net income was $101.8 billion"
        ✗ "reported earnings of $13.64 per share"
        ✗ "operating income increased 17%"
        ✗ "fiscal year ended June 30, 2025 results"

        Look for forward-looking verbs: expects, guidance, outlook, forecast, projects, anticipates, targets, will be, plans to
        REJECT past-tense verbs: was, were, reported, announced, increased, decreased, grew, ended, posted

        For each distinct FORWARD-LOOKING guidance item, create a JSON object with these exact fields:
        - company: Company name (string or null)
        - guidance_type: MUST be one of: "revenue", "earnings", "EPS", "opex", "margin", "cash_flow", "other" (or null)
        - metric_name: The specific metric name mentioned in text (e.g. "Net Interest Income", "Cloud Revenue", "Organic Growth") (string or null)
        - reporting_period: The reporting period referenced (e.g., "Q2 2025", "FY2025") (or null)
        - current_value: Current/most-recent numeric value (number or null)
        - current_unit: MUST be one of: "USD", "EUR", "GBP", "%", "million", "billion", "units", "other" (or null)
        - guided_value: Guided numeric value (number or null)
        - guided_range_low: If a numeric range is provided, low end (number or null)
        - guided_range_high: If a numeric range is provided, high end (number or null)
        - change_pct: Percent change implied (number or null)
        - is_quantitative: true if guidance contains numeric guidance, false otherwise

        Do NOT extract historical results. Do NOT return past performance data.
        """
    
    def _extract_relevant_sections(self, text: str, context_chars: int = 500) -> str:
        """
        Smart extraction: Find text sections containing guidance keywords.
        Instead of processing entire document, focus on relevant paragraphs.
        
        Args:
            text: Full document text
            context_chars: Characters to include around each match
            
        Returns:
            Focused text containing guidance-relevant sections
        """
        relevant_sections = []
        seen_ranges = set()
        
        # Find all matches for guidance keywords
        # IMPORTANT: Focus heavily on text AFTER guidance keywords (forward-looking)
        # rather than historical results that appear before them.
        # Give minimal preceding context, but substantial trailing context.
        pre_context = 100  # Just enough to capture the intro phrase
        post_context = 600  # Focus on what comes AFTER the guidance keyword
        for pattern in GUIDANCE_KEYWORDS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, match.start() - pre_context)
                end = min(len(text), match.end() + post_context)
                
                # Avoid duplicate overlapping sections
                range_key = (start // 100, end // 100)  # Bucket by 100-char blocks
                if range_key not in seen_ranges:
                    relevant_sections.append((start, end, text[start:end]))
                    seen_ranges.add(range_key)
        
        # Sort by position and merge overlapping sections
        relevant_sections.sort()
        merged = []
        for start, end, section in relevant_sections:
            if merged and start <= merged[-1][1]:
                # Merge overlapping sections
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]), text[merged[-1][0]:max(end, merged[-1][1])])
            else:
                merged.append((start, end, section))
        
        # Join sections with clear separators
        focused_text = "\n\n[...]\n\n".join(section for _, _, section in merged)
        
        if focused_text:
            reduction = len(text) - len(focused_text)
            print(f"  [SMART] Focused on {len(merged)} sections (reduced by {reduction:,} chars)")
            return focused_text
        
        # If no matches, return beginning (shouldn't happen with our pre-filtered candidates)
        return text[:15000]
    
    def extract_from_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Guidance]:
        """
        Extract guidance items from raw text with smart section extraction.
        
        Args:
            text: The document text to analyze
            metadata: Optional metadata (company, source_url, etc.)
            
        Returns:
            List of Guidance objects
        """
        if not text or len(text.strip()) < 50:
            print("  [WARN] Text too short for extraction")
            return []
        
        # Smart extraction: focus on relevant sections
        focused_text = self._extract_relevant_sections(text)
        
        # Final length check (gpt-4o-mini handles ~8k tokens, we'll use ~4k tokens = ~16k chars max)
        max_chars = 16000
        if len(focused_text) > max_chars:
            print(f"  [WARN] Truncating focused text from {len(focused_text)} to {max_chars} chars")
            focused_text = focused_text[:max_chars] + "\n... [truncated]"
        
        # Retry logic for LLM calls
        result = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Create structured extraction program
                program = LLMTextCompletionProgram.from_defaults(
                    output_cls=MultiGuidanceExtraction,
                    prompt_template_str=self.system_prompt + "\n\nDocument text:\n{text}\n\nExtract all guidance items:",
                    llm=self.llm,
                    verbose=False
                )
                
                # Run extraction
                result = program(text=focused_text)
                break  # Success!
                
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                
                # Special handling: if the LLM returned a single object instead of wrapping it,
                # try to parse it manually as a workaround
                if "field required" in msg and "guidance_items" in msg:
                    print(f"  [FALLBACK] LLM returned single object instead of array, attempting manual wrap...")
                    try:
                        # Try to parse the raw output as a single Guidance object and wrap it
                        # This is a heuristic fallback for common LLM mistakes
                        raw_output = getattr(e, 'raw_output', None) or str(e)
                        # For now, let it retry normally - the prompt update should fix this
                        pass
                    except:
                        pass
                
                # If it's a rate/quota-related error, attempt to switch to the
                # smaller/faster `gpt-4o-mini` model automatically (one-time).
                if any(tok in msg for tok in ("rate", "quota", "limit", "throttl")) and self.model != "gpt-4o-mini":
                    print(f"  [FALLBACK] Detected rate/quota error: {e}. Switching model -> gpt-4o-mini and retrying...")
                    try:
                        self.model = "gpt-4o-mini"
                        self.llm = setup_llm(provider=self.provider, model=self.model, temperature=self.temperature)
                        # continue to next attempt without sleeping so fallback is fast
                        continue
                    except Exception as reconfig_err:
                        print(f"  [ERROR] Failed to reconfigure fallback model: {reconfig_err}")

                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"  [RETRY] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  [ERROR] All {self.max_retries} attempts failed: {e}")
                    return []
        
        if not result:
            print(f"  [ERROR] Extraction failed after {self.max_retries} attempts")
            return []
        
        try:
            # The program was configured to output MultiGuidanceExtraction where
            # each item is (or can be parsed into) the Guidance pydantic model.
            guidance_items: List[Guidance] = []
            for item in result.guidance_items:
                # item may already be a Guidance instance or a dict-like object
                if isinstance(item, Guidance):
                    parsed = item
                else:
                    # Parse into Guidance model to ensure types/validation
                    parsed = Guidance.parse_obj(item)

                # Attach some metadata if provided
                if metadata:
                    if metadata.get("source_url"):
                        parsed.source_url = metadata.get("source_url")
                    if metadata.get("published_at"):
                        parsed.extracted_at = metadata.get("published_at")
                    if metadata.get("source_id"):
                        parsed.source_type = metadata.get("source_id")

                # Improve statement_text by finding it in the full document and
                # expanding the snippet so we show more preceding context.
                st = parsed.statement_text or ""
                if st:
                    try:
                        # Case-insensitive search for the statement snippet in the full text
                        match_idx = text.lower().find(st.strip().lower())
                        if match_idx != -1:
                            # SNAP TO PARAGRAPH BOUNDARIES to avoid "shifted text" duplicates
                            # 1. Find start of current paragraph (double newline)
                            para_start = text.rfind('\n\n', 0, match_idx)
                            para_start = para_start + 2 if para_start != -1 else 0
                            
                            # 2. Find end of current paragraph
                            match_end = match_idx + len(st)
                            para_end = text.find('\n\n', match_end)
                            if para_end == -1:
                                para_end = len(text)
                                
                            # 3. Include preceding paragraph if it looks like a header or context
                            # (often segment names are in the line above)
                            context_start = para_start
                            if para_start > 0:
                                prev_para_start = text.rfind('\n\n', 0, para_start - 2)
                                prev_para_start = prev_para_start + 2 if prev_para_start != -1 else 0
                                # If previous paragraph is reasonable length (e.g. header or short intro), include it
                                if (para_start - prev_para_start) < 300:
                                    context_start = prev_para_start
                            
                            # 4. Extract and validate length
                            expanded_text = text[context_start:para_end].strip()
                            
                            # Fallback: if paragraph structure is missing or huge, use fixed window
                            if len(expanded_text) > 2000 or len(expanded_text) < len(st) + 50:
                                pre = 600
                                post = 400
                                start_snip = max(0, match_idx - pre)
                                end_snip = min(len(text), match_end + post)
                                expanded_text = text[start_snip:end_snip].strip()
                                
                            parsed.statement_text = expanded_text
                    except Exception:
                        # If anything goes wrong, keep the original statement_text
                        pass

                # SMART FILTER: Check if statement is predominantly historical vs forward-looking
                # Instead of binary reject, score both types of language and only reject if heavily historical
                statement_lower = (parsed.statement_text or "").lower()
                
                # Count forward-looking signals (strong indicators of guidance)
                forward_keywords = [
                    r'\bexpect\w*\b', r'\bforecast\w*\b', r'\bproject\w*\b', r'\banticipat\w*\b',
                    r'\bguidance\b', r'\boutlook\b', r'\btarget\w*\b', r'\bplan\w*\b',
                    r'\bwill be\b', r'\bwill reach\b', r'\bwill grow\b',
                    r'\bto be\b.*\$', r'\bfor (?:Q|FY)\d+\b'  # "to be $X" or "for Q2/FY2025"
                ]
                forward_score = sum(1 for pattern in forward_keywords if re.search(pattern, statement_lower))
                
                # Count past-tense signals (only strongly historical ones, not comparative context)
                # Exclude phrases that are often used alongside guidance (e.g., "compared to" in ranges)
                strong_past_keywords = [
                    r'\bfiscal year ended\b', r'\bquarter ended\b',
                    r'\breported (?:revenue|earnings|income)\b',
                    r'\bannounced.*results?\b',
                    r'\bwas \$[\d.]+ (?:billion|million)\b',  # "was $X billion" (specific past value)
                    r'\bincreased \d+%\b.*\bcompared to\b'  # "increased X% compared to" (pure historical)
                ]
                strong_past_score = sum(1 for pattern in strong_past_keywords if re.search(pattern, statement_lower))
                
                # Only reject if statement is PREDOMINANTLY historical (no forward signals + strong past signals)
                if strong_past_score >= 2 and forward_score == 0:
                    print(f"  [FILTER] Rejected purely historical item: {parsed.guidance_type} (past_score={strong_past_score}, forward_score={forward_score})")
                    continue  # Skip this item

                guidance_items.append(parsed)
            
            # Deduplicate: Remove items with highly overlapping statement_text AND similar content
            # This happens when LLM extracts the same guidance multiple times from one section
            # CRITICAL: Only deduplicate if BOTH text overlaps AND guidance content is identical
            if len(guidance_items) > 1:
                deduplicated = []
                
                for item in guidance_items:
                    statement = (item.statement_text or "").strip()
                    
                    # Check if this is a true duplicate (same content, not just same source text)
                    is_duplicate = False
                    for existing in deduplicated:
                        existing_statement = (existing.statement_text or "").strip()
                        
                        # First check: do the statements significantly overlap?
                        text_overlap = False
                        if len(statement) > 50 and len(existing_statement) > 50:
                            shorter = min(statement, existing_statement, key=len)
                            longer = max(statement, existing_statement, key=len)
                            overlap_ratio = len(shorter) / len(longer) if len(longer) > 0 else 0
                            text_overlap = (overlap_ratio > 0.7 or shorter in longer)
                        
                        # Second check: is the guidance content identical?
                        # Compare guidance_type, metric_name, and numeric values
                        content_identical = (
                            item.guidance_type == existing.guidance_type and
                            item.metric_name == existing.metric_name and
                            item.guided_value == existing.guided_value and
                            item.guided_range_low == existing.guided_range_low and
                            item.guided_range_high == existing.guided_range_high and
                            item.change_pct == existing.change_pct
                        )
                        
                        # Only mark as duplicate if BOTH conditions are true
                        if text_overlap and content_identical:
                            is_duplicate = True
                            print(f"  [DEDUP] Skipped duplicate: {item.guidance_type} with same values")
                            break
                    
                    if not is_duplicate:
                        deduplicated.append(item)
                
                guidance_items = deduplicated
                
            if guidance_items:
                print(f"  [LLM] Extracted {len(guidance_items)} guidance items")
            else:
                print("  [LLM] No guidance items found")
            return guidance_items

        except Exception as e:
            print(f"  [ERROR] Post-processing failed: {e}")
            return []


def load_candidates(candidates_path: Path = CANDIDATES_PATH, max_items: int = None, randomize: bool = False) -> List[Dict[str, Any]]:
    """Load filtered candidate documents."""
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Only stop early if NOT randomizing
            if not randomize and max_items and i >= max_items:
                break
            candidates.append(json.loads(line))
            
    if randomize and max_items and len(candidates) > max_items:
        print(f"Randomly sampling {max_items} from {len(candidates)} candidates...")
        candidates = random.sample(candidates, max_items)
    
    print(f"Loaded {len(candidates)} candidate documents")
    return candidates


def load_contents_for_candidates(
    candidates: List[Dict[str, Any]], 
    contents_path: Path = CONTENTS_PATH
) -> Dict[str, Dict[str, Any]]:
    """
    Load full text content for candidate UIDs.
    
    Returns:
        Dictionary mapping uid -> content record
    """
    candidate_uids = {c['uid'] for c in candidates}
    contents = {}
    
    with open(contents_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            uid = record.get('uid')
            if uid in candidate_uids:
                contents[uid] = record
    
    print(f"Loaded {len(contents)} content records (matched {len(contents)}/{len(candidates)} candidates)")
    return contents


def main():
    """Main extraction pipeline."""
    parser = argparse.ArgumentParser(description="Extract financial guidance using LLM")
    parser.add_argument("--provider", default="github", help="LLM provider (github, openai)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    parser.add_argument("--max-items", type=int, help="Limit number of candidates to process (for testing)")
    parser.add_argument("--random", action="store_true", help="Randomly sample max-items instead of taking the first N")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts for LLM calls")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output file path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM GUIDANCE EXTRACTION")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Max retries: {args.max_retries}")
    if args.max_items:
        print(f"Max items: {args.max_items} (TEST MODE)")
    print()
    
    # Initialize extractor
    print("Initializing LLM extractor...")
    extractor = LLMExtractor(
        provider=args.provider, 
        model=args.model,
        max_retries=args.max_retries
    )
    print()
    
    # Load data
    print("Loading candidate documents...")
    candidates = load_candidates(max_items=args.max_items, randomize=args.random)
    
    print("Loading full text content...")
    contents = load_contents_for_candidates(candidates)
    print()
    
    # Extract guidance
    print("Starting extraction...")
    print("-" * 60)
    
    all_extractions = []
    success_count = 0
    error_count = 0
    
    for i, candidate in enumerate(candidates, 1):
        uid = candidate['uid']
        title = candidate.get('title', 'N/A')
        
        print(f"\n[{i}/{len(candidates)}] Processing: {title[:80]}")
        print(f"  UID: {uid}")
        
        # Get full content
        content_record = contents.get(uid)
        if not content_record:
            print(f"  [SKIP] No content found for UID {uid}")
            error_count += 1
            continue
        
        full_text = content_record.get('extracted_text', '')
        if not full_text:
            print(f"  [SKIP] Empty text for UID {uid}")
            error_count += 1
            continue
        
        print(f"  Text length: {len(full_text):,} chars")
        
        # Extract metadata
        metadata = {
            'source_url': candidate.get('source_url'),
            'published_at': content_record.get('published_at'),
            'source_id': candidate.get('source_id'),
            'match_patterns': candidate.get('matched_patterns', [])
        }
        
        # Run extraction
        try:
            guidance_items = extractor.extract_from_text(full_text, metadata)
            
            if guidance_items:
                success_count += 1
                # Save each guidance item with source info
                for guidance in guidance_items:
                    extraction_record = {
                        'uid': uid,
                        'source_id': metadata['source_id'],
                        'source_url': metadata['source_url'],
                        'title': title,
                        'guidance': guidance.to_dict()
                    }
                    all_extractions.append(extraction_record)
            else:
                print(f"  [WARN] No guidance extracted")
                
        except Exception as e:
            print(f"  [ERROR] Extraction exception: {e}")
            error_count += 1
            continue
    
    # Save results
    print()
    print("-" * 60)
    print(f"Extraction complete!")
    print(f"  Successful: {success_count}/{len(candidates)}")
    print(f"  Errors: {error_count}/{len(candidates)}")
    print(f"  Total guidance items: {len(all_extractions)}")
    print()
    
    if all_extractions:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for record in all_extractions:
                f.write(json.dumps(record) + '\n')
        
        print(f"Saved {len(all_extractions)} guidance items to: {args.output}")
    else:
        print("No guidance items extracted. No output file created.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
