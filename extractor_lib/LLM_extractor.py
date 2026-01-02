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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4
import argparse
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your LLM setup and guidance schema
sys.path.append(str(Path(__file__).parent.parent))
from llm_setup import setup_llm
from extractor_lib.guidance_schema import Guidance, GuidanceExtraction
from extractor_lib.period_normalizer import (
    normalize_period,
    create_period_normalization_agent,
)
from ticker_map import load_ticker_map

# Load company mapping once
COMPANY_MAP = load_ticker_map()


# File paths
CANDIDATES_PATH = Path("extractor_lib") / "candidate_guidance.jsonl"
CONTENTS_PATH = Path("ingestion_json") / "contents.jsonl"
OUTPUT_PATH = Path("extractor_lib") / "extracted_guidance.jsonl"

# Guidance keywords for smart text extraction (from regex_filter)
GUIDANCE_KEYWORDS = [
    r'\b(expect|expects|expected|expecting|anticipat\w*|forecast\w*|project\w*|guidance)\b',
    r'\b(outlook|target\w*|goal\w*|aim\w*)\b',
]


class MultiGuidanceExtraction(BaseModel):
    """Container for multiple guidance items in a document."""
    guidance_items: List[GuidanceExtraction] = Field(
        description="List of all financial guidance items found in the document"
    )
    review_summary: Optional[str] = Field(
        description="Short explanation of what was fixed or added during review. If no changes, state 'No changes'."
    )
    changes_made: bool = Field(
        description="True if any items were added, removed, or modified. False otherwise."
    )


class LLMExtractor:
    """Extract financial guidance using an LLM with structured output."""
    
    def __init__(self, provider: str = "deepseek", model: str = None, reasoning_model: str = None, temperature: float = 0.0, max_retries: int = 3):
        """
        Initialize the extractor.
        
        Args:
            provider: LLM provider ("deepseek", "github", "openai", etc.)
            model: Model for initial extraction (default: deepseek-chat)
            reasoning_model: Model for reasoning extraction (default: deepseek-reasoner if provider is deepseek)
            temperature: 0.0 for deterministic extraction
            max_retries: Number of retry attempts for failed LLM calls
        """
        # store config so we can reconfigure on errors (e.g., rate limits)
        self.provider = provider
        self.model = model or "deepseek-chat"
        self.temperature = temperature
        self.llm = setup_llm(provider=self.provider, model=self.model, temperature=self.temperature)
        
        # Setup separate LLM for reasoning (Reasoning model)
        if reasoning_model:
            self.reasoning_model_name = reasoning_model
        elif provider == "deepseek":
            self.reasoning_model_name = "deepseek-reasoner" # Default to reasoning model for DeepSeek
        else:
            self.reasoning_model_name = self.model # Fallback to same model

        print(f"[LLM] Reasoning Model: {self.reasoning_model_name}")
        self.reasoning_llm = setup_llm(provider=self.provider, model=self.reasoning_model_name, temperature=self.temperature)

        # Setup utility LLM for tools (agents) - MUST support function calling
        # DeepSeek Reasoner (R1) does NOT support tools, so we force deepseek-chat for agents
        if provider == "deepseek":
            self.tool_llm_name = "deepseek-chat"
        else:
            self.tool_llm_name = self.model # Assume other providers' main models support tools
            
        self.tool_llm = setup_llm(provider=self.provider, model=self.tool_llm_name, temperature=self.temperature)

        self.max_retries = max_retries

        # Create period normalization agent (lazy-loaded, only created if needed)
        self._period_agent = None

        # System prompt for extraction
        # NOTE: the field names below match the Guidance pydantic model in
        # extractor_lib.guidance_schema so the LLM's structured output can be
        # parsed directly into that model. 
        self.system_prompt = """
        You are a financial analyst specialized in extracting forward-looking financial guidance from corporate filings, earnings releases, and press statements.

        CRITICAL: Only extract FORWARD-LOOKING guidance. DO NOT extract historical results or past performance.

        EXAMPLES OF FORWARD-LOOKING (extract these):
        ✓ "expects revenue to be $50-52 billion for Q2"
        ✓ "guidance for full year EPS of $5.00 to $5.50"
        ✓ "anticipates operating margin of approximately 30%"
        ✓ "targeting 10% growth in FY2026"
        ✓ "plans to increase capital expenditures to $2 billion"

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

        EXAMPLES OF OPERATIONAL/NON-FINANCIAL (DO NOT extract these):
        ✗ "plans to reduce team size by 10,000" (Headcount/HR)
        ✗ "expects to close 5,000 open roles" (Headcount/HR)
        ✗ "aims to complete efficiency analysis by summer" (Strategic milestone without financial value)
        ✗ "targeting developer productivity enhancements" (Operational goal)
        ✗ "launching new product in Q3" (Product launch without revenue guidance)

        Look for forward-looking verbs: expects, guidance, outlook, forecast, projects, anticipates, targets, will be, plans to
        REJECT operational metrics: headcount, employees, roles, team size, users, subscribers (unless explicitly revenue-related)

        You will extract guidance on financial statement items (revenue, earnings, etc.) and key operational metrics that directly
        impact financial performance (e.g., AOV for marketplaces, subscriber growth for SaaS). Purely qualitative commentary without clear financial linkage is excluded.
        
        Category definitions for financial guidance extraction:

        - revenue: Dollar-denominated sales figures (net sales, total revenue, segment revenue). Not unit volumes or delivery counts.
        - earnings: Net income, profit, operating income, or other profit-line metrics in dollar terms.
        - EPS: Earnings per share, explicitly stated as a per-share figure.
        - opex: Operating expenses, SG&A, R&D spend, or other cost-line items. Includes expense growth guidance.
        - capex: Capital expenditures, infrastructure investment, property/equipment spending.
        - margin: Percentage-based profitability metrics (gross margin, operating margin, net margin, profit margin).
        - cash_flow: Operating cash flow, free cash flow, or general cash flow guidance in dollar terms.
        - ebitda: EBITDA or adjusted EBITDA, explicitly stated.
        - other: Financial metrics that don't fit above categories (e.g., average order value, gross merchandise value, ARPU, segment-specific losses). Use for dollar-denominated or ratio-based business metrics only. Exclude purely operational KPIs like unit volumes, subscriber counts, or delivery numbers.

        For each distinct FORWARD-LOOKING guidance item, create a JSON object with these exact fields:
        - guidance_type: MUST be one of: "revenue", "earnings", "EPS", "opex", "capex", "margin", "cash_flow", "ebitda", "other" (or null)
        - metric_name: The exact name of the metric as it appears in the text (e.g. "Total Revenue", "Adjusted EBITDA", "Capital Expenditures", "Organic Growth"). ALWAYS extract this.
        - statement_text: The exact sentence or text snippet from the document where this guidance was found. (string or null)
        - reporting_period: The reporting period referenced (e.g., "Q2 2025", "FY2025", keep format consistent e.g. don't vary format to sometimes say full-year 2024, sometimes FY2024)  (or null)
        - current_value: Current/most-recent numeric value (number or null)
        - unit: MUST be one of: "million", "billion", "%", "USD", "units" (or null)
        - guided_range_low: The guided value (if single number) OR the low end of the range (if range). (number or null)
        - guided_range_high: The high end of the range (if range). Leave null if single number. (number or null)
        - is_revision: true/false indicating if this is a revision to prior guidance, e.g. updated from our prior outlook of $94-99 billion would yield true (boolean)
        - revision_direction: "increased", "decreased" or null, compared to previous guidance ONLY (string or null)
        - qualitative_direction: when no value is being given, but a qualitative direction is indicated (e.g., "increase", "decrease", "improve", "decline") (string or null)
        - rationales: Any qualitative explanations or reasons given for this guidance, keep it brief (string or null)


        Do NOT extract historical results. Do NOT return past performance data.
        """

        

    @property
    def period_agent(self):
        """Lazy-load the period normalization agent with hierarchical LLM setup."""
        if self._period_agent is None:
            try:
                print("  [AGENT] Creating period normalization agent...")
                print(f"    - Agent LLM (tool-calling): {self.tool_llm_name}")
                # Hierarchical setup:
                # - agent_llm: deepseek-chat (supports function calling)
                self._period_agent = create_period_normalization_agent(
                    agent_llm=self.tool_llm,
                    reasoning_llm=self.tool_llm
                )
            except Exception as e:
                print(f"  [WARN] Failed to create period agent: {e}")
                # Fallback to None (will skip agent-based normalization)
                self._period_agent = None
        return self._period_agent

    def _normalize_period(
        self,
        raw_period: Optional[str],
        company: str = "",
        published_at: str = "",
        statement_text: str = "",
        use_agent: bool = False
    ) -> Optional[str]:
        """
        Two-stage period normalization:
        1. Fast regex (handles 80%+ of cases)
        2. FunctionAgent for complex/ambiguous periods (if use_agent=True)

        Args:
            raw_period: Raw period string from extraction (can be None)
            company: Company name for fiscal calendar lookup
            published_at: Document publish date
            statement_text: Context text
            use_agent: Whether to use the agent for complex cases

        Returns:
            Normalized period string, or None if normalization fails/not possible
        """
        # normalize_period handles None/empty gracefully
        # It will try agent-based inference if raw_period is missing AND agent is provided
        
        agent_to_use = self.period_agent if use_agent else None
        
        result = normalize_period(
            raw_period=raw_period or "",
            company=company,
            published_at=published_at,
            statement_text=statement_text,
            agent=agent_to_use
        )
        
        # Return None instead of empty string for cleaner downstream handling
        return result if result else None

    def _attach_metadata(self, item: Guidance, metadata: Optional[Dict[str, Any]]) -> None:
        """Attach metadata to a Guidance item.
        
        Note: Document-level metadata (source_url, published_at, ingested_at) are stored
        at the document level, not duplicated per guidance item.
        """
        if not metadata:
            return

        # Set extraction time (now)
        item.extracted_at = datetime.now(timezone.utc).isoformat()

        # Auto-fill Company Name (priority: extracted > metadata > source_id map)
        if not item.company:
            if metadata.get("company_name"):
                item.company = metadata.get("company_name")
            elif metadata.get("source_id") and metadata.get("source_id") in COMPANY_MAP:
                item.company = COMPANY_MAP[metadata.get("source_id")]

        if metadata.get("source_id"):
            sid = str(metadata.get("source_id")).lower()
            
            # Map source_type based on source_id
            if "8-k" in sid or "8k" in sid:
                item.source_type = "8-K"
            elif "10-k" in sid or "10k" in sid:
                item.source_type = "10-K"
            elif "10-q" in sid or "10q" in sid:
                item.source_type = "10-Q"
            elif "press" in sid or "release" in sid:
                item.source_type = "press_release"
            elif "call" in sid or "transcript" in sid:
                item.source_type = "earnings_call"
            elif "presentation" in sid:
                item.source_type = "investor_presentation"
            else:
                item.source_type = "other"

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
        post_context = 800  # Focus on what comes AFTER the guidance keyword
        for pattern in GUIDANCE_KEYWORDS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, match.start() - pre_context)
                end = min(len(text), match.end() + post_context)
                
                # Extend to the end of the sentence to avoid cutting off text
                # Look ahead up to 300 chars for a sentence terminator or newline
                snippet_after = text[end:min(len(text), end+300)]
                sent_end_match = re.search(r'[.!?\n]', snippet_after)
                if sent_end_match:
                    end += sent_end_match.end()
                
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
                new_end = max(end, merged[-1][1])
                merged[-1] = (merged[-1][0], new_end, text[merged[-1][0]:new_end])
            else:
                merged.append((start, end, section))
        
        # Join sections with clear separators
        focused_text = "\n\n[...]\n\n".join(section for _, _, section in merged)
        
        if focused_text:
            reduction = len(text) - len(focused_text)
            print(f"  [SMART] Focused on {len(merged)} sections (reduced by {reduction:,} chars)")
            return focused_text
        
        # If no matches, return beginning (shouldn't happen with our pre-filtered candidates)
        return text[:200000]
    
    def extract_from_text(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        use_agentic_normalization: bool = False
    ) -> List[Guidance]:
        """
        Extract guidance items from raw text with smart section extraction.
        
        Args:
            text: The document text to analyze
            metadata: Optional metadata (company, source_url, etc.)
            use_agentic_normalization: Whether to use agent for period normalization
            
        Returns:
            List of Guidance objects
        """
        if not text or len(text.strip()) < 50:
            print("  [WARN] Text too short for extraction")
            return []
        
        # Smart extraction: focus on relevant sections
        focused_text = self._extract_relevant_sections(text)
        
        # Final length check (DeepSeek handles 128k tokens, so we can be generous)
        max_chars = 200000
        if len(focused_text) > max_chars:
            print(f"  [WARN] Truncating focused text from {len(focused_text)} to {max_chars} chars")
            focused_text = focused_text[:max_chars] + "\n... [truncated]"
        
        # Capture start time for performance tracking
        processing_start_time = time.time()

        # Retry logic for LLM calls
        result = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Use regular chat with JSON schema in prompt
                # as_structured_llm can conflict with detailed system prompts
                schema_reminder = """
                
OUTPUT FORMAT: Return a JSON object with this exact structure:
{
    "guidance_items": [
        {
            "guidance_type": "revenue|earnings|EPS|opex|capex|margin|cash_flow|ebitda|other",
            "metric_name": "string",
            "statement_text": "string",
            "reporting_period": "string or null",
            "current_value": "number or null",
            "unit": "million|billion|%|USD|null",
            "guided_range_low": "number or null",
            "guided_range_high": "number or null",
            "is_revision": "boolean",
            "revision_direction": "increased|decreased|null",
            "qualitative_direction": "string or null",
            "rationales": "string or null"
        }
    ],
    "review_summary": "Brief summary",
    "changes_made": true
}

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.
If you find forward-looking guidance, extract it. Do not return an empty guidance_items array if there is guidance present.
"""
                
                messages = [
                    ChatMessage(role="system", content=self.system_prompt + schema_reminder),
                    ChatMessage(role="user", content=f"Document text:\n{focused_text}\n\nExtract all forward-looking guidance items as JSON:")
                ]
                
                # Run extraction with regular LLM
                response = self.llm.chat(messages)
                json_content = response.message.content
                
                # Clean markdown if present
                if "```json" in json_content:
                    json_content = json_content.split("```json")[1].split("```")[0].strip()
                elif "```" in json_content:
                    json_content = json_content.split("```")[1].split("```")[0].strip()
                
                # Parse JSON
                result = MultiGuidanceExtraction.model_validate_json(json_content)
                break  # Success!
                
            except Exception as e:
                last_error = e
                
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
        
        # Calculate duration
        processing_duration = time.time() - processing_start_time

        try:
            # The program was configured to output MultiGuidanceExtraction where
            # each item is (or can be parsed into) the GuidanceExtraction pydantic model.
            guidance_items: List[Guidance] = []
            for item in result.guidance_items:
                # item is GuidanceExtraction (from LLM)
                if isinstance(item, GuidanceExtraction):
                    # Convert to Guidance (full model)
                    parsed = Guidance(**item.model_dump())
                elif isinstance(item, dict):
                    parsed = Guidance.parse_obj(item)
                else:
                    # Fallback
                    parsed = Guidance(**item.dict())

                # Force a fresh unique ID because LLM often hallucinates generic IDs like "guid_1"
                parsed.guid = uuid4().hex

                # Attach metadata
                self._attach_metadata(parsed, metadata)

                # Normalize reporting period (two-stage: fast regex then agent if needed)
                # Even if reporting_period is missing, we try to infer it using the agent
                parsed.reporting_period = self._normalize_period(
                    raw_period=parsed.reporting_period,
                    company=parsed.company or "",
                    published_at=metadata.get('published_at', '') if metadata else "",
                    statement_text=parsed.statement_text or "",
                    use_agent=use_agentic_normalization
                )

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
                            item.guided_range_low == existing.guided_range_low and
                            item.guided_range_high == existing.guided_range_high
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
                # Amortize processing duration across all kept items
                amortized_duration = processing_duration / len(guidance_items)
                for item in guidance_items:
                    item.processing_duration_seconds = round(amortized_duration, 2)
                
                print(f"  [LLM] Extracted {len(guidance_items)} guidance items")
            else:
                print("  [LLM] No guidance items found")
            return guidance_items

        except Exception as e:
            print(f"  [ERROR] Post-processing failed: {e}")
            return []

    # REMOVED: extract_with_reasoning (deprecated in favor of extract_from_text + agentic normalization)


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
    parser.add_argument("--provider", default="deepseek", help="LLM provider (deepseek, github, openai)")
    parser.add_argument("--model", default="deepseek-chat", help="Model name")
    parser.add_argument("--max-items", type=int, help="Limit number of candidates to process (for testing)")
    parser.add_argument("--random", action="store_true", help="Randomly sample max-items instead of taking the first N")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts for LLM calls")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Output file path")
    parser.add_argument("--reasoning", action="store_true", help="Enable agentic period normalization (slower but more accurate)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers to use")
    
    args = parser.parse_args()

    # If using default output path and reasoning mode is on, switch to reasoning filename
    if args.reasoning and args.output == OUTPUT_PATH:
        args.output = OUTPUT_PATH.with_name("extracted_guidance_agentic.jsonl")
    
    print("=" * 60)
    print("LLM GUIDANCE EXTRACTION")
    print("=" * 60)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Mode: {'Standard + Agentic Normalization' if args.reasoning else 'Standard + Regex Normalization'}")
    print(f"Max retries: {args.max_retries}")
    print(f"Output: {args.output}")
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
    print("Starting parallel extraction...")
    print(f"Using {args.workers} workers")
    print("-" * 60)
    
    all_extractions = []
    success_count = 0
    error_count = 0
    
    # Create a single extractor instance to share (if thread-safe) or create per-thread
    # Since we're using llama-index which might have shared state, let's create per-thread
    # But we need to pass the config, not the instance
    
    def process_candidate(candidate_data):
        """Process a single candidate - returns (success, extraction_records, error_msg)"""
        i, candidate, contents = candidate_data
        uid = candidate['uid']
        title = candidate.get('title', 'N/A')
        
        try:
            content_record = contents.get(uid)
            if not content_record:
                return (False, [], f"No content for {uid}")
            
            full_text = content_record.get('extracted_text', '')
            if not full_text:
                return (False, [], f"Empty text for {uid}")
            
            metadata = {
                'source_url': candidate.get('source_url'),
                'published_at': content_record.get('published_at'),
                'fetched_at': content_record.get('fetched_at'),
                'source_id': candidate.get('source_id')
            }
            
            # Create a NEW extractor instance per thread to avoid race conditions
            # This is safer than sharing one instance across threads
            thread_extractor = LLMExtractor(
                provider=args.provider,
                model=args.model,
                max_retries=args.max_retries
            )
            
            if args.reasoning:
                # Standard extraction + Agentic normalization
                guidance_items = thread_extractor.extract_from_text(
                    full_text, 
                    metadata,
                    use_agentic_normalization=True
                )
            else:
                # Standard extraction + Regex normalization only
                guidance_items = thread_extractor.extract_from_text(
                    full_text, 
                    metadata,
                    use_agentic_normalization=False
                )
            
            records = []
            for guidance in guidance_items:
                extraction_record = {
                    'uid': uid,
                    'source_id': metadata['source_id'],
                    'source_url': metadata['source_url'],
                    'title': title,
                    'guidance': guidance.to_dict()
                }
                records.append(extraction_record)
            
            return (True, records, None)
            
        except Exception as e:
            return (False, [], str(e))

    # Prepare candidate data with indices
    candidate_data = [(i+1, cand, contents) for i, cand in enumerate(candidates)]

    # Process with ThreadPool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Map future to candidate info for progress tracking
        futures = {executor.submit(process_candidate, data): data for data in candidate_data}
        
        for future in as_completed(futures):
            # Retrieve original data to print progress
            i, candidate, _ = futures[future]
            uid = candidate['uid']
            title = candidate.get('title', 'N/A')[:80]
            
            print(f"\n[{i}/{len(candidates)}] Completed: {title}")
            
            try:
                success, records, error_msg = future.result()
                
                if success:
                    success_count += 1
                    all_extractions.extend(records)
                    print(f"  ✓ Extracted {len(records)} items")
                else:
                    error_count += 1
                    print(f"  ✗ {error_msg}")
            except Exception as e:
                error_count += 1
                print(f"  ✗ Critical error in thread: {e}")
        
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