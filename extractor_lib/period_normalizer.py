"""
Period normalization for LLM-based guidance extraction.

Two-stage normalization:
1. quick_normalize() - Fast regex for 80%+ of cases
2. FunctionAgent - For complex/ambiguous periods requiring reasoning
    - Can use multiple tools:
        - Fiscal calendar lookup (no LLM cost)
        - Date context inference (no LLM cost)
    Outputs in standard format
    If really unclear, falls outputs "None"

Standard formats:
- Fiscal Year: "FY2025"
- Fiscal Quarter: "Q1 FY2025" 
- Half-Year: "H1 FY2025"
- Multi-Year: "FY2021-FY2024"
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import re
import asyncio
from llama_index.core.agent.workflow import FunctionAgent


# Load fiscal calendar mappings
FISCAL_CALENDARS_PATH = Path(__file__).parent / "fiscal_calendars.json"

def load_fiscal_calendars() -> Dict[str, Dict[str, Any]]:
    """Load company fiscal year end dates."""
    if FISCAL_CALENDARS_PATH.exists():
        with open(FISCAL_CALENDARS_PATH, 'r') as f:
            return json.load(f)
    return {}

FISCAL_CALENDARS = load_fiscal_calendars()


def get_fiscal_calendar_info(company: str) -> Optional[Dict[str, Any]]:
    """Get fiscal calendar information for a company."""
    return FISCAL_CALENDARS.get(company)


def normalize_period(
    raw_period: str,
    company: str = "",
    published_at: str = "",
    statement_text: str = "",
    agent: Optional[FunctionAgent] = None
) -> str:
    """
    Two-stage period normalization (main entry point).
    
    1. Try fast regex normalization (handles 80%+ of cases)
    2. Fall back to agent-based reasoning for complex cases OR missing periods
    
    Args:
        raw_period: Raw period string from extraction (can be None/empty)
        company: Company name for fiscal calendar lookup
        published_at: Document publish date (ISO format)
        statement_text: Context text for ambiguous cases
        agent: Optional pre-created FunctionAgent (if None, skips agent stage)
        
    Returns:
        Normalized period string
    """
    # Stage 1: Fast regex normalization (only if we have a period string)
    if raw_period:
        result = quick_normalize(raw_period)
        if result:
            return result
    
    # Stage 2: Agent-based reasoning (if agent provided)
    # Triggers if regex failed OR if raw_period was missing
    if agent:
        return normalize_with_agent(agent, raw_period, company, published_at, statement_text)
    
    # No agent available, return original
    return raw_period


def create_period_normalization_agent(
    agent_llm,
    reasoning_llm=None
) -> FunctionAgent:
    """
    Create a FunctionAgent for normalizing complex period strings that 
    quick_normalize() couldn't handle.
    
    The agent has access to:
    - Cheap lookup tools (no LLM cost): fiscal calendar, date context
    - Expensive reasoning tool (uses reasoning_llm): for truly ambiguous periods
    
    This implements hierarchical reasoning:
    - Outer agent (agent_llm): decides WHEN to use tools, handles simple cases directly
    - Inner reasoning (reasoning_llm): deep inference for genuinely ambiguous periods
    
    The agent autonomously decides which tools to use based on complexity.

    Args:
        agent_llm: LLM for the agent itself (must support function calling, e.g., deepseek-chat)
        reasoning_llm: LLM for deep reasoning tool (can be a reasoning model, e.g., deepseek-reasoner).
                       If None, falls back to agent_llm.

    Returns:
        FunctionAgent configured with period normalization tools
    """
    # Fall back to agent_llm if no separate reasoning LLM provided
    if reasoning_llm is None:
        reasoning_llm = agent_llm

    # ===== CHEAP TOOLS (pure lookups, no LLM cost) =====

    def lookup_fiscal_calendar(company: str) -> str:
        """
        Look up the fiscal year end for a company.
        Use this FIRST to understand the company's fiscal calendar.
        
        Args:
            company: Company name or ticker symbol
            
        Returns:
            Fiscal year end info or default assumption
        """
        info = get_fiscal_calendar_info(company)
        if info:
            fy_end = info.get('fy_end_month', 12)
            notes = info.get('notes', '')
            return f"{company}: FY ends month {fy_end} ({notes}). " \
                   f"{'Calendar year company.' if fy_end == 12 else 'Non-calendar fiscal year.'}"
        # Default: assume calendar year (January-December = FY matches calendar year)
        return f"{company}: Unknown company - ASSUME CALENDAR YEAR. FY ends December, so FY2025 = Jan-Dec 2025."

    def infer_fiscal_year(company: str, date_str: str) -> str:
        """
        Determine which fiscal year a specific date falls into.
        
        Args:
            company: Company name for fiscal calendar lookup
            date_str: ISO date string (e.g., "2024-08-21")
            
        Returns:
            Fiscal year like "FY2025" and explanation
        """
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('T')[0])
            info = get_fiscal_calendar_info(company)
            fy_end_month = info.get('fy_end_month', 12) if info else 12
            
            # If date is after FY end month, it's in the next FY
            if date.month > fy_end_month:
                fy = date.year + 1
                return f"Date {date_str} falls in FY{fy} (after month {fy_end_month} cutoff)"
            return f"Date {date_str} falls in FY{date.year} (before/during month {fy_end_month} cutoff)"
        except Exception as e:
            return f"Unable to parse date: {e}"

    def get_current_fiscal_context(company: str, published_at: str) -> str:
        """
        Get full fiscal context for period inference.
        Use this when you need to understand "current quarter" or "next year" references.
        
        Args:
            company: Company name
            published_at: Document publication date
            
        Returns:
            Context string with current FY and quarter info
        """
        try:
            date = datetime.fromisoformat(published_at.replace('Z', '+00:00').split('T')[0])
            info = get_fiscal_calendar_info(company)
            fy_end_month = info.get('fy_end_month', 12) if info else 12
            
            # Calculate current FY
            if date.month > fy_end_month:
                current_fy = date.year + 1
            else:
                current_fy = date.year
            
            # Estimate current quarter (simplified)
            month_in_fy = (date.month - fy_end_month - 1) % 12 + 1
            current_q = (month_in_fy - 1) // 3 + 1
            
            # Calculate next quarter/year for relative references
            next_q = current_q + 1 if current_q < 4 else 1
            next_q_fy = current_fy if current_q < 4 else current_fy + 1
            
            return (
                f"As of {published_at}:\n"
                f"  - Current period: Q{current_q} FY{current_fy}\n"
                f"  - 'Next quarter' would be: Q{next_q} FY{next_q_fy}\n"
                f"  - 'This year' / 'Full year' likely means: FY{current_fy}\n"
                f"  - 'Next year' likely means: FY{current_fy + 1}"
            )
        except Exception as e:
            return f"Unable to determine fiscal context: {e}"

    # ===== AGENT CONFIGURATION =====
    
    tools = [
        lookup_fiscal_calendar,
        infer_fiscal_year,
        get_current_fiscal_context,
        # reason_about_ambiguous_period,  # LLM-powered, use sparingly
    ]

    system_prompt = """You are a financial reporting period normalization specialist.

STANDARD FORMATS (use exactly):
- Fiscal Year: "FY2025" (not "FY 2025" or "Fiscal Year 2025")
- Fiscal Quarter: "Q1 FY2025" (with single space)
- Half-Year: "H1 FY2025" or "H2 FY2025"
- Multi-Year: "FY2021-FY2024" (hyphen, no spaces)

TOOL USAGE STRATEGY (use cheapest approach that works):

1. SIMPLE CASES - No tools needed:
   - "Q1 2025" → output "Q1 FY2025" directly (fiscal is always implied)
   - "FY2025" → output "FY2025" directly
   - "First quarter 2024" → output "Q1 FY2024" directly

2. NEED FISCAL CALENDAR - Use lookup_fiscal_calendar:
   - When company has non-standard fiscal year (Apple, Walmart, etc.)
   - If company is UNKNOWN, assume calendar year (FY = Jan-Dec)

3. NEED DATE CONTEXT - Use get_current_fiscal_context:
   - "Next quarter", "this year", "coming fiscal year"
   - "for the coming year / quarter" means for the next fiscal period based on publication date and company fiscal calendar
   - Any relative time reference that needs the document date
   - Works even for unknown companies (assumes calendar year)

4. TRULY AMBIGUOUS - Return "None":
   - Only if you genuinely cannot determine the period even with assumptions
   - Do NOT ask for clarification - just output "None"

CRITICAL RULES:
- "Q1 2025" ALWAYS means fiscal → "Q1 FY2025"
- Corporate guidance never uses calendar quarters
- UNKNOWN COMPANIES: Always assume calendar year fiscal (FY = Jan-Dec)
- Use the publication date + calendar year assumption to resolve "current quarter", "next year", etc.

OUTPUT FORMAT - STRICT:
- Return ONLY the normalized period string (e.g., "Q1 FY2025", "FY2024")
- Or return "None" if truly impossible to determine
- NEVER output explanations, questions, or reasoning
- NEVER say "I need" or ask for information"""

    try:
        return FunctionAgent(
            name="period_normalizer",
            tools=tools,
            llm=agent_llm,  # Agent uses tool-calling LLM
            system_prompt=system_prompt
        )
    except Exception as e:
        print(f"  [WARN] Error in FunctionAgent creation: {e}")
        return None


def normalize_with_agent(
    agent: FunctionAgent,
    raw_period: str,
    company: str = "",
    published_at: str = "",
    statement_text: str = ""
) -> str:
    """
    Use the FunctionAgent to normalize a complex period.
    Called only when quick_normalize() returns None or when raw_period is missing.

    Args:
        agent: The FunctionAgent instance
        raw_period: The raw period string
        company: Company name for fiscal calendar lookup
        published_at: ISO date string of document publication
        statement_text: Context text snippet

    Returns:
        Normalized period string, or original if normalization fails
    """
    # Build concise context for the agent
    if raw_period:
        context_parts = [f"Period to normalize: '{raw_period}'"]
    else:
        context_parts = ["Period: MISSING - Please infer from context and date"]
        
    if company:
        context_parts.append(f"Company: {company}")
    if published_at:
        context_parts.append(f"Published: {published_at}")
    if statement_text:
        # Truncate but try to keep meaningful context
        context_parts.append(f"Statement context: \"{statement_text[:200]}\"")

    prompt = f"""Normalize this period to standard format:

{chr(10).join(context_parts)}

Use tools if needed. Return ONLY the normalized period (e.g., "Q1 FY2025", "FY2024", "H1 FY2025")."""

    try:
        # Run async agent in sync context
        async def run_agent():
            return await agent.run(user_msg=prompt)

        response = asyncio.run(run_agent())
        normalized = str(response).strip()

        # CRITICAL: Detect and reject reasoning/question output
        # These indicate the agent failed to follow output format
        bad_patterns = [
            r'I need',
            r'Could you',
            r'Please provide',
            r'I cannot',
            r'I don\'t have',
            r'information',
            r'company name',
            r'publication date',
            r'\?',  # Any question marks
        ]
        for bad in bad_patterns:
            if re.search(bad, normalized, re.IGNORECASE):
                # Agent output reasoning instead of period - return None
                return None

        # Check for explicit "None" output
        if normalized.lower() == 'none':
            return None

        # Extract standard period format from response
        # Priority order: most specific patterns first
        patterns = [
            r'(Q[1-4]\s+FY\d{4})',      # Q1 FY2025
            r'(H[1-2]\s+FY\d{4})',       # H1 FY2025
            r'(FY\d{4}-FY\d{4})',        # FY2021-FY2024
            r'(FY\d{4})',                # FY2025
        ]
        
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return match.group(1)

        # If no standard format found and response is short, might be valid
        # If long (>50 chars), it's probably reasoning text - return None
        if len(normalized) > 50:
            return None
            
        return normalized if normalized else raw_period
        
    except Exception as e:
        print(f"  [WARN] Agent normalization failed: {e}")
        return raw_period


# Quick regex-based normalization for simple cases
# This can be used as a fast-path before calling the LLM
def quick_normalize(raw_period: str) -> Optional[str]:
    """
    Fast regex-based normalization for common period formats.
    Returns None if the period is too complex and needs agent reasoning.
    
    Handles ~80% of cases without LLM calls.
    """
    if not raw_period:
        return None

    raw = raw_period.strip()
    
    # ===== ALREADY STANDARD =====
    if re.match(r"^FY\d{4}$", raw):
        return raw
    if re.match(r"^Q[1-4]\s+FY\d{4}$", raw):
        return raw
    if re.match(r"^H[1-2]\s+FY\d{4}$", raw):
        return raw
    if re.match(r"^FY\d{4}-FY\d{4}$", raw):
        return raw

    # ===== FISCAL YEAR PATTERNS =====
    
    # FY 2025 → FY2025
    m = re.match(r"^FY\s+(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"FY{m.group(1)}"

    # FY25 → FY2025
    m = re.match(r"^FY(\d{2})$", raw, re.IGNORECASE)
    if m:
        year = 2000 + int(m.group(1))
        return f"FY{year}"

    # Fiscal Year 2025, Fiscal 2025 → FY2025
    m = re.match(r"^[Ff]iscal\s+(?:[Yy]ear\s+)?(\d{4})$", raw)
    if m:
        return f"FY{m.group(1)}"

    # Full Year 2025, Full-Year 2025 → FY2025
    m = re.match(r"^[Ff]ull[\s-]?[Yy]ear\s+(\d{4})$", raw)
    if m:
        return f"FY{m.group(1)}"

    # Bare year: 2025 → FY2025 (assume fiscal year in corporate context)
    m = re.match(r"^(20\d{2})$", raw)
    if m:
        return f"FY{m.group(1)}"

    # ===== QUARTER PATTERNS =====
    
    # Q1 FY 2025 → Q1 FY2025
    m = re.match(r"^Q([1-4])\s+FY\s*(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"Q{m.group(1)} FY{m.group(2)}"

    # Q1FY2025 (no space) → Q1 FY2025
    m = re.match(r"^Q([1-4])FY(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"Q{m.group(1)} FY{m.group(2)}"

    # Q1 2025 → Q1 FY2025 (always fiscal in corporate guidance)
    m = re.match(r"^Q([1-4])\s+(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"Q{m.group(1)} FY{m.group(2)}"

    # Q1'25, Q1 '25 → Q1 FY2025
    m = re.match(r"^Q([1-4])\s*'?(\d{2})$", raw, re.IGNORECASE)
    if m:
        year = 2000 + int(m.group(2))
        return f"Q{m.group(1)} FY{year}"

    # FY25 Q1, FY2025 Q1 → Q1 FY2025
    m = re.match(r"^FY(\d{2,4})\s+Q([1-4])$", raw, re.IGNORECASE)
    if m:
        year = int(m.group(1))
        if year < 100:
            year = 2000 + year
        return f"Q{m.group(2)} FY{year}"

    # First/Second/Third/Fourth Quarter 2025 → Q# FY2025
    quarter_words = {
        'first': '1', 'second': '2', 'third': '3', 'fourth': '4',
        '1st': '1', '2nd': '2', '3rd': '3', '4th': '4'
    }
    m = re.match(r"^(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(\d{4})$", raw, re.IGNORECASE)
    if m:
        q = quarter_words[m.group(1).lower()]
        return f"Q{q} FY{m.group(2)}"

    # ===== HALF-YEAR PATTERNS =====
    
    # H1 2025, H1 FY 2025 → H1 FY2025
    m = re.match(r"^H([1-2])\s+(?:FY\s*)?(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"H{m.group(1)} FY{m.group(2)}"

    # First Half 2025, 1H 2025 → H1 FY2025
    m = re.match(r"^(?:first\s+half|1H)\s+(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"H1 FY{m.group(1)}"

    # Second Half 2025, 2H 2025 → H2 FY2025
    m = re.match(r"^(?:second\s+half|2H)\s+(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"H2 FY{m.group(1)}"

    # ===== MULTI-YEAR PATTERNS =====
    
    # 2021-2024 → FY2021-FY2024
    m = re.match(r"^(\d{4})\s*[-–—]\s*(\d{4})$", raw)
    if m:
        return f"FY{m.group(1)}-FY{m.group(2)}"

    # FY21-FY24 → FY2021-FY2024
    m = re.match(r"^FY(\d{2})\s*[-–—]\s*FY(\d{2})$", raw, re.IGNORECASE)
    if m:
        y1 = 2000 + int(m.group(1))
        y2 = 2000 + int(m.group(2))
        return f"FY{y1}-FY{y2}"

    # ===== CALENDAR YEAR (explicit only) =====
    
    # CY2025, Calendar Year 2025 → CY2025 (keep distinct from fiscal)
    m = re.match(r"^(?:CY|[Cc]alendar\s+[Yy]ear)\s*(\d{4})$", raw)
    if m:
        return f"CY{m.group(1)}"

    # Too complex - needs agent
    return None