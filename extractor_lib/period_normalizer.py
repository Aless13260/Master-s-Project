"""
Period normalization for LLM-based guidance extraction.

Two-stage normalization:
1. quick_normalize() - Fast regex for 80%+ of cases
2. ReActAgent - For complex/ambiguous periods requiring reasoning

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
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent


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
    agent: Optional[ReActAgent] = None
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
        agent: Optional pre-created ReActAgent (if None, skips agent stage)
        
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


def create_period_normalization_agent(llm) -> ReActAgent:
    """
    Create a ReActAgent for normalizing complex period strings that 
    quick_normalize() couldn't handle.

    Args:
        llm: The LLM instance to use for the agent

    Returns:
        ReActAgent configured with period normalization tools
    """

    def lookup_fiscal_calendar(company: str) -> str:
        """
        Look up the fiscal year end for a company.
        
        Args:
            company: Company name or ticker symbol
            
        Returns:
            Fiscal year end info or default assumption
        """
        info = get_fiscal_calendar_info(company)
        if info:
            return f"{company}: FY ends {info.get('notes', 'Unknown')}"
        return f"{company}: No fiscal calendar found. Assuming calendar year (Dec 31)."

    def infer_fiscal_year(company: str, date_str: str) -> str:
        """
        Determine which fiscal year a date falls into.
        
        Args:
            company: Company name for fiscal calendar lookup
            date_str: ISO date string (e.g., "2024-08-21")
            
        Returns:
            Fiscal year like "FY2025"
        """
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('T')[0])
            info = get_fiscal_calendar_info(company)
            fy_end_month = info.get('fy_end_month', 12) if info else 12
            
            # If date is after FY end month, it's in the next FY
            if date.month > fy_end_month:
                return f"FY{date.year + 1}"
            return f"FY{date.year}"
        except Exception:
            return "Unable to parse date"

    def get_current_fiscal_context(company: str, published_at: str) -> str:
        """
        Get full fiscal context for period inference.
        
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
            
            return f"As of {published_at}: Currently in Q{current_q} FY{current_fy}"
        except Exception:
            return "Unable to determine fiscal context"

    tools = [
        FunctionTool.from_defaults(fn=lookup_fiscal_calendar),
        FunctionTool.from_defaults(fn=infer_fiscal_year),
        FunctionTool.from_defaults(fn=get_current_fiscal_context),
    ]

    system_prompt = """You are a financial reporting period normalization specialist.

STANDARD FORMATS (use exactly):
- Fiscal Year: "FY2025" (not "FY 2025" or "Fiscal Year 2025")
- Fiscal Quarter: "Q1 FY2025" (with single space)
- Half-Year: "H1 FY2025" or "H2 FY2025"
- Multi-Year: "FY2021-FY2024" (hyphen, no spaces)

KEY RULES:
- "Q1 2025" ALWAYS means fiscal → "Q1 FY2025"
- Vague terms ("next year", "this quarter") require date inference
- Use lookup_fiscal_calendar for company-specific FY ends
- Use get_current_fiscal_context when year is ambiguous

OUTPUT: Return ONLY the normalized period string, nothing else."""

    return ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=False,
        max_iterations=4,
        system_prompt=system_prompt
    )


def normalize_with_agent(
    agent: ReActAgent,
    raw_period: str,
    company: str = "",
    published_at: str = "",
    statement_text: str = ""
) -> str:
    """
    Use the ReActAgent to normalize a complex period.
    Called only when quick_normalize() returns None or when raw_period is missing.

    Args:
        agent: The ReActAgent instance
        raw_period: The raw period string
        company: Company name for fiscal calendar lookup
        published_at: ISO date string of document publication
        statement_text: Context text snippet

    Returns:
        Normalized period string, or original if normalization fails
    """
    # Build concise context for the agent
    if raw_period:
        context_parts = [f"Period: '{raw_period}'"]
    else:
        context_parts = ["Period: MISSING (Please infer from context/date)"]
        
    if company:
        context_parts.append(f"Company: {company}")
    if published_at:
        context_parts.append(f"Published: {published_at}")
    if statement_text:
        context_parts.append(f"Context: {statement_text[:150]}")

    prompt = f"""Normalize to standard format (or infer if missing):
{chr(10).join(context_parts)}

Return ONLY the normalized period (e.g., "Q1 FY2025", "FY2024", "H1 FY2025")."""

    try:
        response = agent.chat(prompt)
        normalized = str(response).strip()

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

        # If no standard format found, return cleaned response
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

