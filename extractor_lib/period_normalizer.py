"""
Period normalization tool for LLM-based guidance extraction.

Provides a function-calling tool that reasoning models can use to normalize
reporting periods into consistent formats.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
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


def normalize_period_tool(
    raw_period: str,
    company: str = "",
    published_at: str = "",
    statement_text: str = ""
) -> str:
    """
    Normalize a reporting period string to standardized format.

    This tool is designed to be called by an LLM agent during extraction.
    The LLM should use its reasoning capabilities along with the metadata
    provided to make intelligent normalization decisions.

    STANDARDIZATION RULES:

    1. FISCAL YEAR FORMAT:
       - Use: "FY2025", "FY2024", etc.
       - NOT: "FY 2025", "Fiscal Year 2025", "fiscal 2025"

    2. FISCAL QUARTER FORMAT:
       - Use: "Q1 FY2025", "Q2 FY2024", etc. (with space)
       - NOT: "Q1FY2025", "FY25 Q1", "First Quarter 2025"

    3. CALENDAR YEAR:
       - Use: "2025", "2024" (only when explicitly calendar-based)
       - Most guidance refers to FISCAL years, not calendar

    4. IMPLICIT FISCAL YEAR RULE (CRITICAL):
       - "Q1 2025", "Q2 2024" etc. ALWAYS means fiscal quarter
       - Convert to: "Q1 FY2025", "Q2 FY2024"
       - This is standard corporate guidance convention

    5. HALF-YEAR:
       - Use: "H1 FY2025", "H2 FY2024"
       - "first half 2025" → "H1 FY2025"
       - "second half of the year" → infer year from context

    6. MULTI-YEAR:
       - Use: "FY2021-FY2024" (with hyphen)
       - "2021-2024" → "FY2021-FY2024"

    7. VAGUE TERMS:
       - "next 12 months" → Calculate from published_at, convert to FY range
       - "later this year" → Infer FY from published_at
       - "near term" → "NEAR_TERM" (keep as tag)
       - "multi-year horizon" → "MULTI_YEAR"

    8. SPECIFIC MONTHS:
       - "June 2020" → "FY2020" (if June is FY end) or "June 2020" (keep as is)
       - "November 2021 and November 2022" → "FY2022-FY2023" (infer FY)

    9. NULL/MISSING:
       - If truly ambiguous or missing → return null
       - Only use null as last resort

    COMPANY FISCAL CALENDARS:
    Use the fiscal_calendar metadata when available to resolve ambiguities.
    Common patterns:
    - Microsoft: FY ends June 30
    - Tesla: FY = Calendar year (Dec 31)
    - NVIDIA: FY ends January (last Sunday)
    - Apple: FY ends September

    INFERENCE FROM PUBLISH DATE:
    - If guidance says "Q1" with no year, look at publish_at
    - If published in Aug 2024, "Q1" likely means next Q1 (FY2025 Q1)
    - Use common sense: guidance is forward-looking

    REASONING PROCESS:
    1. Check if period is already in standard format → return as-is
    2. Check for simple patterns (FY2025, Q1 2025) → normalize directly
    3. For vague terms, use published_at and company context
    4. For ambiguous cases, use statement_text for additional clues
    5. Return normalized period with confidence and reasoning

    Args:
        raw_period: The raw period string from extraction
        company: Company name (for fiscal calendar lookup)
        published_at: Document publish date (ISO format)
        statement_text: Surrounding text for context clues

    Returns:
        dict with:
            - normalized_period: Standardized period string
            - confidence: "high", "medium", "low"
            - reasoning: Explanation of normalization decision
            - period_start: ISO date string (optional, for debugging)
            - period_end: ISO date string (optional, for debugging)
    """

    # First try quick normalization
    quick_result = quick_normalize(raw_period)
    if quick_result:
        return quick_result

    # If quick normalization fails, return the raw period with context hint
    # The LLM reasoning model will apply the rules from the docstring
    return raw_period


def get_fiscal_calendar_info(company: str) -> Optional[Dict[str, Any]]:
    """
    Get fiscal calendar information for a company.

    Returns:
        dict with fy_end_month, fy_end_day, or None if not found
    """
    return FISCAL_CALENDARS.get(company)


def create_period_normalization_agent(llm) -> ReActAgent:
    """
    Create a ReActAgent specialized for normalizing complex period strings.

    This agent has access to tools and can reason about ambiguous cases.
    Use this only for periods that quick_normalize() couldn't handle.

    Args:
        llm: The LLM instance to use for the agent

    Returns:
        ReActAgent configured with period normalization tools
    """

    # Tool 1: Fiscal calendar lookup
    def lookup_fiscal_calendar(company: str) -> str:
        """
        Look up the fiscal year end month for a company.

        Args:
            company: Company name or ticker symbol

        Returns:
            String describing the fiscal year end (e.g., "June 30" or "Calendar year (Dec 31)")
        """
        info = get_fiscal_calendar_info(company)
        if info:
            notes = info.get('notes', '')
            return f"{company}: {notes}"
        return f"{company}: Fiscal calendar not found. Assume calendar year (December 31)."

    # Tool 2: Infer fiscal year from date
    def infer_fiscal_year_from_date(company: str, date_str: str) -> str:
        """
        Infer which fiscal year a given date falls into for a company.

        Args:
            company: Company name
            date_str: ISO date string (e.g., "2024-08-21")

        Returns:
            String like "FY2025" indicating the fiscal year
        """
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('T')[0])
            info = get_fiscal_calendar_info(company)

            if not info:
                # Default to calendar year
                return f"FY{date.year}"

            fy_end_month = info.get('fy_end_month', 12)

            # If date is after FY end month, it's in the next FY
            if date.month > fy_end_month:
                return f"FY{date.year + 1}"
            else:
                return f"FY{date.year}"
        except:
            return "Unable to parse date"

    # Tool 3: Quick normalize (wrapped)
    def try_quick_normalize(period: str) -> str:
        """
        Try regex-based normalization for common period formats.

        Args:
            period: Raw period string

        Returns:
            Normalized period or original string if no pattern matched
        """
        result = quick_normalize(period)
        return result if result else f"No quick normalization pattern matched for: {period}"

    # Create FunctionTool instances
    tools = [
        FunctionTool.from_defaults(fn=lookup_fiscal_calendar),
        FunctionTool.from_defaults(fn=infer_fiscal_year_from_date),
        FunctionTool.from_defaults(fn=try_quick_normalize),
    ]

    # System prompt for the agent
    system_prompt = """
You are a financial reporting period normalization specialist.

Your job is to normalize reporting periods from corporate guidance to standard formats.

STANDARD FORMATS:
1. Fiscal Year: "FY2025", "FY2024" (NOT "FY 2025" or "Fiscal Year 2025")
2. Fiscal Quarter: "Q1 FY2025", "Q2 FY2024" (with space)
3. Half-Year: "H1 FY2025", "H2 FY2024"
4. Multi-Year: "FY2021-FY2024"

CRITICAL RULES:
- "Q1 2025", "Q2 2024" etc. ALWAYS means fiscal quarters → "Q1 FY2025", "Q2 FY2024"
- Use tools to look up fiscal calendars and infer dates when needed
- For vague terms like "next 12 months", use the publish date to infer the FY
- Return ONLY the normalized period string, nothing else

You have access to tools to help you:
- lookup_fiscal_calendar: Get a company's fiscal year end
- infer_fiscal_year_from_date: Calculate which FY a date falls into
- try_quick_normalize: Try regex-based normalization first

Process:
1. Try quick_normalize first (already done before you're called)
2. Use tools to gather context (fiscal calendar, date inference)
3. Apply normalization rules
4. Return the normalized period string ONLY
"""

    # Create the agent
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=False,
        max_iterations=5,
        system_prompt=system_prompt
    )

    return agent


def normalize_with_agent(
    agent: ReActAgent,
    raw_period: str,
    company: str = "",
    published_at: str = "",
    statement_text: str = ""
) -> str:
    """
    Use the ReActAgent to normalize a complex period that quick_normalize couldn't handle.

    Args:
        agent: The ReActAgent instance
        raw_period: The raw period string
        company: Company name
        published_at: ISO date string of document publication
        statement_text: Context text snippet

    Returns:
        Normalized period string
    """

    # Build context for the agent
    context_parts = [f"Raw period: '{raw_period}'"]
    if company:
        context_parts.append(f"Company: {company}")
    if published_at:
        context_parts.append(f"Published: {published_at}")
    if statement_text:
        snippet = statement_text[:200] if len(statement_text) > 200 else statement_text
        context_parts.append(f"Context: {snippet}")

    context = "\n".join(context_parts)

    prompt = f"""
Normalize this reporting period to standard format:

{context}

Use your tools if needed. Return ONLY the normalized period (e.g., "Q1 FY2025" or "FY2024").
"""

    try:
        response = agent.chat(prompt)
        normalized = str(response).strip()

        # Clean up any extra text the agent might have added
        # Extract just the period format (FY2025, Q1 FY2025, etc.)
        import re
        match = re.search(r'(FY\d{4}|Q[1-4]\s+FY\d{4}|H[1-2]\s+FY\d{4}|FY\d{4}-FY\d{4}|\w+)', normalized)
        if match:
            return match.group(1)

        return normalized
    except Exception as e:
        print(f"  [WARN] Agent normalization failed: {e}")
        return raw_period  # Return original if agent fails


# Quick regex-based normalization for simple cases
# This can be used as a fast-path before calling the LLM
def quick_normalize(raw_period: str) -> Optional[str]:
    """
    Fast regex-based normalization for obvious cases.
    Returns None if case is too complex and needs LLM reasoning.
    """
    if not raw_period:
        return None

    raw = raw_period.strip()

    # Already standard
    if re.match(r"^FY\d{4}$", raw):
        return raw
    if re.match(r"^Q[1-4]\s+FY\d{4}$", raw):
        return raw

    # FY 2025 → FY2025
    m = re.match(r"^FY\s+(\d{4})$", raw)
    if m:
        return f"FY{m.group(1)}"

    # FY25 → FY2025
    m = re.match(r"^FY(\d{2})$", raw)
    if m:
        year = int(m.group(1))
        full_year = 2000 + year if year >= 0 else 1900 + year
        return f"FY{full_year}"

    # Fiscal Year 2025 → FY2025
    m = re.match(r"^[Ff]iscal\s+[Yy]ear\s+(\d{4})$", raw)
    if m:
        return f"FY{m.group(1)}"

    # Full Year 2025 → FY2025
    m = re.match(r"^[Ff]ull[\s-]?[Yy]ear\s+(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"FY{m.group(1)}"

    # Q1 FY 2025 → Q1 FY2025
    m = re.match(r"^Q([1-4])\s+FY\s+(\d{4})$", raw)
    if m:
        return f"Q{m.group(1)} FY{m.group(2)}"

    # FY25 Q1 → Q1 FY2025
    m = re.match(r"^FY(\d{2})\s+Q([1-4])$", raw)
    if m:
        year = int(m.group(1))
        full_year = 2000 + year
        return f"Q{m.group(2)} FY{full_year}"

    # FY2025 Q1 → Q1 FY2025
    m = re.match(r"^FY(\d{4})\s+Q([1-4])$", raw)
    if m:
        return f"Q{m.group(2)} FY{m.group(1)}"

    # Q1 2025 → Q1 FY2025 (ALWAYS fiscal)
    m = re.match(r"^Q([1-4])\s+(\d{4})$", raw)
    if m:
        return f"Q{m.group(1)} FY{m.group(2)}"

    # First Quarter 2025 → Q1 FY2025
    m = re.match(r"^First\s+Quarter\s+(\d{4})$", raw, re.IGNORECASE)
    if m:
        return f"Q1 FY{m.group(1)}"

    # 2021-2024 → FY2021-FY2024
    m = re.match(r"^(\d{4})\s*-\s*(\d{4})$", raw)
    if m:
        return f"FY{m.group(1)}-FY{m.group(2)}"

    # Too complex - needs LLM
    return None
