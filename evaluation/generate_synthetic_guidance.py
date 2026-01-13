"""
Synthetic Forward-Looking Guidance Event Generator
===================================================

Generates realistic synthetic 8-K earnings documents with or without forward-looking 
guidance statements. Designed to match the distribution of real financial disclosures.

Key Features:
- Realistic company profiles (tech, healthcare, retail, industrial, financial)
- Multiple guidance types with proper distribution weights
- Edge cases: no guidance, qualitative-only, overly-specific (excluded) statements
- Revisions with directional changes
- Standard financial reporting language
- Configurable output volume and distribution
- **LLM-augmented mode** for realistic prose generation using DeepSeek

Usage:
    python generate_synthetic_guidance.py --count 100 --output synthetic_guidance.jsonl
    python generate_synthetic_guidance.py --analyze-distribution ../evaluation/gold_standard_draft.jsonl
    python generate_synthetic_guidance.py --count 50 --use-llm  # Use LLM for text generation
"""

from __future__ import annotations
import json
import random
import argparse
import hashlib
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from uuid import uuid4
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_PATH = Path("evaluation") / "ground_truth" / "synthetic_guidance_output.jsonl"
# LLM imports (optional, only needed for --use-llm mode)
LLM_AVAILABLE = False
try:
    from llm_setup import setup_llm
    LLM_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# CONFIGURABLE DISTRIBUTION WEIGHTS
# =============================================================================
# These can be tuned based on analysis of real gold_standard data

DISTRIBUTION_CONFIG = {
    # Probability that a document contains NO guidance at all (~45% in real data)
    "prob_no_guidance": 0.40,
    
    # Probability of "too specific" forward-looking statements (edge case, rare)
    "prob_too_specific_fls": 0.03,
    
    # When guidance exists, number of items per document (weighted)
    "guidance_count_weights": {
        1: 0.30,  # Single guidance
        2: 0.25,
        3: 0.20,
        4: 0.12,
        5: 0.08,
        6: 0.05,
    },
    
    # Guidance type distribution (based on gold_standard analysis)
    "guidance_type_weights": {
        "revenue": 0.25,
        "EPS": 0.18,
        "capex": 0.12,
        "margin": 0.10,
        "opex": 0.08,
        "earnings": 0.08,
        "ebitda": 0.06,
        "cash_flow": 0.05,
        "other": 0.08,
    },
    
    # Probability that guidance is qualitative-only (no numeric range)
    "prob_qualitative_only": 0.12,
    
    # Probability that guidance is a revision of prior guidance
    "prob_is_revision": 0.18,
    
    # Probability of having a single-value vs range
    "prob_range_guidance": 0.55,
    
    # Reporting period distribution
    "reporting_period_weights": {
        "quarterly": 0.45,
        "annual": 0.45,
        "multi_year": 0.10,
    },
    
    # Industry sector distribution
    "sector_weights": {
        "technology": 0.30,
        "healthcare": 0.15,
        "consumer_discretionary": 0.15,
        "financials": 0.12,
        "industrials": 0.10,
        "consumer_staples": 0.08,
        "energy": 0.05,
        "utilities": 0.05,
    },
}

# =============================================================================
# COMPANY PROFILE DATA
# =============================================================================

SECTOR_COMPANIES = {
    "technology": [
        {"name": "TechVision Corp", "ticker": "TVCO", "revenue_scale": 50, "industry": "Software"},
        {"name": "CloudMatrix Inc", "ticker": "CMTX", "revenue_scale": 30, "industry": "Cloud Computing"},
        {"name": "SynergyAI Holdings", "ticker": "SYAI", "revenue_scale": 15, "industry": "AI/ML"},
        {"name": "DataStream Technologies", "ticker": "DSTM", "revenue_scale": 8, "industry": "Data Analytics"},
        {"name": "CyberShield Systems", "ticker": "CYSS", "revenue_scale": 5, "industry": "Cybersecurity"},
        {"name": "NexGen Semiconductors", "ticker": "NXGS", "revenue_scale": 25, "industry": "Semiconductors"},
        {"name": "Quantum Networks Ltd", "ticker": "QNET", "revenue_scale": 3, "industry": "Networking"},
    ],
    "healthcare": [
        {"name": "BioGenesis Pharmaceuticals", "ticker": "BGPH", "revenue_scale": 12, "industry": "Biotech"},
        {"name": "MedDevice Innovations", "ticker": "MDVI", "revenue_scale": 8, "industry": "Medical Devices"},
        {"name": "HealthFirst Solutions", "ticker": "HFST", "revenue_scale": 20, "industry": "Healthcare Services"},
        {"name": "NeuraTech Therapeutics", "ticker": "NRTH", "revenue_scale": 2, "industry": "Biotech"},
        {"name": "CardioMed Systems", "ticker": "CRMD", "revenue_scale": 6, "industry": "Medical Devices"},
    ],
    "consumer_discretionary": [
        {"name": "RetailMax Holdings", "ticker": "RTMX", "revenue_scale": 45, "industry": "Retail"},
        {"name": "HomeComfort Brands", "ticker": "HCBR", "revenue_scale": 15, "industry": "Home Improvement"},
        {"name": "AutoDrive Motors", "ticker": "ADRV", "revenue_scale": 35, "industry": "Automotive"},
        {"name": "LuxuryGoods International", "ticker": "LXGI", "revenue_scale": 10, "industry": "Luxury Retail"},
        {"name": "TravelEase Corp", "ticker": "TVEZ", "revenue_scale": 8, "industry": "Travel & Leisure"},
    ],
    "financials": [
        {"name": "Meridian Capital Group", "ticker": "MRCG", "revenue_scale": 30, "industry": "Banking"},
        {"name": "SecureVest Insurance", "ticker": "SVST", "revenue_scale": 18, "industry": "Insurance"},
        {"name": "AssetPrime Management", "ticker": "ASPM", "revenue_scale": 12, "industry": "Asset Management"},
        {"name": "FintechFirst Inc", "ticker": "FTFI", "revenue_scale": 5, "industry": "Fintech"},
    ],
    "industrials": [
        {"name": "IndustrialForce Corp", "ticker": "IFRC", "revenue_scale": 25, "industry": "Manufacturing"},
        {"name": "LogiTrans Holdings", "ticker": "LGTR", "revenue_scale": 18, "industry": "Logistics"},
        {"name": "AeroDynamics Inc", "ticker": "ADYN", "revenue_scale": 40, "industry": "Aerospace"},
        {"name": "ConstructionPro Group", "ticker": "CPGR", "revenue_scale": 12, "industry": "Construction"},
    ],
    "consumer_staples": [
        {"name": "FoodWorks International", "ticker": "FWKI", "revenue_scale": 22, "industry": "Food Products"},
        {"name": "BeverageCraft Corp", "ticker": "BVCR", "revenue_scale": 15, "industry": "Beverages"},
        {"name": "HouseholdEssentials Inc", "ticker": "HSEI", "revenue_scale": 10, "industry": "Household Products"},
    ],
    "energy": [
        {"name": "RenewPower Solutions", "ticker": "RPWS", "revenue_scale": 8, "industry": "Renewable Energy"},
        {"name": "PetroGlobal Resources", "ticker": "PGLR", "revenue_scale": 35, "industry": "Oil & Gas"},
        {"name": "CleanEnergy Dynamics", "ticker": "CLED", "revenue_scale": 5, "industry": "Clean Energy"},
    ],
    "utilities": [
        {"name": "PowerGrid Utilities", "ticker": "PGUT", "revenue_scale": 15, "industry": "Electric Utilities"},
        {"name": "AquaServe Holdings", "ticker": "AQSV", "revenue_scale": 6, "industry": "Water Utilities"},
    ],
}

# =============================================================================
# METRIC TEMPLATES BY GUIDANCE TYPE
# =============================================================================

METRIC_TEMPLATES = {
    "revenue": {
        "metrics": [
            "total revenue", "net sales", "sales", "organic revenue", "product revenue",
            "service revenue", "subscription revenue", "recurring revenue", "segment revenue",
            "comparable sales growth", "same-store sales", "net revenue", "gross revenue",
        ],
        "units": ["billion", "million", "%"],
        "value_ranges": {  # by company revenue scale
            "small": (0.5, 5),      # <10B revenue
            "medium": (5, 30),      # 10-30B revenue  
            "large": (30, 150),     # >30B revenue
        },
        "growth_pct_range": (-5, 25),
    },
    "EPS": {
        "metrics": [
            "earnings per share", "adjusted earnings per share", "diluted EPS",
            "adjusted diluted EPS", "GAAP EPS", "non-GAAP EPS", "basic EPS",
        ],
        "units": ["USD", None],
        "value_ranges": {
            "small": (0.10, 2.00),
            "medium": (1.00, 5.00),
            "large": (2.00, 15.00),
        },
    },
    "capex": {
        "metrics": [
            "capital expenditures", "capex", "capital investments", 
            "property and equipment spending", "infrastructure investments",
            "capital spending", "total capital expenditure",
        ],
        "units": ["billion", "million"],
        "value_ranges": {
            "small": (0.2, 2),
            "medium": (2, 10),
            "large": (10, 80),
        },
    },
    "margin": {
        "metrics": [
            "operating margin", "gross margin", "EBITDA margin", "net margin",
            "adjusted operating margin", "profit margin", "contribution margin",
            "adjusted gross margin",
        ],
        "units": ["%"],
        "value_ranges": {
            "small": (5, 40),
            "medium": (10, 50),
            "large": (15, 60),
        },
    },
    "opex": {
        "metrics": [
            "operating expenses", "total expenses", "SG&A", "R&D expenses",
            "selling expenses", "administrative expenses", "total operating costs",
            "restructuring charges", "severance charges", "exit costs",
        ],
        "units": ["billion", "million"],
        "value_ranges": {
            "small": (0.5, 5),
            "medium": (5, 25),
            "large": (25, 100),
        },
    },
    "earnings": {
        "metrics": [
            "operating income", "net income", "adjusted operating profit",
            "EBIT", "pre-tax income", "adjusted net income", "operating profit",
            "segment operating income", "currency-neutral operating profit",
        ],
        "units": ["billion", "million"],
        "value_ranges": {
            "small": (0.1, 2),
            "medium": (2, 10),
            "large": (10, 50),
        },
    },
    "ebitda": {
        "metrics": [
            "EBITDA", "adjusted EBITDA", "segment EBITDA", "consolidated EBITDA",
        ],
        "units": ["billion", "million"],
        "value_ranges": {
            "small": (0.2, 3),
            "medium": (3, 15),
            "large": (15, 80),
        },
    },
    "cash_flow": {
        "metrics": [
            "operating cash flow", "free cash flow", "cash flow from operations",
            "adjusted free cash flow", "cash generation",
        ],
        "units": ["billion", "million"],
        "value_ranges": {
            "small": (0.1, 2),
            "medium": (2, 12),
            "large": (12, 60),
        },
    },
    "other": {
        "metrics": [
            "effective tax rate", "share repurchases", "dividend payout",
            "debt reduction", "headcount", "store count", "subscriber growth",
            "customer acquisition", "market share", "production volume",
            "backlog", "order book", "deliveries", "unit sales",
        ],
        "units": ["%", "million", "billion", "units", None],
        "value_ranges": {
            "small": (5, 30),
            "medium": (10, 50),
            "large": (15, 100),
        },
    },
}

# =============================================================================
# TEXT GENERATION TEMPLATES
# =============================================================================

QUALITATIVE_DIRECTIONS = ["increase", "decrease", "improve", "decline", "accelerate", "moderate", "stabilize", "flat"]

RATIONALE_TEMPLATES = {
    "positive": [
        "driven by strong demand across our core markets",
        "reflecting continued momentum in our strategic initiatives",
        "supported by favorable market conditions and operational improvements",
        "as we continue to execute on our growth strategy",
        "due to robust customer demand and pricing improvements",
        "driven by our investments in AI and digital transformation",
        "reflecting the strength of our diversified business model",
        "as we capitalize on market share gains",
        "supported by the successful launch of new products",
        "benefiting from our cost optimization initiatives",
    ],
    "negative": [
        "due to challenging macroeconomic conditions",
        "reflecting increased competitive pressures",
        "impacted by supply chain disruptions",
        "due to foreign currency headwinds",
        "as we invest in long-term growth initiatives",
        "reflecting higher input costs",
        "impacted by regulatory changes in key markets",
        "due to softer demand in certain end markets",
        "reflecting integration costs from recent acquisitions",
    ],
    "neutral": [
        "in line with our strategic priorities",
        "consistent with our long-term growth targets",
        "reflecting our balanced approach to growth and profitability",
        "as we continue to invest in innovation",
        "based on current market conditions",
    ],
}

CEO_QUOTE_TEMPLATES = [
    '"{statement}," said {ceo_name}, {ceo_title}.',
    '"{statement}," commented {ceo_name}, {ceo_title}.',
    '{ceo_name}, {ceo_title}, stated: "{statement}"',
    '"We are {sentiment} about {topic}," said {ceo_name}. "{additional}"',
]

PREAMBLES_VERB = [
    "For {period}, we expect",
    "Looking ahead to {period}, we anticipate",
    "For {period}, the company expects",
    "Turning to our outlook for {period}, we project",
    "For the upcoming {period}, we forecast",
    "As we enter {period}, we expect",
    "We estimate for {period}",
]

PREAMBLES_NOUN = [
    "Our outlook for {period} includes",
    "We are providing guidance for {period}:",
    "With respect to {period}, we are targeting",
    "Based on current trends, our guidance for {period} is as follows:",
    "Management anticipates {period} results to reflect",
    "Our guidance for {period} assumes",
]

# Natural language variations for reporting periods
PERIOD_VARIATIONS = {
    "quarterly": [
        "{quarter}",                   # Q3
        "the {quarter} quarter",       # the third quarter
        "the {quarter} quarter of {year}", # the third quarter of 2025
        "{quarter} {year}",            # Q3 2025
        "the coming quarter",          # the coming quarter
        "the next quarter",            # the next quarter
        "the current quarter",         # the current quarter
    ],
    "annual": [
        "FY{year}",                    # FY2025
        "fiscal {year}",               # fiscal 2025
        "fiscal year {year}",          # fiscal year 2025
        "the full year",               # the full year
        "the full fiscal year",        # the full fiscal year
        "the year ending December 31, {year}", # the year ending December 31, 2025
        "{year}",                      # 2025
    ],
    "multi_year": [
        "FY{start}-FY{end}",           # FY2025-FY2027
        "the next three years",        # the next three years
        "the {start} to {end} period", # the 2025 to 2027 period
        "fiscal {start} through {end}", # fiscal 2025 through 2027
    ]
}

REVISION_TEMPLATES = [
    "We are {direction} our prior guidance for {metric} from {old_range} to {new_range}",
    "We are revising our {period} {metric} outlook to {new_range}, {direction} from our previous guidance of {old_range}",
    "{metric} guidance is now expected to be {new_range}, representing a revision {direction_text} our prior outlook",
]

# Edge cases: overly specific forward-looking statements (should NOT be extracted)
TOO_SPECIFIC_FLS_TEMPLATES = [
    "We expect to complete the renovation of our {location} facility by {date}, pending regulatory approvals.",
    "The company anticipates finalizing the {product} patent application in the {timeframe}.",
    "We are targeting completion of our new {facility_type} in {region} by {target_date}.",
    "Management expects the {system_name} integration to be substantially complete by {date}.",
    "We anticipate receiving FDA clearance for our {device_name} device in the {timeframe}.",
]

# Standard earnings press release sections
PRESS_RELEASE_TEMPLATES = {
    "header": """{company} Reports {quarter} {year} Results
{sub_headline}

{city}, {state} – {date} – {company} (NYSE: {ticker}) today announced financial results for the {period_description}.
""",
    "highlights": """
{quarter} {year} Key Financial Highlights:
• Revenue of ${revenue} {revenue_unit}, {revenue_change}
• {earnings_metric} of ${earnings} {earnings_unit}
• {eps_type} of ${eps} per share
{additional_highlights}
""",
    "ceo_statement": """
{ceo_quote}
""",
    "guidance_section": """
{period} Outlook
{guidance_statements}
""",
    "no_guidance_section": """
The company will provide updated financial guidance during its upcoming Investor Day scheduled for {future_date}.
""",
    "results_section": """
{quarter} {year} Results Summary

{performance_paragraph}

{segment_results}
""",
    "safe_harbor": """
Forward-Looking Statements
This press release contains forward-looking statements within the meaning of the Private Securities Litigation Reform Act of 1995. These statements involve known and unknown risks, uncertainties and other factors which may cause actual results to be materially different from those expressed or implied.
""",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def weighted_choice(weights_dict: dict) -> str:
    """Select item based on weight distribution."""
    items = list(weights_dict.keys())
    weights = list(weights_dict.values())
    return random.choices(items, weights=weights, k=1)[0]


def generate_uid(text: str) -> str:
    """Generate deterministic UID from content."""
    return hashlib.sha1(text.encode()).hexdigest()


def get_company_size_category(revenue_scale: float) -> str:
    """Categorize company by revenue scale."""
    if revenue_scale < 10:
        return "small"
    elif revenue_scale < 30:
        return "medium"
    else:
        return "large"


def generate_value_range(guidance_type: str, size_cat: str, is_range: bool, unit: str) -> tuple:
    """Generate realistic value range for guidance."""
    template = METRIC_TEMPLATES[guidance_type]
    
    if "value_ranges" not in template:
        base = random.uniform(5, 50)
    else:
        low, high = template["value_ranges"].get(size_cat, (1, 10))
        base = random.uniform(low, high)
    
    if unit == "%":
        base = random.uniform(5, 35)  # Percentages are typically 5-35%
    
    # Round appropriately
    if base >= 10:
        base = round(base, 1)
    elif base >= 1:
        base = round(base, 2)
    else:
        base = round(base, 2)
    
    if is_range:
        # Generate a range (typically 5-15% spread)
        spread_pct = random.uniform(0.05, 0.15)
        range_low = round(base * (1 - spread_pct / 2), 2)
        range_high = round(base * (1 + spread_pct / 2), 2)
        return (range_low, range_high)
    else:
        return (base, None)


def generate_reporting_period(base_year: int, period_type: str) -> tuple[str, str]:
    """Generate reporting period string with natural language variation.
    Returns: (standard_period, natural_period)
    """
    if period_type == "quarterly":
        q_num = random.randint(1, 4)
        quarter_short = f"Q{q_num}"
        quarter_ordinal = ["first", "second", "third", "fourth"][q_num-1]
        
        standard = f"Q{q_num} FY{base_year}"
        
        template = random.choice(PERIOD_VARIATIONS["quarterly"])
        natural = template.format(quarter=quarter_short, year=base_year).replace(f"the {quarter_short} quarter", f"the {quarter_ordinal} quarter")
        return standard, natural
        
    elif period_type == "annual":
        standard = f"FY{base_year}"
        template = random.choice(PERIOD_VARIATIONS["annual"])
        natural = template.format(year=base_year)
        return standard, natural
        
    else:  # multi_year
        standard = f"FY{base_year}-FY{base_year + 2}"
        template = random.choice(PERIOD_VARIATIONS["multi_year"])
        natural = template.format(start=base_year, end=base_year + 2)
        return standard, natural


def generate_date_string(base_date: datetime, offset_days: int = 0) -> str:
    """Generate formatted date string."""
    target_date = base_date + timedelta(days=offset_days)
    return target_date.strftime("%B %d, %Y")


def compute_published_at_for_period(standard_period: str, natural_period: str, base_year: int) -> datetime:
    """
    Compute a realistic published_at date that's consistent with the reporting period.
    
    The logic:
    - For "current quarter" language: published IN that quarter
    - For "next quarter" language: published in the PREVIOUS quarter
    - For explicit periods like "Q2 FY2026": published early in that quarter or late in prior quarter
    - For annual "FY2026": published early in the fiscal year (Q1)
    - For multi-year: published early in the first year
    
    Returns a datetime in UTC timezone.
    """
    import re
    
    # Detect relative period language
    natural_lower = natural_period.lower()
    is_current = "current" in natural_lower
    is_next = "next" in natural_lower or "coming" in natural_lower
    is_full_year = "full year" in natural_lower or "full fiscal" in natural_lower
    
    # Parse the standard period to get quarter/year info
    # Formats: "Q1 FY2026", "FY2026", "FY2026-FY2028"
    q_match = re.match(r'Q(\d)\s+FY(\d{4})', standard_period)
    fy_match = re.match(r'FY(\d{4})(?:-FY(\d{4}))?', standard_period)
    
    if q_match:
        quarter = int(q_match.group(1))
        year = int(q_match.group(2))
        
        # For calendar-year companies, Q1 = Jan-Mar, Q2 = Apr-Jun, etc.
        # Quarter start months: Q1=Jan(1), Q2=Apr(4), Q3=Jul(7), Q4=Oct(10)
        quarter_start_month = (quarter - 1) * 3 + 1
        
        if is_current:
            # Published during the quarter (middle of the quarter)
            pub_month = quarter_start_month + 1  # Middle month of quarter
            pub_day = random.randint(10, 20)
        elif is_next:
            # Published in the PREVIOUS quarter
            if quarter == 1:
                # Previous quarter is Q4 of prior year
                pub_month = 11  # November
                year -= 1
            else:
                pub_month = quarter_start_month - 2  # Middle of previous quarter
            pub_day = random.randint(10, 25)
        else:
            # Explicit period - published near the start of that quarter
            # (e.g., earnings release at start of quarter with forward guidance)
            pub_month = quarter_start_month
            pub_day = random.randint(5, 20)
        
        # Handle month overflow
        if pub_month > 12:
            pub_month -= 12
            year += 1
        elif pub_month < 1:
            pub_month += 12
            year -= 1
            
        return datetime(year, pub_month, pub_day, tzinfo=timezone.utc)
    
    elif fy_match:
        year = int(fy_match.group(1))
        
        if is_full_year:
            # "Full year" language - published mid-year with annual outlook
            pub_month = random.randint(4, 8)
        else:
            # Annual guidance typically given early in fiscal year (Q1)
            pub_month = random.randint(1, 3)
        
        pub_day = random.randint(10, 25)
        return datetime(year, pub_month, pub_day, tzinfo=timezone.utc)
    
    # Fallback: use base_year with random date in Q1
    return datetime(base_year, random.randint(1, 3), random.randint(10, 25), tzinfo=timezone.utc)


def select_random_ceo_name() -> tuple:
    """Generate plausible CEO name and title."""
    first_names = ["Michael", "Sarah", "David", "Jennifer", "Robert", "Lisa", "James", "Maria", 
                   "William", "Patricia", "Richard", "Elizabeth", "Thomas", "Susan", "John"]
    last_names = ["Chen", "Johnson", "Williams", "Brown", "Martinez", "Anderson", "Thompson",
                  "Garcia", "Miller", "Davis", "Wilson", "Taylor", "Moore", "Jackson", "Lee"]
    titles = ["Chairman and CEO", "CEO", "President and CEO", "Chief Executive Officer"]
    
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    title = random.choice(titles)
    return name, title


def select_random_cfo_name() -> tuple:
    """Generate plausible CFO name and title."""
    first_names = ["Amy", "Brian", "Christine", "Daniel", "Emily", "Frank", "Grace", "Henry"]
    last_names = ["Park", "Smith", "Roberts", "Kim", "Patel", "Murphy", "Collins", "Stewart"]
    titles = ["CFO", "Chief Financial Officer", "Executive Vice President and CFO"]
    
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    title = random.choice(titles)
    return name, title


# =============================================================================
# LLM TEXT GENERATION (DeepSeek Integration)
# =============================================================================

class LLMTextGenerator:
    """
    Uses DeepSeek LLM to generate realistic press release prose.
    The structured guidance data remains deterministic; only prose is LLM-generated.
    """
    
    def __init__(self, provider: str = "deepseek", temperature: float = 0.7, timeout: float = 60.0):
        """Initialize LLM for text generation."""
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM setup not available. Ensure llm_setup.py exists and DEEPSEEK_API_KEY is set.")
        
        self.llm = setup_llm(provider=provider, temperature=temperature, timeout=timeout)
        self.cache = {}  # Simple cache to avoid repeated calls for similar inputs
        print(f"[LLM TextGen] Initialized with {provider} (temperature={temperature}, timeout={timeout}s)")
    
    def _call_llm(self, prompt: str, cache_key: str = None) -> str:
        """Call LLM with optional caching and error handling."""
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            response = self.llm.complete(prompt)
            result = response.text.strip()
            
            if cache_key:
                self.cache[cache_key] = result
            
            return result
        except KeyboardInterrupt:
            raise  # Re-raise keyboard interrupt
        except Exception as e:
            print(f"[LLM TextGen] Error calling LLM: {type(e).__name__}: {e}")
            return None
    
    def generate_ceo_statement(self, company: dict, quarter: str, year: int, 
                               has_guidance: bool, performance: str = "strong") -> str:
        """Generate a realistic CEO quote for earnings release."""
        
        prompt = f"""You are writing an earnings press release for {company['name']} ({company['ticker']}), 
a {company['industry']} company in the {company['sector']} sector with ~${company['revenue_scale']}B in annual revenue.

Generate a CEO quote for the {quarter} quarter {year} earnings announcement.
Performance was {performance}.
{"Include forward-looking optimism about guidance." if has_guidance else "Focus on operational achievements, no forward guidance."}

Requirements:
- 2-3 sentences maximum
- Professional, formal tone like real 8-K filings
- Include the CEO's perspective on the quarter
- Do NOT include the CEO's name (it will be added later)
- Return ONLY the quote text, in quotes

Example format:
"We delivered another strong quarter with solid execution across our strategic initiatives. Our team's focus on operational excellence continues to drive sustainable growth."
"""
        
        result = self._call_llm(prompt, cache_key=f"ceo_{company['ticker']}_{quarter}_{year}_{has_guidance}")
        
        # Fallback to template if LLM fails
        if not result:
            return f'"We are pleased with our {quarter.lower()} quarter results, which reflect strong execution and disciplined capital allocation."'
        
        # Ensure it's properly quoted
        result = result.strip('"').strip("'")
        return f'"{result}"'
    
    def generate_guidance_statement(self, guidance_item: dict, company: dict) -> str:
        """Generate natural language for a single guidance item."""
        
        # Build context about the guidance
        metric = guidance_item.get("metric_name", "revenue")
        # Use natural language period for text generation if available, else fallback to standard
        period = guidance_item.get("original_period_text") or guidance_item.get("reporting_period", "FY2025")
        low = guidance_item.get("guided_range_low")
        high = guidance_item.get("guided_range_high")
        unit = guidance_item.get("unit", "")
        qualitative = guidance_item.get("qualitative_direction")
        is_revision = guidance_item.get("is_revision", False)
        revision_dir = guidance_item.get("revision_direction", "")
        rationale = guidance_item.get("rationales", "")
        
        if qualitative and not low:
            # Qualitative-only guidance
            prompt = f"""Generate a forward-looking guidance statement for an 8-K filing.

Company: {company['name']} ({company['sector']})
Metric: {metric}
Period: {period}
Direction: expected to {qualitative}
Rationale: {rationale}

Requirements:
- One sentence, formal financial language
- Include the direction and rationale naturally
- Start with "We expect" or "Management anticipates" or similar
- Return ONLY the statement, no quotes

Example: "We expect operating expenses to increase year-over-year as we continue to invest in AI research and development capabilities."
"""
        else:
            # Quantitative guidance
            if high and high != low:
                range_text = f"${low} to ${high} {unit}"
            else:
                range_text = f"approximately ${low} {unit}"
            
            revision_text = f" This represents a revision {revision_dir} from prior guidance." if is_revision else ""
            
            prompt = f"""Generate a forward-looking guidance statement for an 8-K filing.

Company: {company['name']} ({company['sector']})
Metric: {metric}
Period: {period}
Expected Range: {range_text}
{"Rationale: " + rationale if rationale else ""}
{revision_text}

Requirements:
- One sentence, formal financial language  
- Include the exact numbers provided
- Use phrasing like "we expect", "we anticipate", "guidance is"
- Return ONLY the statement, no quotes

Example: "For fiscal year 2025, we expect total revenue to be in the range of $45.0 to $47.0 billion, reflecting continued demand strength in our cloud segment."
"""
        
        result = self._call_llm(prompt)
        
        # Fallback to simple template
        if not result:
            # Handle metric casing - Use as is
            metric_display = metric
            
            if qualitative:
                preamble = random.choice(PREAMBLES_VERB).format(period=period)
                stmt = f"{preamble} {metric_display} to {qualitative} {rationale}."
                return stmt.replace("upcoming the", "upcoming")
            
            # Quantitative
            use_verb_structure = random.choice([True, False])
            
            if use_verb_structure:
                preamble = random.choice(PREAMBLES_VERB).format(period=period)
                verb = random.choice(["to be", "to come in at", "to reach"])
                connector = verb
            else:
                preamble = random.choice(PREAMBLES_NOUN).format(period=period)
                connector = "of"

            if high and high != low:
                range_str = f"${low} to ${high} {unit}"
                if use_verb_structure:
                    if "come in at" in connector:
                        phrase = random.choice([
                            f"{connector} between {range_str}",
                            f"{connector} {range_str}"
                        ])
                    else:
                        phrase = random.choice([
                            f"{connector} in the range of {range_str}",
                            f"{connector} between {range_str}"
                        ])
                else:
                    phrase = f"{connector} {range_str}"
            else:
                val_str = f"${low} {unit}"
                phrase = f"{connector} approximately {val_str}"
            
            stmt = f"{preamble} {metric_display} {phrase}."
            return stmt.replace("upcoming the", "upcoming")
        
        return result.strip('"').strip()
    
    def generate_press_release_body(self, company: dict, quarter: str, year: int,
                                    financial_highlights: dict, 
                                    guidance_items: list = None,
                                    has_too_specific_fls: bool = False) -> str:
        """Generate the full body of a press release with realistic prose."""
        
        # Build structured context for the LLM
        guidance_context = ""
        if guidance_items:
            guidance_bullets = []
            for item in guidance_items:
                stmt = self.generate_guidance_statement(item, company)
                guidance_bullets.append(f"• {stmt}")
            guidance_context = "\n".join(guidance_bullets)
        
        too_specific_context = ""
        if has_too_specific_fls:
            too_specific_context = """
Additionally, include one overly-specific forward-looking statement that mentions a very specific operational detail 
(like completing a facility renovation in a specific city by a specific date, or receiving a specific regulatory approval).
This should be something too granular to be considered financial guidance."""
        
        prompt = f"""Generate an 8-K earnings press release for {company['name']} (NYSE: {company['ticker']}).

Company Details:
- Industry: {company['industry']}
- Sector: {company['sector']}  
- Annual Revenue Scale: ~${company['revenue_scale']}B

Quarter: {quarter} {year}
Financial Highlights:
- Revenue: ${financial_highlights['revenue']:.1f}B ({financial_highlights['revenue_growth']:+.0f}% YoY)
- EPS: ${financial_highlights['eps']:.2f}
- Operating Income: ${financial_highlights['operating_income']:.2f}B

{"Forward Guidance (include these EXACTLY as provided):" + chr(10) + guidance_context if guidance_context else "Do NOT include any forward-looking guidance. The company will provide guidance at a later date."}
{too_specific_context}

Generate a complete press release body including:
1. Opening paragraph with key results
2. CEO quote (2-3 sentences, attribute to CEO name placeholder [CEO_NAME])
3. Financial highlights bullet points
4. {"Business Outlook section with the guidance statements above" if guidance_items else "Note that guidance will be provided at upcoming investor day"}
5. Standard safe harbor statement (abbreviated)

Requirements:
- Professional, formal SEC filing language
- 400-600 words
- Include all numerical guidance exactly as provided
- Use realistic financial terminology
- Structure with clear sections

Return the press release text only, starting with the headline.
"""
        
        result = self._call_llm(prompt)
        
        if not result:
            # Return None to trigger fallback to template generation
            return None
        
        return result
    
    def generate_too_specific_fls(self, company: dict, base_date: datetime) -> str:
        """Generate an overly-specific forward-looking statement (edge case)."""
        
        prompt = f"""Generate an overly-specific forward-looking statement for {company['name']} ({company['industry']}).

This should be a statement about a very specific operational milestone that would NOT qualify as 
financial guidance for extraction. Examples:
- Completing a facility in a specific city by a specific date
- Receiving a specific regulatory approval for a named product
- Finishing an IT system migration
- Opening a specific number of stores in a named region

Requirements:
- One sentence only
- Include specific names, dates, or locations
- This is meant to test that extractors do NOT pick this up as financial guidance
- Return ONLY the statement

Example: "We expect to complete the renovation of our Austin manufacturing facility by Q2 2026, pending final municipal approvals."
"""
        
        result = self._call_llm(prompt)
        
        if not result:
            location = random.choice(["Austin", "Singapore", "Munich", "Tokyo", "Dublin"])
            return f"We expect to complete our new {location} facility by Q{random.randint(1,4)} {base_date.year + 1}."
        
        return result.strip('"').strip()


# =============================================================================
# MAIN GENERATION CLASSES
# =============================================================================

@dataclass
class SyntheticGuidanceItem:
    """Single guidance statement matching the schema.
    
    Note: Document-level metadata (source_url, published_at, ingested_at) are stored
    at the document level (SyntheticDocument), not duplicated per guidance item.
    """
    guid: str = field(default_factory=lambda: uuid4().hex)
    ticker: Optional[str] = None  # Stock ticker symbol (e.g., AAPL, MSFT)
    guidance_type: Optional[str] = None
    metric_name: Optional[str] = None
    reporting_period: Optional[str] = None
    original_period_text: Optional[str] = None  # Natural language version used in text
    current_value: Optional[float] = None
    unit: Optional[str] = None
    guided_range_low: Optional[float] = None
    guided_range_high: Optional[float] = None
    is_revision: Optional[bool] = None
    revision_direction: Optional[str] = None
    qualitative_direction: Optional[str] = None
    rationales: Optional[str] = None
    source_type: str = "8-K"
    extracted_at: Optional[str] = None
    processing_duration_seconds: Optional[float] = None
    extraction_method: str = "synthetic"
    was_updated_by_agent: bool = False


@dataclass
class SyntheticDocument:
    """Complete synthetic 8-K document with guidance."""
    uid: str
    source_id: str
    source_url: str
    title: str
    full_text_snippet: str
    gold_standard_guidance: list
    synthetic_metadata: dict  # Extra info about generation
    company_name: str = ""  # Company name for extraction
    published_at: str = ""  # ISO format publication date
    manual_verification_status: str = "synthetic"
    manual_notes: str = ""


class SyntheticGuidanceGenerator:
    """
    Main generator class for creating synthetic forward-looking guidance events.
    Supports both template-based and LLM-augmented text generation.
    """
    
    def __init__(self, config: dict = None, seed: int = None, use_llm: bool = False, 
                 llm_provider: str = "deepseek", llm_temperature: float = 0.7, llm_timeout: float = 60.0):
        self.config = config or DISTRIBUTION_CONFIG
        if seed is not None:
            random.seed(seed)
        
        self.base_year = datetime.now().year
        self.base_date = datetime.now(timezone.utc)
        
        # LLM text generation (optional)
        self.use_llm = use_llm
        self.llm_generator = None
        if use_llm:
            try:
                self.llm_generator = LLMTextGenerator(
                    provider=llm_provider, 
                    temperature=llm_temperature,
                    timeout=llm_timeout,
                )
                print(f"[Generator] LLM text generation enabled")
            except Exception as e:
                print(f"[Generator] Warning: LLM initialization failed: {e}")
                print(f"[Generator] Falling back to template-based generation")
                self.use_llm = False
    
    def _select_company(self) -> dict:
        """Select a company profile based on sector weights."""
        sector = weighted_choice(self.config["sector_weights"])
        company = random.choice(SECTOR_COMPANIES[sector])
        company["sector"] = sector
        return company
    
    def _generate_guidance_item(
        self,
        company: dict,
        guidance_type: str,
        is_qualitative: bool,
        is_revision: bool,
        period_type: str,
    ) -> SyntheticGuidanceItem:
        """Generate a single guidance item."""
        
        template = METRIC_TEMPLATES.get(guidance_type, METRIC_TEMPLATES["other"])
        metric_name = random.choice(template["metrics"])
        unit = random.choice(template.get("units", [None]))
        
        size_cat = get_company_size_category(company["revenue_scale"])
        standard_period, natural_period = generate_reporting_period(self.base_year, period_type)
        
        item = SyntheticGuidanceItem(
            ticker=company["ticker"],
            guidance_type=guidance_type,
            metric_name=metric_name,
            reporting_period=standard_period,
            original_period_text=natural_period,
            source_type="8-K",
            extracted_at=self.base_date.isoformat(),
        )
        
        if is_qualitative:
            # Qualitative-only guidance (no numeric values)
            item.qualitative_direction = random.choice(QUALITATIVE_DIRECTIONS)
            sentiment = "positive" if item.qualitative_direction in ["increase", "improve", "accelerate"] else (
                "negative" if item.qualitative_direction in ["decrease", "decline"] else "neutral"
            )
            item.rationales = random.choice(RATIONALE_TEMPLATES[sentiment])
        else:
            # Quantitative guidance
            is_range = random.random() < self.config["prob_range_guidance"]
            range_low, range_high = generate_value_range(guidance_type, size_cat, is_range, unit)
            
            item.unit = unit
            item.guided_range_low = range_low
            item.guided_range_high = range_high
            
            # Add rationale sometimes
            if random.random() < 0.6:
                sentiment = random.choice(["positive", "neutral"])
                item.rationales = random.choice(RATIONALE_TEMPLATES[sentiment])
        
        if is_revision:
            item.is_revision = True
            item.revision_direction = random.choice(["increased", "decreased"])
        
        return item
    
    def _generate_too_specific_fls(self, company: dict) -> str:
        """Generate an overly-specific forward-looking statement (edge case)."""
        # Try LLM first if available
        if self.use_llm and self.llm_generator:
            try:
                return self.llm_generator.generate_too_specific_fls(company, self.base_date)
            except Exception as e:
                print(f"[Generator] LLM too-specific FLS failed, using template: {e}")
        
        # Fallback to template
        template = random.choice(TOO_SPECIFIC_FLS_TEMPLATES)
        
        placeholders = {
            "location": random.choice(["Austin", "Singapore", "Munich", "Tokyo", "Dublin"]),
            "date": generate_date_string(self.base_date, random.randint(90, 365)),
            "product": random.choice(["next-generation", "proprietary", "enhanced"]),
            "timeframe": random.choice(["first half of 2026", "second quarter", "coming months"]),
            "facility_type": random.choice(["manufacturing", "R&D", "distribution"]),
            "region": random.choice(["Southeast Asia", "Europe", "Latin America"]),
            "target_date": f"Q{random.randint(1,4)} {self.base_year + 1}",
            "system_name": random.choice(["ERP", "SAP S/4HANA", "Oracle Cloud"]),
            "device_name": random.choice(["CardioMonitor Pro", "NeuraScan", "BioTrack"]),
        }
        
        return template.format(**placeholders)
    
    def _generate_press_release_text_llm(
        self,
        company: dict,
        guidance_items: list,
        has_too_specific_fls: bool,
    ) -> str:
        """Generate press release text using LLM for more realistic prose."""
        
        ceo_name, ceo_title = select_random_ceo_name()
        quarter = random.choice(["First", "Second", "Third", "Fourth"])
        
        # Generate financial highlights (deterministic)
        revenue = company['revenue_scale'] * random.uniform(0.2, 0.35)
        revenue_growth = random.randint(5, 20)
        eps = random.uniform(0.50, 5.00)
        operating_income = company['revenue_scale'] * random.uniform(0.03, 0.12)
        
        financial_highlights = {
            "revenue": revenue,
            "revenue_growth": revenue_growth,
            "eps": eps,
            "operating_income": operating_income,
        }
        
        # Convert guidance items to dicts for LLM
        guidance_dicts = [asdict(item) for item in guidance_items] if guidance_items else None
        
        # Try LLM generation
        try:
            llm_text = self.llm_generator.generate_press_release_body(
                company=company,
                quarter=quarter,
                year=self.base_year,
                financial_highlights=financial_highlights,
                guidance_items=guidance_dicts,
                has_too_specific_fls=has_too_specific_fls,
            )
            
            if llm_text:
                # Replace CEO placeholder
                llm_text = llm_text.replace("[CEO_NAME]", f"{ceo_name}, {ceo_title}")
                
                # Wrap in 8-K format
                full_text = f"""EX-99.1
2
{company['ticker'].lower()}-ex991.htm
EX-99.1
Document
Exhibit 99.1
{llm_text}
"""
                return full_text
        except Exception as e:
            print(f"[Generator] LLM press release failed, using template: {e}")
        
        # Fallback to template-based generation
        return self._generate_press_release_text_template(company, guidance_items, has_too_specific_fls)

    def _generate_press_release_text_template(
        self,
        company: dict,
        guidance_items: list,
        has_too_specific_fls: bool,
    ) -> str:
        """Generate press release text using templates (fallback/default)."""
        
        ceo_name, ceo_title = select_random_ceo_name()
        cfo_name, cfo_title = select_random_cfo_name()
        
        quarter = random.choice(["First", "Second", "Third", "Fourth"])
        quarter_short = quarter[0] + ("1" if quarter == "First" else "2" if quarter == "Second" else "3" if quarter == "Third" else "4")
        
        # Generate header
        header = PRESS_RELEASE_TEMPLATES["header"].format(
            company=company["name"],
            quarter=quarter,
            year=self.base_year,
            sub_headline=random.choice([
                f"— Revenue of ${company['revenue_scale']:.1f} billion, up {random.randint(5, 20)}% year over year —",
                f"— Strong execution drives {random.randint(10, 30)}% earnings growth —",
                f"— Company achieves record quarterly results —",
            ]),
            city=random.choice(["NEW YORK", "SAN FRANCISCO", "CHICAGO", "BOSTON", "SEATTLE"]),
            state=random.choice(["CA", "NY", "IL", "MA", "WA"]),
            date=generate_date_string(self.base_date),
            ticker=company["ticker"],
            period_description=f"the quarter ended {generate_date_string(self.base_date, -30)}",
        )
        
        # Generate highlights
        highlights = PRESS_RELEASE_TEMPLATES["highlights"].format(
            quarter=quarter,
            year=self.base_year,
            revenue=f"{company['revenue_scale'] * random.uniform(0.2, 0.35):.1f}",
            revenue_unit="billion",
            revenue_change=f"up {random.randint(5, 20)}% year over year",
            earnings_metric=random.choice(["Operating income", "Net income", "Adjusted operating profit"]),
            earnings=f"{company['revenue_scale'] * random.uniform(0.03, 0.12):.2f}",
            earnings_unit="billion",
            eps_type=random.choice(["Diluted EPS", "Adjusted EPS", "GAAP EPS"]),
            eps=f"{random.uniform(0.50, 5.00):.2f}",
            additional_highlights=random.choice([
                f"• Cloud revenue increased {random.randint(15, 40)}% year over year",
                f"• Free cash flow of ${random.uniform(0.5, 3.0):.1f} billion",
                f"• Returned ${random.uniform(0.5, 2.0):.1f} billion to shareholders through buybacks and dividends",
            ]) if random.random() > 0.3 else "",
        )
        
        # CEO statement
        ceo_sentiments = [
            f"We delivered another quarter of strong results, demonstrating the durability of our business model",
            f"Our {quarter.lower()} quarter performance reflects solid execution across our key initiatives",
            f"I'm pleased with our progress this quarter as we continue to invest in long-term growth",
            f"The strength of our diversified portfolio drove outstanding results this quarter",
        ]
        ceo_quote = PRESS_RELEASE_TEMPLATES["ceo_statement"].format(
            ceo_quote=f'"{random.choice(ceo_sentiments)}," said {ceo_name}, {ceo_title}.',
        )
        
        # Guidance section
        if guidance_items:
            guidance_statements = []
            for item in guidance_items:
                # Handle metric casing - Use as is since generator provides good casing
                metric_display = item.metric_name
                
                # Use natural language period for text generation if available
                period = item.original_period_text or item.reporting_period
                low = item.guided_range_low
                high = item.guided_range_high
                unit = item.unit or ''
                qualitative = item.qualitative_direction
                rationale = item.rationales
                
                if qualitative:
                    preamble = random.choice(PREAMBLES_VERB).format(period=period)
                    stmt = f"• {preamble} {metric_display} to {qualitative}"
                    if rationale:
                        stmt += f", {rationale}"
                else:
                    # Quantitative
                    use_verb_structure = random.choice([True, False])
                    
                    if use_verb_structure:
                        preamble = random.choice(PREAMBLES_VERB).format(period=period)
                        verb = random.choice(["to be", "to come in at", "to reach"])
                        connector = verb
                    else:
                        preamble = random.choice(PREAMBLES_NOUN).format(period=period)
                        connector = "of"

                    if high and high != low:
                        range_str = f"${low:.2f} to ${high:.2f} {unit}"
                        if use_verb_structure:
                            # Avoid "come in at in the range of"
                            if "come in at" in connector:
                                phrase = random.choice([
                                    f"{connector} between {range_str}",
                                    f"{connector} {range_str}"
                                ])
                            else:
                                phrase = random.choice([
                                    f"{connector} in the range of {range_str}",
                                    f"{connector} between {range_str}"
                                ])
                        else:
                            phrase = f"{connector} {range_str}"
                    else:
                        val_str = f"${low:.2f} {unit}"
                        phrase = f"{connector} approximately {val_str}"
                    
                    stmt = f"• {preamble} {metric_display} {phrase}"
                
                # Fix grammar issues with "the"
                stmt = stmt.replace("upcoming the", "upcoming")
                
                if item.is_revision:
                    stmt += f" ({item.revision_direction} from prior guidance)"
                
                guidance_statements.append(stmt)
            
            guidance_section = PRESS_RELEASE_TEMPLATES["guidance_section"].format(
                period=f"FY{self.base_year}",
                guidance_statements="\n".join(guidance_statements),
            )
        else:
            guidance_section = PRESS_RELEASE_TEMPLATES["no_guidance_section"].format(
                future_date=generate_date_string(self.base_date, random.randint(30, 90)),
            )
        
        # Add too-specific FLS if applicable
        too_specific_text = ""
        if has_too_specific_fls:
            too_specific_text = f"\n\nOther Updates:\n{self._generate_too_specific_fls(company)}\n"
        
        # Assemble full text
        full_text = f"""EX-99.1
2
{company['ticker'].lower()}-ex991.htm
EX-99.1
Document
Exhibit 99.1
{header}
{highlights}
{ceo_quote}
{guidance_section}
{too_specific_text}
{PRESS_RELEASE_TEMPLATES["safe_harbor"]}
"""
        return full_text
    
    def _generate_press_release_text(
        self,
        company: dict,
        guidance_items: list,
        has_too_specific_fls: bool,
    ) -> str:
        """Generate press release text - routes to LLM or template based on config."""
        if self.use_llm and self.llm_generator:
            return self._generate_press_release_text_llm(company, guidance_items, has_too_specific_fls)
        else:
            return self._generate_press_release_text_template(company, guidance_items, has_too_specific_fls)
    
    def generate_document(self) -> SyntheticDocument:
        """Generate a complete synthetic document with or without guidance."""
        
        company = self._select_company()
        
        # Determine if this document has guidance
        has_guidance = random.random() > self.config["prob_no_guidance"]
        has_too_specific_fls = random.random() < self.config["prob_too_specific_fls"]
        
        guidance_items = []
        document_published_at = None  # Will be set based on first guidance item
        
        if has_guidance:
            # Determine number of guidance items
            num_items = int(weighted_choice(self.config["guidance_count_weights"]))
            
            # Generate each guidance item
            used_types = set()
            for _ in range(num_items):
                # Select guidance type (avoid duplicates when possible)
                attempts = 0
                while attempts < 10:
                    guidance_type = weighted_choice(self.config["guidance_type_weights"])
                    if guidance_type not in used_types or attempts >= 5:
                        break
                    attempts += 1
                used_types.add(guidance_type)
                
                is_qualitative = random.random() < self.config["prob_qualitative_only"]
                is_revision = random.random() < self.config["prob_is_revision"]
                period_type = weighted_choice(self.config["reporting_period_weights"])
                
                item = self._generate_guidance_item(
                    company=company,
                    guidance_type=guidance_type,
                    is_qualitative=is_qualitative,
                    is_revision=is_revision,
                    period_type=period_type,
                )
                
                # Compute published_at based on the period (first item sets document date)
                if document_published_at is None:
                    document_published_at = compute_published_at_for_period(
                        item.reporting_period,
                        item.original_period_text or item.reporting_period,
                        self.base_year
                    )
                
                # Note: published_at is stored at document level, not per guidance item
                
                guidance_items.append(item)
        else:
            # No guidance - still need a publication date for the document
            document_published_at = datetime(
                self.base_year, 
                random.randint(1, 12), 
                random.randint(5, 25), 
                tzinfo=timezone.utc
            )
        
        # Generate press release text
        full_text = self._generate_press_release_text(
            company=company,
            guidance_items=guidance_items,
            has_too_specific_fls=has_too_specific_fls,
        )
        
        # Create document
        uid = generate_uid(full_text + str(random.random()))
        source_id = f"{company['ticker'].lower()}_8k_synthetic"
        
        doc = SyntheticDocument(
            uid=uid,
            source_id=source_id,
            source_url=f"https://synthetic.sec.gov/Archives/edgar/data/{random.randint(100000, 999999)}/synthetic-8k.htm",
            title="8-K  - Current report",
            full_text_snippet=full_text,  # No truncation
            gold_standard_guidance=[asdict(item) for item in guidance_items],
            synthetic_metadata={
                "company": company,
                "has_too_specific_fls": has_too_specific_fls,
                "generation_timestamp": self.base_date.isoformat(),
                "generator_version": "1.2.0",  # Version bump for published_at support
                "llm_augmented": self.use_llm,
            },
            company_name=company["name"],
            published_at=document_published_at.isoformat() if document_published_at else "",
        )
        
        return doc
    
    def generate_dataset(self, count: int) -> list[SyntheticDocument]:
        """Generate multiple synthetic documents with progress tracking."""
        documents = []
        for i in range(count):
            # We don't know the company yet, so we'll just show the index
            doc = self.generate_document()
            company_name = doc.synthetic_metadata["company"]["name"]
            print(f"  [{i+1}/{count}] Generated: {company_name:<30}", flush=True)
            documents.append(doc)
        return documents


# =============================================================================
# DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_gold_standard_distribution(filepath: str) -> dict:
    """
    Analyze the distribution of guidance in a gold standard file.
    Returns statistics that can be used to calibrate the generator.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        documents = [json.loads(line) for line in f if line.strip()]
    
    stats = {
        "total_documents": len(documents),
        "documents_with_guidance": 0,
        "documents_without_guidance": 0,
        "total_guidance_items": 0,
        "guidance_counts": {},
        "guidance_types": {},
        "has_revision": 0,
        "is_qualitative_only": 0,
        "has_range": 0,
        "reporting_periods": {},
    }
    
    for doc in documents:
        guidance = doc.get("gold_standard_guidance", [])
        
        if guidance:
            stats["documents_with_guidance"] += 1
            count = len(guidance)
            stats["guidance_counts"][count] = stats["guidance_counts"].get(count, 0) + 1
            
            for item in guidance:
                stats["total_guidance_items"] += 1
                
                # Count guidance types
                gtype = item.get("guidance_type", "unknown")
                stats["guidance_types"][gtype] = stats["guidance_types"].get(gtype, 0) + 1
                
                # Track revisions
                if item.get("is_revision"):
                    stats["has_revision"] += 1
                
                # Track qualitative vs quantitative
                if item.get("qualitative_direction") and not item.get("guided_range_low"):
                    stats["is_qualitative_only"] += 1
                
                # Track ranges
                if item.get("guided_range_high"):
                    stats["has_range"] += 1
                
                # Track periods
                period = item.get("reporting_period", "unknown")
                period_type = "quarterly" if period and period.startswith("Q") else "annual" if period and "FY" in period else "other"
                stats["reporting_periods"][period_type] = stats["reporting_periods"].get(period_type, 0) + 1
        else:
            stats["documents_without_guidance"] += 1
    
    # Calculate percentages
    stats["pct_no_guidance"] = stats["documents_without_guidance"] / stats["total_documents"] if stats["total_documents"] > 0 else 0
    stats["pct_with_revision"] = stats["has_revision"] / stats["total_guidance_items"] if stats["total_guidance_items"] > 0 else 0
    stats["pct_qualitative_only"] = stats["is_qualitative_only"] / stats["total_guidance_items"] if stats["total_guidance_items"] > 0 else 0
    stats["pct_has_range"] = stats["has_range"] / stats["total_guidance_items"] if stats["total_guidance_items"] > 0 else 0
    
    # Normalize type distribution
    total_types = sum(stats["guidance_types"].values())
    stats["guidance_type_distribution"] = {k: v / total_types for k, v in stats["guidance_types"].items()} if total_types > 0 else {}
    
    return stats


def print_distribution_analysis(stats: dict):
    """Pretty print distribution analysis."""
    print("\n" + "=" * 60)
    print("GOLD STANDARD DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal Documents: {stats['total_documents']}")
    print(f"  With Guidance: {stats['documents_with_guidance']} ({1 - stats['pct_no_guidance']:.1%})")
    print(f"  Without Guidance: {stats['documents_without_guidance']} ({stats['pct_no_guidance']:.1%})")
    print(f"\nTotal Guidance Items: {stats['total_guidance_items']}")
    
    print("\nGuidance Count Distribution:")
    for count, num in sorted(stats["guidance_counts"].items()):
        print(f"  {count} item(s): {num} documents")
    
    print("\nGuidance Type Distribution:")
    for gtype, pct in sorted(stats["guidance_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {gtype}: {pct:.1%}")
    
    print(f"\nOther Statistics:")
    print(f"  Revisions: {stats['pct_with_revision']:.1%}")
    print(f"  Qualitative-only: {stats['pct_qualitative_only']:.1%}")
    print(f"  Has Range (vs point): {stats['pct_has_range']:.1%}")
    
    print("\nReporting Period Distribution:")
    total_periods = sum(stats["reporting_periods"].values())
    for ptype, count in stats["reporting_periods"].items():
        print(f"  {ptype}: {count / total_periods:.1%}")
    
    print("\n" + "=" * 60)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic forward-looking guidance events"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=50,
        help="Number of synthetic documents to generate (default: 50)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="synthetic_guidance.jsonl",
        help="Output file path (default: synthetic_guidance.jsonl)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--analyze-distribution", "-a",
        type=str,
        default=None,
        metavar="FILE",
        help="Analyze distribution of existing gold standard file instead of generating"
    )
    parser.add_argument(
        "--calibrate",
        type=str,
        default=None,
        metavar="FILE",
        help="Calibrate generator based on gold standard file before generating"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use DeepSeek LLM for more realistic text generation (requires DEEPSEEK_API_KEY)"
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="deepseek",
        choices=["deepseek", "github"],
        help="LLM provider to use (default: deepseek)"
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.7,
        help="LLM temperature for text generation (default: 0.7, higher = more creative)"
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="LLM request timeout in seconds (default: 60.0)"
    )
    
    args = parser.parse_args()
    
    # Analysis mode
    if args.analyze_distribution:
        filepath = Path(args.analyze_distribution)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            return 1
        
        stats = analyze_gold_standard_distribution(str(filepath))
        print_distribution_analysis(stats)
        
        # Print suggested config updates
        print("\nSUGGESTED CONFIG UPDATES:")
        print(f'  "prob_no_guidance": {stats["pct_no_guidance"]:.2f},')
        print(f'  "prob_is_revision": {stats["pct_with_revision"]:.2f},')
        print(f'  "prob_qualitative_only": {stats["pct_qualitative_only"]:.2f},')
        print(f'  "prob_range_guidance": {stats["pct_has_range"]:.2f},')
        
        return 0
    
    # Calibration mode (analyze then generate with updated config)
    config = DISTRIBUTION_CONFIG.copy()
    if args.calibrate:
        filepath = Path(args.calibrate)
        if filepath.exists():
            stats = analyze_gold_standard_distribution(str(filepath))
            print_distribution_analysis(stats)
            
            # Update config based on analysis
            config["prob_no_guidance"] = stats["pct_no_guidance"]
            config["prob_is_revision"] = stats["pct_with_revision"]
            config["prob_qualitative_only"] = stats["pct_qualitative_only"]
            config["prob_range_guidance"] = stats["pct_has_range"]
            
            # Update guidance type weights
            if stats["guidance_type_distribution"]:
                config["guidance_type_weights"] = stats["guidance_type_distribution"]
            
            print("\n✓ Calibrated generator based on gold standard distribution\n")
    
    # Generation mode
    generator = SyntheticGuidanceGenerator(
        config=config, 
        seed=args.seed,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_temperature=args.llm_temperature,
        llm_timeout=args.llm_timeout,
    )
    
    mode_str = "LLM-augmented" if args.use_llm else "template-based"
    print(f"Generating {args.count} synthetic documents ({mode_str})...")
    documents = generator.generate_dataset(args.count)
    
    # Write output
    if Path(args.output):
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_PATH
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + '\n')
    
    # Print summary
    docs_with_guidance = sum(1 for d in documents if d.gold_standard_guidance)
    total_items = sum(len(d.gold_standard_guidance) for d in documents)
    
    print(f"\n✓ Generated {len(documents)} documents")
    print(f"  With guidance: {docs_with_guidance} ({docs_with_guidance/len(documents):.1%})")
    print(f"  Total guidance items: {total_items}")
    print(f"  Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
