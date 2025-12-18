from __future__ import annotations
from typing import Optional, Literal
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Guidance(BaseModel):
    """Pydantic model for a guidance item. Mirrors the previous dataclass fields.

    This model is used as the structured output type for LLM extraction and is
    intentionally compatible with the old dataclass-based shape.
    """
    guid: str = Field(default_factory=lambda: uuid4().hex)
    company: Optional[str] = None
    # NOTE: ticker removed to reduce complexity; can be added later via mapping

    # What guidance is about - constrained to valid categories
    guidance_type: Optional[Literal[
        "revenue",
        "earnings",
        "margin",
        "opex",
        "capex",
        "cash_flow",
        "ebitda",
        "EPS",
        "other"
    ]] = None
    
    # The specific metric name mentioned in text (e.g. "Net Interest Income", "Cloud Revenue")
    metric_name: Optional[str] = None

    # Temporal context
    reporting_period: Optional[str] = None

    # Quantitative data
    current_value: Optional[float] = None
    unit: Optional[Literal["USD", "EUR", "GBP", "%", "million", "billion", "units", "other"]] = None
    # guided_value removed in favor of using guided_range_low for single values
    guided_range_low: Optional[float] = None
    guided_range_high: Optional[float] = None

    # Revision / Qualitative
    is_revision: Optional[bool] = None
    revision_direction: Optional[Literal["increased", "decreased"]] = None
    qualitative_direction: Optional[str] = None
    rationales: Optional[str] = None

    # Metadata
    statement_text: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[Literal["8-K", "10-K", "10-Q", "press_release", "earnings_call", "investor_presentation", "other"]] = None
    
    # Dates
    published_at: Optional[str] = None  # When the document was published by the source
    ingested_at: Optional[str] = None   # When the document was fetched/ingested by our system
    extracted_at: Optional[str] = None  # When this guidance item was extracted (output created)
    
    # Performance metrics
    processing_duration_seconds: Optional[float] = None # Time taken for LLM processing in seconds

    # Track which pipeline stage produced this item (for research comparison)
    extraction_method: Optional[Literal["standard", "agentic_review"]] = "standard"
    
    # Agentic review details (for debugging/analysis)
    agentic_review_comment: Optional[str] = None
    was_updated_by_agent: Optional[bool] = False
    
    # Agentic Enrichment (New for Thesis)
    sentiment_label: Optional[Literal["positive", "negative", "neutral", "cautious", "optimistic"]] = None
    sentiment_score: Optional[float] = None # 0.0 to 1.0 (0=Bearish, 1=Bullish)
    risk_factors: Optional[str] = None # Comma-separated list of risks mentioned (e.g. "FX, Supply Chain")

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_raw(cls, **kwargs) -> "Guidance":
        if "extracted_at" not in kwargs or not kwargs.get("extracted_at"):
            kwargs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        return cls(**kwargs)
