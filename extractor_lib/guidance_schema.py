from __future__ import annotations
from typing import Optional, Literal
from uuid import uuid4
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


class GuidanceExtraction(BaseModel):
    """Schema for LLM extraction (excludes metadata like company)."""
    
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
    unit: Optional[Literal["USD", "%", "million", "billion", "units"]] = None
    # guided_value removed in favor of using guided_range_low for single values
    guided_range_low: Optional[float] = None
    guided_range_high: Optional[float] = None

    # Revision / Qualitative
    is_revision: Optional[bool] = None
    revision_direction: Optional[Literal["increased", "decreased"]] = None
    qualitative_direction: Optional[str] = None
    rationales: Optional[str] = None
    
    # Extracted text snippet
    statement_text: Optional[str] = None


class Guidance(GuidanceExtraction):
    """Full guidance model including metadata. Mirrors the previous dataclass fields.

    This model is used for storage and analysis.
    
    Note: Document-level metadata (source_url, published_at, ingested_at) are stored
    at the document level in contents.jsonl, not duplicated per guidance item.
    """
    guid: str = Field(default_factory=lambda: uuid4().hex)
    ticker: Optional[str] = None  # Stock ticker symbol (e.g., AAPL, MSFT)

    # Source classification
    source_type: Optional[Literal["8-K", "10-K", "10-Q", "press_release", "earnings_call", "investor_presentation", "other"]] = None
    
    # Extraction metadata
    extracted_at: Optional[str] = None  # When this guidance item was extracted (output created)
    processing_duration_seconds: Optional[float] = None  # Time taken for LLM processing in seconds
    extraction_method: Optional[Literal["standard", "agentic_review"]] = "standard"
    was_updated_by_agent: Optional[bool] = False

    @field_validator("guid", mode="before")
    @classmethod
    def _coerce_guid(cls, value):
        """Ensure guid is always a non-empty string.

        Some LLM runs intermittently return `null` for `guid`, which fails Pydantic
        validation and triggers extractor retries. We treat missing/empty guids as
        a signal to auto-generate one.
        """
        if value is None:
            return uuid4().hex
        if isinstance(value, str) and not value.strip():
            return uuid4().hex
        return value

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_raw(cls, **kwargs) -> "Guidance":
        if "extracted_at" not in kwargs or not kwargs.get("extracted_at"):
            kwargs["extracted_at"] = datetime.now(timezone.utc).isoformat()
        return cls(**kwargs)
