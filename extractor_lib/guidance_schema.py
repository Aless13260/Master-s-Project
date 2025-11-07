from __future__ import annotations
from typing import Optional
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field


class Guidance(BaseModel):
    """Pydantic model for a guidance item. Mirrors the previous dataclass fields.

    This model is used as the structured output type for LLM extraction and is
    intentionally compatible with the old dataclass-based shape.
    """
    guid: str = Field(default_factory=lambda: uuid4().hex)
    company: Optional[str] = None
    # NOTE: ticker removed to reduce complexity; can be added later via mapping

    # What guidance is about
    guidance_type: Optional[str] = None  # revenue|earnings|margin|product|strategic
    metric: Optional[str] = None

    # Temporal context
    reporting_period: Optional[str] = None
    reporting_date: Optional[str] = None  # ISO date
    guidance_period: Optional[str] = None
    guidance_period_type: Optional[str] = None  # quarter|year|multi-year|ongoing
    guidance_timeframe: Optional[str] = None  # next_quarter|next_year|long_term

    # Quantitative data
    current_value: Optional[float] = None
    current_unit: Optional[str] = None
    guided_value: Optional[float] = None
    guided_range_low: Optional[float] = None
    guided_range_high: Optional[float] = None
    change_pct: Optional[float] = None

    # Metadata
    confidence: Optional[str] = None  # high|medium|low
    is_quantitative: Optional[bool] = None
    statement_text: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[str] = None  # 8-K|press_release|earnings_call
    extracted_at: Optional[str] = None

    def to_dict(self) -> dict:
        return self.dict()

    @classmethod
    def from_raw(cls, **kwargs) -> "Guidance":
        if "extracted_at" not in kwargs or not kwargs.get("extracted_at"):
            kwargs["extracted_at"] = datetime.utcnow().isoformat() + "Z"
        return cls(**kwargs)
