"""Clinical fact model: every documented statement traces to transcript evidence."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FactCategory(StrEnum):
    COMPLAINT = "complaint"
    DURATION = "duration"
    ONSET = "onset"
    PROGRESSION = "progression"
    TEMPERATURE = "temperature"
    TREATMENT = "treatment"
    TREATMENT_EFFECT = "treatment_effect"
    INVESTIGATION = "investigation"
    PREVIOUS_VISIT = "previous_visit"
    OTHER_HISTORY = "other_history"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Temporality(StrEnum):
    CURRENT = "current"
    PAST = "past"
    UNKNOWN = "unknown"


class Certainty(StrEnum):
    EXPLICIT = "explicit"
    AMBIGUOUS = "ambiguous"


class ClinicalFact(BaseModel):
    id: str
    category: FactCategory
    value: str = Field(description="Short normalised medical phrase in Russian")
    statement: str = Field("", description="One Russian sentence for the history section")
    polarity: Polarity = Polarity.POSITIVE
    temporality: Temporality = Temporality.UNKNOWN
    certainty: Certainty = Certainty.EXPLICIT
    evidence_segment_ids: list[str] = Field(default_factory=list)
    evidence_quote: str = ""
    is_correction: bool = Field(False, description="Patient explicitly corrected an earlier answer")
    relevant_to_current: bool = Field(
        True, description="For other_history: related to this illness"
    )
    ambiguity_note: str = ""
    llm_confidence: float | None = Field(None, description="Auxiliary only; never proof")


class ExtractionResult(BaseModel):
    facts: list[ClinicalFact] = Field(default_factory=list)


class FormattedDocument(BaseModel):
    complaints_text: str = ""
    history_text: str = ""


class ReviewWarning(BaseModel):
    code: str
    message: str
    segment_ids: list[str] = Field(default_factory=list)


class RejectedFact(BaseModel):
    fact: ClinicalFact
    reason: str


class SentenceTrace(BaseModel):
    section: str
    sentence: str
    fact_ids: list[str]


class DocumentDraft(BaseModel):
    complaints_text: str = ""
    history_text: str = ""
    warnings: list[ReviewWarning] = Field(default_factory=list)
    accepted_facts: list[ClinicalFact] = Field(default_factory=list)
    rejected_facts: list[RejectedFact] = Field(default_factory=list)
    trace: list[SentenceTrace] = Field(default_factory=list)
    formatter: str = "deterministic"
    extraction_prompt_version: str = ""
    formatter_prompt_version: str = ""
