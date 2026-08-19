"""Structured data contracts shared across the arbitration pipeline.

Every LLM call in this system (critics and adjudicator) is forced through one
of these Pydantic models via the `instructor` library, so nothing downstream
ever has to parse free-text model output.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Dimension = Literal["accuracy", "logic", "completeness"]
Severity = Literal[1, 2, 3, 4, 5]
CriticName = Literal["accuracy_critic", "logic_critic", "completeness_critic"]


class Issue(BaseModel):
    """A single problem a critic found in the evaluated output."""

    quote: str = Field(..., description="Exact substring from the original output that the issue refers to.")
    problem: str = Field(..., description="What is wrong with the quoted text, stated plainly.")
    severity: int = Field(..., ge=1, le=5, description="1 = trivial nitpick, 5 = critical/blocking error.")


class CritiqueReport(BaseModel):
    """The structured verdict a single critic produces for a single dimension."""

    dimension: Dimension
    score: int = Field(..., ge=1, le=5, description="Overall quality score for this dimension only.")
    issues: list[Issue] = Field(default_factory=list)
    validated_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims/spans the critic explicitly checked and confirms are correct/sound.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Critic's confidence in its own assessment.")
    summary: str = Field(..., description="One or two sentence summary of the dimension-specific assessment.")


class CriticRunResult(BaseModel):
    """Wraps a CritiqueReport with run metadata (which model/provider produced it, or how it failed)."""

    critic: CriticName
    provider: str
    model: str
    report: CritiqueReport | None = None
    error: str | None = None
    degraded: bool = False  # True if this critic failed and the verdict proceeded without it

    @property
    def ok(self) -> bool:
        return self.report is not None


class Disagreement(BaseModel):
    """A detected conflict between two or more critics, surfaced to the adjudicator."""

    type: Literal["issue_presence", "severity_gap", "unique_finding", "score_gap"]
    description: str
    critics_involved: list[CriticName]
    details: dict = Field(default_factory=dict)


class ConfirmedIssue(BaseModel):
    quote: str
    problem: str
    severity: int = Field(..., ge=1, le=5)
    evidence: str = Field(..., description="Why the adjudicator believes this issue is real.")
    source_critics: list[CriticName]


class DismissedFlag(BaseModel):
    quote: str
    problem: str
    original_critic: CriticName
    reasoning: str = Field(..., description="Why the adjudicator overruled this flag.")


class Verdict(BaseModel):
    """Final structured output of the adjudication step."""

    overall_score: int = Field(..., ge=1, le=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    confirmed_issues: list[ConfirmedIssue] = Field(default_factory=list)
    dismissed_flags: list[DismissedFlag] = Field(default_factory=list)
    summary: str = Field(..., description="One paragraph overall assessment.")
    short_circuited: bool = Field(
        default=False, description="True if all critics agreed the output was clean and the adjudicator was skipped."
    )


class ArbitrationRecord(BaseModel):
    """The full audit-trail record persisted for one arbitration run."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_prompt: str | None = None
    original_output: str
    critic_runs: list[CriticRunResult] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    verdict: Verdict

    def critic_report(self, name: CriticName) -> CritiqueReport | None:
        for run in self.critic_runs:
            if run.critic == name and run.report is not None:
                return run.report
        return None
