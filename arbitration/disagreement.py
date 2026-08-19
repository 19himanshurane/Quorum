"""Compares critic reports and surfaces the conflicts worth an adjudicator's attention.

Three kinds of disagreement, per the project spec:
  1. issue_presence  - one critic flags a span as a problem; another explicitly validated
                        overlapping text as correct.
  2. severity_gap    - two+ critics flag overlapping spans as issues, but rate severity
                        more than 2 points apart.
  3. unique_finding  - an issue has no overlap with anything another critic said at all
                        (flagged or validated) - it may be a real catch, or noise.

A bonus `score_gap` type also fires when dimension scores diverge sharply, since that
signals critics disagree about overall quality even without a specific overlapping span.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from arbitration.models import CriticRunResult, Disagreement, Issue

QUOTE_OVERLAP_THRESHOLD = 0.35
SEVERITY_GAP_THRESHOLD = 2
SCORE_GAP_THRESHOLD = 3


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _quotes_overlap(a: str, b: str) -> bool:
    a, b = _normalize(a), _normalize(b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= QUOTE_OVERLAP_THRESHOLD


def detect_disagreements(critic_runs: list[CriticRunResult]) -> list[Disagreement]:
    ok_runs = [r for r in critic_runs if r.ok and r.report is not None]
    disagreements: list[Disagreement] = []

    # --- score_gap: dimension scores diverge sharply ---
    if len(ok_runs) >= 2:
        scores = [(r.critic, r.report.score) for r in ok_runs]
        max_critic, max_score = max(scores, key=lambda t: t[1])
        min_critic, min_score = min(scores, key=lambda t: t[1])
        if max_score - min_score >= SCORE_GAP_THRESHOLD:
            disagreements.append(
                Disagreement(
                    type="score_gap",
                    description=(
                        f"{max_critic} scored this a {max_score}/5 while {min_critic} scored it a "
                        f"{min_score}/5 - critics disagree substantially on overall quality."
                    ),
                    critics_involved=[max_critic, min_critic],
                    details={"scores": dict(scores)},
                )
            )

    # --- pairwise issue/validation comparison ---
    for i, run_a in enumerate(ok_runs):
        for run_b in ok_runs[i + 1 :]:
            _compare_pair(run_a, run_b, disagreements)

    # --- unique_finding: issues with no overlap anywhere else ---
    for run in ok_runs:
        others = [r for r in ok_runs if r is not run]
        other_spans: list[str] = []
        for other in others:
            other_spans.extend(issue.quote for issue in other.report.issues)
            other_spans.extend(other.report.validated_claims)
        for issue in run.report.issues:
            if not any(_quotes_overlap(issue.quote, span) for span in other_spans) and others:
                disagreements.append(
                    Disagreement(
                        type="unique_finding",
                        description=(
                            f"{run.critic} flagged \"{issue.quote[:80]}\" (severity {issue.severity}) - "
                            f"no other critic touched on this at all."
                        ),
                        critics_involved=[run.critic],
                        details={"quote": issue.quote, "severity": issue.severity},
                    )
                )

    return disagreements


def _compare_pair(run_a: CriticRunResult, run_b: CriticRunResult, out: list[Disagreement]) -> None:
    for issue_a in run_a.report.issues:
        # issue_presence: A flags it, B explicitly validated overlapping text
        for claim_b in run_b.report.validated_claims:
            if _quotes_overlap(issue_a.quote, claim_b):
                out.append(
                    Disagreement(
                        type="issue_presence",
                        description=(
                            f"{run_a.critic} flagged \"{issue_a.quote[:80]}\" as a problem, but "
                            f"{run_b.critic} explicitly validated overlapping content as correct."
                        ),
                        critics_involved=[run_a.critic, run_b.critic],
                        details={"issue_quote": issue_a.quote, "validated_quote": claim_b},
                    )
                )

        # severity_gap: both flag overlapping spans, severities differ a lot
        for issue_b in run_b.report.issues:
            if _quotes_overlap(issue_a.quote, issue_b.quote):
                gap = abs(issue_a.severity - issue_b.severity)
                if gap > SEVERITY_GAP_THRESHOLD:
                    out.append(
                        Disagreement(
                            type="severity_gap",
                            description=(
                                f"{run_a.critic} rated \"{issue_a.quote[:80]}\" severity {issue_a.severity}, "
                                f"{run_b.critic} rated overlapping content severity {issue_b.severity} "
                                f"(gap of {gap})."
                            ),
                            critics_involved=[run_a.critic, run_b.critic],
                            details={
                                "severity_a": issue_a.severity,
                                "severity_b": issue_b.severity,
                                "quote_a": issue_a.quote,
                                "quote_b": issue_b.quote,
                            },
                        )
                    )


def is_clean_sweep(critic_runs: list[CriticRunResult], disagreements: list[Disagreement]) -> bool:
    """True when every critic ran successfully, found nothing, and scored the output highly."""
    ok_runs = [r for r in critic_runs if r.ok and r.report is not None]
    if len(ok_runs) != len(critic_runs) or not ok_runs:
        return False
    if disagreements:
        return False
    return all(r.report.score >= 4 and not r.report.issues for r in ok_runs)


def all_critics_failed(critic_runs: list[CriticRunResult]) -> bool:
    return bool(critic_runs) and all(not r.ok for r in critic_runs)
