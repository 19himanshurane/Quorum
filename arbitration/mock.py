"""Deterministic stand-in critics + adjudicator used when ARBITRATION_PROVIDER_MODE=mock.

These are *not* trying to be a good fact-checker or logic-checker - they're a
lightweight, offline, zero-cost stand-in that exercises the full pipeline
(parallel dispatch, disagreement detection, adjudication, storage, API, UI)
deterministically, with no API keys required. Set ARBITRATION_PROVIDER_MODE=live
and real provider keys/Ollama to route through actual OpenAI/Anthropic/Ollama
models using the same interfaces.
"""
from __future__ import annotations

import re

from arbitration.models import (
    ConfirmedIssue,
    CritiqueReport,
    CriticRunResult,
    Disagreement,
    DismissedFlag,
    Issue,
    Verdict,
)

# ---------------------------------------------------------------------------
# Accuracy critic: small table of well-known misconceptions to catch planted
# factual errors, plus a generic "unsupported absolute claim" heuristic.
# ---------------------------------------------------------------------------
_KNOWN_FALSE_CLAIMS = [
    (re.compile(r"great wall of china.{0,40}visible from space", re.I),
     "The Great Wall of China is not visible from space with the naked eye - this is a persistent myth."),
    (re.compile(r"humans? (?:only )?use[s]? (?:only )?10\s*%\s*of (?:their|the|our) brain", re.I),
     "The '10% of the brain' claim is a myth; brain imaging shows we use virtually all of it."),
    (re.compile(r"napoleon.{0,30}(?:was\s+)?short", re.I),
     "Napoleon was of average height for his era (~5'7\"); the 'short' claim stems from a unit mix-up."),
    (re.compile(r"eiffel tower.{0,20}(?:is|located)\s+in\s+london", re.I),
     "The Eiffel Tower is located in Paris, France, not London."),
    (re.compile(r"einstein.{0,20}failed\s+math", re.I),
     "Einstein did not fail math; he excelled at it from a young age."),
    (re.compile(r"goldfish.{0,30}memory.{0,20}(?:3|three)\s*seconds?", re.I),
     "The 'goldfish have a 3-second memory' claim is a myth; studies show it's closer to months."),
    (re.compile(r"lightning\s+never\s+strikes\s+the\s+same\s+place\s+twice", re.I),
     "Lightning frequently strikes the same location repeatedly (e.g. tall structures)."),
    (re.compile(r"sun\s+(?:revolves|orbits)\s+around\s+(?:the\s+)?earth", re.I),
     "The Earth orbits the Sun, not the other way around."),
    (re.compile(r"great\s+depression.{0,20}(?:started|began).{0,20}194\d", re.I),
     "The Great Depression began in 1929, not the 1940s."),
    (re.compile(r"python\s+(?:is|was)\s+(?:a\s+)?compiled\s+language", re.I),
     "Python's reference implementation is interpreted/bytecode-compiled at runtime, not ahead-of-time compiled like C/C++."),
]

_ABSOLUTE_CLAIM = re.compile(r"\b(always|never|every single|no one ever|proven fact that|impossible for)\b", re.I)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def mock_accuracy_critique(prompt: str | None, output: str) -> CritiqueReport:
    sentences = _split_sentences(output)
    issues: list[Issue] = []
    validated: list[str] = []

    for sentence in sentences:
        matched = False
        for pattern, explanation in _KNOWN_FALSE_CLAIMS:
            if pattern.search(sentence):
                issues.append(Issue(quote=sentence, problem=explanation, severity=5))
                matched = True
                break
        if matched:
            continue
        if _ABSOLUTE_CLAIM.search(sentence):
            issues.append(
                Issue(
                    quote=sentence,
                    problem="Unqualified absolute claim ('always'/'never'/etc.) stated without supporting evidence.",
                    severity=2,
                )
            )
        elif re.search(r"\d", sentence) and len(validated) < 3:
            validated.append(sentence)

    if not issues:
        score = 5
        summary = "No factual errors detected; specific/verifiable claims checked out."
    else:
        max_sev = max(i.severity for i in issues)
        score = max(1, 5 - max_sev + 1) if max_sev >= 4 else 4
        summary = (
            f"Found {len(issues)} factual concern(s), including "
            f"{sum(1 for i in issues if i.severity >= 4)} high-severity error(s)."
        )

    confidence = 0.9 if issues or sentences else 0.6
    return CritiqueReport(
        dimension="accuracy", score=score, issues=issues, validated_claims=validated,
        confidence=confidence, summary=summary,
    )


# ---------------------------------------------------------------------------
# Shared keyword-overlap helper (used by the logic and completeness critics).
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to", "in", "on",
    "for", "what", "how", "why", "does", "do", "did", "can", "could", "would", "should",
    "please", "explain", "describe", "with", "that", "this", "it", "be", "you", "your",
    "also", "additionally", "as", "well", "i", "we", "me", "us", "my", "our",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


# ---------------------------------------------------------------------------
# Logic critic: pattern-based fallacy detection.
# ---------------------------------------------------------------------------
_HASTY_GENERALIZATION = re.compile(r"\b(all|every|everyone|no one|nobody)\b.{0,80}\b(always|never)\b", re.I)
_FALSE_DICHOTOMY = re.compile(r"\beither\b.{0,60}\bor\b.{0,60}\b(no other|nothing else|there('?s| is) no)\b", re.I)
_UNSUPPORTED_THEREFORE = re.compile(r"\btherefore\b|\bthus\b|\bso clearly\b|\bwhich proves\b", re.I)
_CIRCULAR_MARKERS = re.compile(r"\bbecause it (?:is|just is)\b|\bthat'?s just how it (?:is|works)\b", re.I)


def mock_logic_critique(prompt: str | None, output: str) -> CritiqueReport:
    sentences = _split_sentences(output)
    issues: list[Issue] = []
    validated: list[str] = []

    for idx, sentence in enumerate(sentences):
        if _CIRCULAR_MARKERS.search(sentence):
            issues.append(Issue(quote=sentence, problem="Circular reasoning: the claim is used to justify itself.", severity=4))
            continue
        if _FALSE_DICHOTOMY.search(sentence):
            issues.append(Issue(quote=sentence, problem="False dichotomy: presents only two options when more may exist.", severity=3))
            continue
        if _HASTY_GENERALIZATION.search(sentence):
            issues.append(
                Issue(
                    quote=sentence,
                    problem="Hasty generalization: broad claim ('all'/'every') asserted without sufficient support.",
                    severity=3,
                )
            )
            continue
        if _UNSUPPORTED_THEREFORE.search(sentence):
            # A conclusion word is only earning its keep if the immediately preceding
            # sentence actually shares topical ground with it - otherwise it's a
            # non sequitur wearing the language of a valid inference.
            prev_keywords = _keywords(sentences[idx - 1]) if idx > 0 else set()
            cur_keywords = _keywords(sentence)
            if idx == 0 or not (prev_keywords & cur_keywords):
                issues.append(
                    Issue(
                        quote=sentence,
                        problem=(
                            "Conclusion word ('therefore'/'thus') used, but the immediately preceding "
                            "sentence shares no topical connection to support it (non sequitur)."
                        ),
                        severity=4,
                    )
                )
            elif len(validated) < 3:
                validated.append(sentence)
        elif len(validated) < 3 and idx == 0:
            validated.append(sentence)

    if not issues:
        score = 5
        summary = "Reasoning chain holds together; conclusions follow from stated premises."
    else:
        max_sev = max(i.severity for i in issues)
        score = max(1, 5 - max_sev + 1) if max_sev >= 4 else 3
        summary = f"Found {len(issues)} reasoning issue(s) (e.g. unsupported leaps or generalizations)."

    return CritiqueReport(
        dimension="logic", score=score, issues=issues, validated_claims=validated,
        confidence=0.8, summary=summary,
    )


# ---------------------------------------------------------------------------
# Completeness critic: does the output touch every sub-part of the prompt?
# ---------------------------------------------------------------------------


def _split_subquestions(prompt: str) -> list[str]:
    parts = re.split(r"\?|;|\band also\b|\bas well as\b", prompt, flags=re.I)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def mock_completeness_critique(prompt: str | None, output: str) -> CritiqueReport:
    if not prompt:
        return CritiqueReport(
            dimension="completeness", score=4, issues=[],
            validated_claims=["No original prompt supplied - completeness checked against the output's own internal scope only."],
            confidence=0.4,
            summary="No original prompt was provided, so sub-question coverage could not be verified against intent.",
        )

    subquestions = _split_subquestions(prompt)
    output_keywords = _keywords(output)
    issues: list[Issue] = []
    validated: list[str] = []

    for sq in subquestions:
        sq_keywords = _keywords(sq)
        if not sq_keywords:
            continue
        overlap = sq_keywords & output_keywords
        coverage = len(overlap) / len(sq_keywords)
        if coverage < 0.3:
            issues.append(
                Issue(quote=sq, problem="The response does not appear to address this part of the question.", severity=4)
            )
        elif coverage >= 0.6:
            validated.append(sq)

    if not issues:
        score = 5
        summary = "The response addresses every part of the original question."
    else:
        score = max(1, 5 - len(issues))
        summary = f"The response leaves {len(issues)} part(s) of the question unaddressed or only partially addressed."

    return CritiqueReport(
        dimension="completeness", score=score, issues=issues, validated_claims=validated,
        confidence=0.75, summary=summary,
    )


_MOCK_CRITIQUE_FNS = {
    "accuracy_critic": mock_accuracy_critique,
    "logic_critic": mock_logic_critique,
    "completeness_critic": mock_completeness_critique,
}


def run_mock_critic(name: str, prompt: str | None, output: str) -> CritiqueReport:
    return _MOCK_CRITIQUE_FNS[name](prompt, output)


# ---------------------------------------------------------------------------
# Mock adjudicator: rule-based synthesis over the critic reports.
# ---------------------------------------------------------------------------

def mock_adjudicate(critic_runs: list[CriticRunResult], disagreements: list[Disagreement]) -> Verdict:
    ok_runs = [r for r in critic_runs if r.ok and r.report is not None]

    if not ok_runs:
        return Verdict(
            overall_score=1,
            confidence=0.0,
            confirmed_issues=[],
            dismissed_flags=[],
            summary="All critics failed to return a report; no verdict could be substantively formed.",
        )

    confirmed: list[ConfirmedIssue] = []
    dismissed: list[DismissedFlag] = []

    unique_finding_quotes = {d.details.get("quote") for d in disagreements if d.type == "unique_finding"}

    for run in ok_runs:
        for issue in run.report.issues:
            is_lone_low_confidence = issue.quote in unique_finding_quotes and run.report.confidence < 0.6
            if issue.severity <= 1 or is_lone_low_confidence:
                dismissed.append(
                    DismissedFlag(
                        quote=issue.quote,
                        problem=issue.problem,
                        original_critic=run.critic,
                        reasoning=(
                            "Severity too trivial to confirm without corroboration."
                            if issue.severity <= 1
                            else "Sole critic finding with below-threshold confidence and no corroboration from other critics."
                        ),
                    )
                )
            else:
                confirmed.append(
                    ConfirmedIssue(
                        quote=issue.quote,
                        problem=issue.problem,
                        severity=issue.severity,
                        evidence=f"Raised by {run.critic} (confidence {run.report.confidence:.2f}).",
                        source_critics=[run.critic],
                    )
                )

    avg_score_5 = sum(r.report.score for r in ok_runs) / len(ok_runs)
    overall_score = round((avg_score_5 / 5) * 10)
    penalty = sum(1 for c in confirmed if c.severity >= 4)
    overall_score = max(1, overall_score - penalty)

    degraded_count = len(critic_runs) - len(ok_runs)
    avg_confidence = sum(r.report.confidence for r in ok_runs) / len(ok_runs)
    confidence = min(1.0, max(0.05, avg_confidence - 0.15 * degraded_count - 0.05 * len(disagreements)))

    degraded_note = (
        f" {degraded_count} critic(s) failed to report and were excluded from this verdict."
        if degraded_count
        else ""
    )
    summary = (
        f"Across {len(ok_runs)} critic report(s), {len(confirmed)} issue(s) were confirmed "
        f"({penalty} high-severity) and {len(dismissed)} flagged concern(s) were dismissed as "
        f"low-confidence or trivial. {len(disagreements)} disagreement(s) were detected between critics."
        f"{degraded_note}"
    )

    return Verdict(
        overall_score=overall_score,
        confidence=round(confidence, 2),
        confirmed_issues=confirmed,
        dismissed_flags=dismissed,
        summary=summary,
    )
