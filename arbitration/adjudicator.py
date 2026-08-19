"""The adjudicator agent: weighs critic evidence and resolves disagreements into a verdict."""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from arbitration.config import CriticConfig, Settings
from arbitration.mock import mock_adjudicate
from arbitration.models import CriticRunResult, Disagreement, Verdict

ADJUDICATOR_SYSTEM_PROMPT = """\
You are the Adjudicator in a multi-agent LLM output arbitration system. Three
independent critics (factual accuracy, logical consistency, completeness) have
each evaluated the same output and may disagree with each other. Your job is to
weigh their evidence and produce a single final verdict.

For EACH disagreement listed, reason through it explicitly before deciding:
- If critics disagree about a FACTUAL claim: mentally re-verify the claim against
  what you know and decide who is right.
- If critics disagree about LOGIC: trace the reasoning chain step by step and
  decide whether the conclusion actually follows.
- If critics disagree about COMPLETENESS: re-read the original prompt and decide
  what was actually required, then judge whether it was met.

Then produce:
- `confirmed_issues`: issues you believe are real, with `evidence` explaining why,
  and `source_critics` listing which critic(s) support it.
- `dismissed_flags`: issues a critic raised that you are overruling, with your
  `reasoning` for dismissing them.
- `overall_score` 1-10 reflecting overall output quality after your review.
- `confidence` 0-1 in this verdict.
- `summary`: one paragraph overall assessment.
Do not simply average the critics' scores - actually adjudicate."""


def _build_user_prompt(
    original_prompt: str | None,
    original_output: str,
    critic_runs: list[CriticRunResult],
    disagreements: list[Disagreement],
) -> str:
    lines = [
        f"ORIGINAL PROMPT/QUESTION:\n{original_prompt or '(not provided)'}",
        f"\nOUTPUT BEING EVALUATED:\n{original_output}",
        "\nCRITIC REPORTS:",
    ]
    for run in critic_runs:
        if not run.ok:
            lines.append(f"- {run.critic} ({run.provider}/{run.model}): FAILED - {run.error}")
            continue
        report = run.report
        lines.append(
            f"- {run.critic} ({run.provider}/{run.model}): dimension={report.dimension} "
            f"score={report.score}/5 confidence={report.confidence:.2f}\n"
            f"  summary: {report.summary}"
        )
        for issue in report.issues:
            lines.append(f"  ISSUE (severity {issue.severity}): \"{issue.quote}\" -> {issue.problem}")
        for claim in report.validated_claims:
            lines.append(f"  VALIDATED: \"{claim}\"")

    lines.append("\nDETECTED DISAGREEMENTS:")
    if not disagreements:
        lines.append("- none")
    for d in disagreements:
        lines.append(f"- [{d.type}] {d.description} (critics: {', '.join(d.critics_involved)})")

    return "\n".join(lines)


def run_adjudicator(
    config: CriticConfig,
    settings: Settings,
    original_prompt: str | None,
    original_output: str,
    critic_runs: list[CriticRunResult],
    disagreements: list[Disagreement],
) -> Verdict:
    if config.provider == "mock":
        return mock_adjudicate(critic_runs, disagreements)

    from arbitration.providers import instructor_client_for, structured_completion

    client = instructor_client_for(config, settings)
    user_prompt = _build_user_prompt(original_prompt, original_output, critic_runs, disagreements)

    @retry(stop=stop_after_attempt(settings.max_retries + 1), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _call() -> Verdict:
        return structured_completion(client, config, Verdict, ADJUDICATOR_SYSTEM_PROMPT, user_prompt)

    try:
        return _call()
    except Exception as exc:  # noqa: BLE001 - adjudicator failure still needs *a* verdict
        return mock_adjudicate(critic_runs, disagreements + [
            Disagreement(
                type="score_gap",
                description=f"Live adjudicator call failed ({type(exc).__name__}: {exc}); fell back to rule-based synthesis.",
                critics_involved=[r.critic for r in critic_runs],
            )
        ])
