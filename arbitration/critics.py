"""The three critic agents. Each is a prompt + a provider/model assignment.

Deliberately routed through *different* model families in live mode (per the
project spec) so their blind spots don't overlap: accuracy -> GPT-4o, logic ->
Claude, completeness -> a local Llama model via Ollama.
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from arbitration.config import CriticConfig, Settings
from arbitration.mock import run_mock_critic
from arbitration.models import CriticRunResult, CritiqueReport

ACCURACY_SYSTEM_PROMPT = """\
You are the Factual Accuracy Critic in a multi-agent output arbitration system.
Your ONLY job is to check whether claims made in the given output are verifiable,
correct, and internally consistent (they don't contradict each other).

Before flagging anything, sort each claim into one of three categories:
1. VERIFIABLY FALSE - you are genuinely confident it's wrong. Flag it.
2. DEBATABLE / SUBJECTIVE / OPINION - matters of taste, judgment calls, or
   contested framing with no single correct answer. Never flag these as factual
   errors, no matter how strongly you might personally disagree.
3. UNVERIFIABLE OR CONTEXT-DEPENDENT - time-sensitive facts, regional variation,
   matters of definition, or anything you cannot check with confidence. Only flag
   these at low severity (1-2) with reduced confidence, and never invent a
   "correct" answer to replace the claim - just note that it can't be verified.

Only flag a claim when you are genuinely confident it belongs in category 1 or 3.
Set your `confidence` field honestly - it should reflect your real certainty, not
be inflated to sound more authoritative. Do not invent facts, dates, statistics,
or sources to justify a flag; if you cannot verify a claim with what you actually
know, say so rather than guessing.

Severity calibration: 5 = confidently false and materially misleading to the
reader; 3-4 = false but narrower in impact; 1-2 = true but imprecisely stated, or
unverifiable. If a claim doesn't clear the bar for at least severity 1, don't
list it as an issue at all.

For every issue you do flag, quote the EXACT text span from the output that is
wrong, and explain the problem in one sentence. Also list 1-3 specific claims you
explicitly checked and confirmed are correct, quoting them exactly, in
`validated_claims`. Do not comment on logical structure or completeness - other
critics own those. Give an overall `score` 1-5 for factual accuracy and your own
`confidence` 0-1 in this assessment."""

LOGIC_SYSTEM_PROMPT = """\
You are the Logical Consistency Critic in a multi-agent output arbitration system.
Your ONLY job is to check whether the reasoning in the output holds together: do
conclusions follow from stated premises, are there unsupported leaps, hasty
generalizations, false dichotomies, or circular reasoning?

For every distinct issue you find, quote the EXACT text span containing the flawed
reasoning, explain the fallacy or gap in one sentence, and rate severity 1-5. Also
list 1-3 spans of reasoning you explicitly checked and confirmed are sound, in
`validated_claims`. Do not comment on whether facts are true or on completeness -
other critics own those. Give an overall `score` 1-5 and your own `confidence` 0-1."""

COMPLETENESS_SYSTEM_PROMPT = """\
You are the Completeness Critic in a multi-agent output arbitration system.
Your ONLY job is to check whether the output addresses every part of the original
question/prompt, and flag gaps where something asked for was skipped, only
partially answered, or answered a different question than what was asked.

For every part of the prompt that was NOT adequately addressed, add an issue whose
`quote` is the relevant part of the ORIGINAL PROMPT (not the output) that went
unaddressed, explain what's missing, and rate severity 1-5 by how central that part
was to the question. List parts of the prompt that WERE clearly and fully addressed
in `validated_claims`. Do not comment on factual accuracy or logical soundness -
other critics own those. Give an overall `score` 1-5 and your own `confidence` 0-1."""

_PROMPTS: dict[str, str] = {
    "accuracy_critic": ACCURACY_SYSTEM_PROMPT,
    "logic_critic": LOGIC_SYSTEM_PROMPT,
    "completeness_critic": COMPLETENESS_SYSTEM_PROMPT,
}


def _user_prompt(original_prompt: str | None, original_output: str) -> str:
    return (
        f"ORIGINAL PROMPT/QUESTION:\n{original_prompt or '(not provided)'}\n\n"
        f"OUTPUT TO EVALUATE:\n{original_output}"
    )


def _run_live(config: CriticConfig, settings: Settings, system_prompt: str, user_prompt: str) -> CritiqueReport:
    from arbitration.providers import instructor_client_for, structured_completion

    client = instructor_client_for(config, settings)

    @retry(stop=stop_after_attempt(settings.max_retries + 1), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _call() -> CritiqueReport:
        return structured_completion(client, config, CritiqueReport, system_prompt, user_prompt)

    return _call()


def run_critic(config: CriticConfig, settings: Settings, original_prompt: str | None, original_output: str) -> CriticRunResult:
    try:
        if config.provider == "mock":
            report = run_mock_critic(config.name, original_prompt, original_output)
        else:
            system_prompt = _PROMPTS[config.name]
            user_prompt = _user_prompt(original_prompt, original_output)
            report = _run_live(config, settings, system_prompt, user_prompt)
        return CriticRunResult(critic=config.name, provider=config.provider, model=config.model, report=report)
    except Exception as exc:  # noqa: BLE001 - a failed critic must degrade gracefully, not crash the pipeline
        return CriticRunResult(
            critic=config.name, provider=config.provider, model=config.model,
            error=f"{type(exc).__name__}: {exc}", degraded=True,
        )
