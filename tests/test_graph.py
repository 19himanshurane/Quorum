from unittest.mock import patch

from arbitration.graph import run_arbitration
from arbitration.models import CriticRunResult


def test_clean_output_short_circuits():
    record = run_arbitration(
        original_output=(
            "Rainbows form when sunlight is refracted, reflected, and dispersed inside water "
            "droplets, splitting white light into its component colors."
        ),
        original_prompt="What causes rainbows to form?",
    )
    assert record.verdict.short_circuited is True
    assert record.verdict.confirmed_issues == []


def test_flawed_output_gets_adjudicated_with_confirmed_issues():
    record = run_arbitration(
        original_output="The Eiffel Tower is in London. It was built therefore it is famous.",
        original_prompt="Where is the Eiffel Tower and why is it famous?",
    )
    assert record.verdict.short_circuited is False
    assert len(record.verdict.confirmed_issues) >= 1
    assert len(record.critic_runs) == 3


def test_all_critics_failing_still_produces_a_verdict():
    def always_fail(config, settings, prompt, output):
        return CriticRunResult(critic=config.name, provider=config.provider, model=config.model, error="down", degraded=True)

    with patch("arbitration.graph.run_critic", always_fail):
        record = run_arbitration("anything", "anything")

    assert record.verdict.confidence == 0.0
    assert all(not r.ok for r in record.critic_runs)


def test_partial_failure_degrades_gracefully_with_a_note():
    from arbitration.critics import run_critic as real_run_critic

    def flaky(config, settings, prompt, output):
        if config.name == "logic_critic":
            return CriticRunResult(critic=config.name, provider=config.provider, model=config.model, error="timeout", degraded=True)
        return real_run_critic(config, settings, prompt, output)

    with patch("arbitration.graph.run_critic", flaky):
        record = run_arbitration("The sky is blue due to Rayleigh scattering.", "Why is the sky blue?")

    ok_critics = {r.critic for r in record.critic_runs if r.ok}
    assert "logic_critic" not in ok_critics
    assert len(ok_critics) == 2
    assert "logic_critic did not return a report" in record.verdict.summary
