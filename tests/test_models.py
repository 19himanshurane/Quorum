import pytest
from pydantic import ValidationError

from arbitration.models import ArbitrationRecord, CritiqueReport, Issue, Verdict


def test_issue_rejects_out_of_range_severity():
    with pytest.raises(ValidationError):
        Issue(quote="x", problem="y", severity=6)


def test_critique_report_defaults():
    report = CritiqueReport(dimension="accuracy", score=5, confidence=0.9, summary="fine")
    assert report.issues == []
    assert report.validated_claims == []


def test_arbitration_record_critic_report_lookup():
    report = CritiqueReport(dimension="accuracy", score=5, confidence=0.9, summary="fine")
    record = ArbitrationRecord(
        original_output="hello world",
        critic_runs=[],
        verdict=Verdict(overall_score=10, confidence=0.9, summary="ok"),
    )
    assert record.critic_report("accuracy_critic") is None

    from arbitration.models import CriticRunResult

    record.critic_runs.append(CriticRunResult(critic="accuracy_critic", provider="mock", model="mock", report=report))
    assert record.critic_report("accuracy_critic") is report
    assert record.critic_report("logic_critic") is None
