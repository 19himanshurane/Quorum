from arbitration.disagreement import all_critics_failed, detect_disagreements, is_clean_sweep
from arbitration.models import CriticRunResult, CritiqueReport, Issue


def _run(critic, score, issues=None, validated=None, confidence=0.8, ok=True):
    if not ok:
        return CriticRunResult(critic=critic, provider="mock", model="mock", error="boom", degraded=True)
    return CriticRunResult(
        critic=critic,
        provider="mock",
        model="mock",
        report=CritiqueReport(
            dimension="accuracy",
            score=score,
            issues=issues or [],
            validated_claims=validated or [],
            confidence=confidence,
            summary="s",
        ),
    )


def test_severity_gap_detected_for_overlapping_quotes():
    runs = [
        _run("accuracy_critic", 3, issues=[Issue(quote="the sky is green", problem="p", severity=5)]),
        _run("logic_critic", 4, issues=[Issue(quote="the sky is green", problem="p2", severity=1)]),
        _run("completeness_critic", 5),
    ]
    disagreements = detect_disagreements(runs)
    assert any(d.type == "severity_gap" for d in disagreements)


def test_issue_presence_detected_when_one_critic_validates_what_another_flags():
    runs = [
        _run("accuracy_critic", 2, issues=[Issue(quote="water boils at 50C", problem="wrong", severity=5)]),
        _run("logic_critic", 5, validated=["water boils at 50C"]),
        _run("completeness_critic", 5),
    ]
    disagreements = detect_disagreements(runs)
    assert any(d.type == "issue_presence" for d in disagreements)


def test_unique_finding_detected_for_non_overlapping_issue():
    runs = [
        _run("accuracy_critic", 3, issues=[Issue(quote="completely unrelated span", problem="p", severity=3)]),
        _run("logic_critic", 5),
        _run("completeness_critic", 5),
    ]
    disagreements = detect_disagreements(runs)
    assert any(d.type == "unique_finding" for d in disagreements)


def test_score_gap_detected_for_diverging_scores():
    runs = [_run("accuracy_critic", 5), _run("logic_critic", 1), _run("completeness_critic", 5)]
    disagreements = detect_disagreements(runs)
    assert any(d.type == "score_gap" for d in disagreements)


def test_is_clean_sweep_true_when_all_high_score_no_issues():
    runs = [_run("accuracy_critic", 5), _run("logic_critic", 5), _run("completeness_critic", 4)]
    assert is_clean_sweep(runs, []) is True


def test_is_clean_sweep_false_if_any_run_failed():
    runs = [_run("accuracy_critic", 5), _run("logic_critic", 5, ok=False), _run("completeness_critic", 5)]
    assert is_clean_sweep(runs, []) is False


def test_all_critics_failed():
    runs = [_run("accuracy_critic", 0, ok=False), _run("logic_critic", 0, ok=False)]
    assert all_critics_failed(runs) is True
    runs.append(_run("completeness_critic", 5))
    assert all_critics_failed(runs) is False
