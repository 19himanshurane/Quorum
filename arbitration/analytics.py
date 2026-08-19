"""Meta-analysis over the audit trail: how do the critics actually behave over time?

Answers the four questions called out in the project spec:
  - which critic finds the most issues
  - which critic gets overruled by the adjudicator most often
  - which failure types are most common
  - how often critics agree vs. disagree
"""
from __future__ import annotations

import pandas as pd

from arbitration.storage import list_arbitrations

_ALL_CRITICS = ("accuracy_critic", "logic_critic", "completeness_critic")


def _load_all(db_path: str) -> list:
    records = []
    offset = 0
    while True:
        batch = list_arbitrations(db_path, limit=500, offset=offset)
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
    return records


def critic_issue_counts(db_path: str) -> pd.DataFrame:
    """Total/average issues raised, and average score/confidence, per critic."""
    rows = {c: {"critic": c, "runs": 0, "ok_runs": 0, "issues_raised": 0, "score_sum": 0, "confidence_sum": 0.0} for c in _ALL_CRITICS}

    for record in _load_all(db_path):
        for run in record.critic_runs:
            row = rows.setdefault(run.critic, {"critic": run.critic, "runs": 0, "ok_runs": 0, "issues_raised": 0, "score_sum": 0, "confidence_sum": 0.0})
            row["runs"] += 1
            if run.ok and run.report is not None:
                row["ok_runs"] += 1
                row["issues_raised"] += len(run.report.issues)
                row["score_sum"] += run.report.score
                row["confidence_sum"] += run.report.confidence

    out = []
    for row in rows.values():
        ok = row["ok_runs"] or 1
        out.append(
            {
                "critic": row["critic"],
                "runs": row["runs"],
                "issues_raised": row["issues_raised"],
                "avg_issues_per_run": round(row["issues_raised"] / ok, 2),
                "avg_score": round(row["score_sum"] / ok, 2),
                "avg_confidence": round(row["confidence_sum"] / ok, 2),
            }
        )
    return pd.DataFrame(out).sort_values("issues_raised", ascending=False).reset_index(drop=True)


def critic_overrule_rates(db_path: str) -> pd.DataFrame:
    """For each critic: of the issues it raised, how many did the adjudicator dismiss?"""
    raised = {c: 0 for c in _ALL_CRITICS}
    dismissed = {c: 0 for c in _ALL_CRITICS}

    for record in _load_all(db_path):
        for run in record.critic_runs:
            if run.ok and run.report is not None:
                raised[run.critic] = raised.get(run.critic, 0) + len(run.report.issues)
        for flag in record.verdict.dismissed_flags:
            dismissed[flag.original_critic] = dismissed.get(flag.original_critic, 0) + 1

    out = []
    for critic in _ALL_CRITICS:
        r = raised.get(critic, 0)
        d = dismissed.get(critic, 0)
        out.append(
            {
                "critic": critic,
                "issues_raised": r,
                "issues_dismissed": d,
                "overrule_rate": round(d / r, 3) if r else 0.0,
            }
        )
    return pd.DataFrame(out).sort_values("overrule_rate", ascending=False).reset_index(drop=True)


def failure_counts(db_path: str) -> pd.DataFrame:
    """How often, and how, does each critic fail to return a report?"""
    rows: list[dict] = []
    for record in _load_all(db_path):
        for run in record.critic_runs:
            if not run.ok:
                rows.append({"critic": run.critic, "provider": run.provider, "model": run.model, "error": run.error})
    if not rows:
        return pd.DataFrame(columns=["critic", "provider", "model", "failure_count"])
    df = pd.DataFrame(rows)
    return (
        df.groupby(["critic", "provider", "model"]).size().reset_index(name="failure_count")
        .sort_values("failure_count", ascending=False).reset_index(drop=True)
    )


def disagreement_type_counts(db_path: str) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for record in _load_all(db_path):
        for d in record.disagreements:
            counts[d.type] = counts.get(d.type, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["type", "count"])
    return pd.DataFrame(sorted(counts.items()), columns=["type", "count"]).sort_values("count", ascending=False).reset_index(drop=True)


def agreement_summary(db_path: str) -> dict:
    records = _load_all(db_path)
    total = len(records)
    if total == 0:
        return {"total_arbitrations": 0, "runs_with_disagreement": 0, "runs_clean": 0, "disagreement_rate": 0.0, "short_circuit_rate": 0.0}
    with_disagreement = sum(1 for r in records if r.disagreements)
    short_circuited = sum(1 for r in records if r.verdict.short_circuited)
    return {
        "total_arbitrations": total,
        "runs_with_disagreement": with_disagreement,
        "runs_clean": total - with_disagreement,
        "disagreement_rate": round(with_disagreement / total, 3),
        "short_circuit_rate": round(short_circuited / total, 3),
    }
