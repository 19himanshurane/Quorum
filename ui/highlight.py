"""Renders the evaluated output as HTML with inline colored annotations.

Red   = confirmed issue (adjudicator agrees it's real)
Yellow = low-confidence / dismissed flag (a critic raised it, adjudicator overruled it)
Green  = explicitly validated claim (a critic checked it and confirmed it's correct)
"""
from __future__ import annotations

import html as html_lib

from arbitration.models import ConfirmedIssue, CriticRunResult, DismissedFlag


# (fill, text, border) - light tinted fills with dark, saturated text/borders,
# tuned to sit on the app's light theme (see .streamlit/config.toml).
_COLORS = {
    "red": ("#FEE2E2", "#B91C1C", "rgba(220, 38, 38, 0.45)"),
    "yellow": ("#FEF3C7", "#92400E", "rgba(180, 83, 9, 0.45)"),
    "green": ("#D1FAE5", "#047857", "rgba(5, 150, 105, 0.45)"),
}
_PRIORITY = {"red": 0, "yellow": 1, "green": 2}


def _find(haystack_lower: str, needle: str) -> tuple[int, int] | None:
    needle = needle.strip()
    if not needle:
        return None
    idx = haystack_lower.find(needle.lower())
    if idx < 0:
        return None
    return idx, idx + len(needle)


def build_annotated_html(
    output: str,
    confirmed_issues: list[ConfirmedIssue],
    dismissed_flags: list[DismissedFlag],
    critic_runs: list[CriticRunResult],
) -> tuple[str, list[str]]:
    """Returns (annotated_html, unmatched_notes). Unmatched notes are flags/claims
    whose quote couldn't be located verbatim in the output (e.g. completeness
    issues, which quote the *prompt*, not the output)."""
    output_lower = output.lower()
    spans: list[tuple[int, int, str, str]] = []
    unmatched: list[str] = []

    for c in confirmed_issues:
        found = _find(output_lower, c.quote)
        tooltip = f"CONFIRMED (severity {c.severity}/5): {c.problem}"
        if found:
            spans.append((*found, "red", tooltip))
        else:
            unmatched.append(f"🔴 {tooltip} — \"{c.quote}\"")

    for d in dismissed_flags:
        found = _find(output_lower, d.quote)
        tooltip = f"DISMISSED: {d.problem} (overruled: {d.reasoning})"
        if found:
            spans.append((*found, "yellow", tooltip))
        else:
            unmatched.append(f"🟡 {tooltip} — \"{d.quote}\"")

    for run in critic_runs:
        if not run.ok or run.report is None or run.critic == "completeness_critic":
            continue  # completeness validated_claims quote the prompt, not the output
        for claim in run.report.validated_claims:
            found = _find(output_lower, claim)
            if found:
                spans.append((*found, "green", f"Validated by {run.critic}"))

    # Resolve overlaps: keep the higher-priority (red > yellow > green) span.
    spans.sort(key=lambda s: (s[0], _PRIORITY[s[2]]))
    kept: list[tuple[int, int, str, str]] = []
    for span in spans:
        if any(span[0] < k[1] and k[0] < span[1] for k in kept):
            continue
        kept.append(span)
    kept.sort(key=lambda s: s[0])

    parts: list[str] = []
    cursor = 0
    for start, end, color, tooltip in kept:
        if start < cursor:
            continue
        bg, fg, border = _COLORS[color]
        parts.append(html_lib.escape(output[cursor:start]))
        parts.append(
            f'<mark title="{html_lib.escape(tooltip)}" '
            f'style="background:{bg};color:{fg};border:1px solid {border};'
            f'border-radius:3px;padding:0 2px;cursor:help;">'
            f"{html_lib.escape(output[start:end])}</mark>"
        )
        cursor = end
    parts.append(html_lib.escape(output[cursor:]))

    annotated = "".join(parts).replace("\n", "<br>")
    return annotated, unmatched
