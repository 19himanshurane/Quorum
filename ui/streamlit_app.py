"""Verdict Explorer - Streamlit UI for Quorum.

Run with:  streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from arbitration import analytics
from arbitration.config import load_settings
from arbitration.graph import run_arbitration
from arbitration.models import ArbitrationRecord
from arbitration.storage import count_arbitrations, list_arbitrations, save_arbitration
from ui.highlight import build_annotated_html

st.set_page_config(page_title="Quorum", layout="wide", page_icon="⚖️")

settings = load_settings()

DIMENSION_LABEL = {
    "accuracy_critic": "Factual Accuracy",
    "logic_critic": "Logical Consistency",
    "completeness_critic": "Completeness",
}
DISAGREEMENT_LABEL = {
    "issue_presence": "Disagree whether it's an issue",
    "severity_gap": "Severity disagreement",
    "unique_finding": "Unique finding",
    "score_gap": "Score disagreement",
}

# --------------------------------------------------------------------------- #
# Design tokens (kept in sync with .streamlit/config.toml) + global CSS
# --------------------------------------------------------------------------- #
SURFACE = "#F1F3F9"
SURFACE_BORDER = "#DDE1EC"
TEXT_MUTED = "#64748B"
ACCENT = "#6366F1"
DANGER, DANGER_BG = "#DC2626", "#FEE2E2"
WARNING, WARNING_BG = "#B45309", "#FEF3C7"
SUCCESS, SUCCESS_BG = "#059669", "#D1FAE5"

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1180px; }}

    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {SURFACE_BORDER};
        border-radius: 10px;
        padding: 0.9rem 1rem 0.7rem;
    }}
    button[data-baseweb="tab"] {{ font-weight: 600; }}
    [data-testid="stExpander"] {{
        border: 1px solid {SURFACE_BORDER};
        border-radius: 10px;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {SURFACE_BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    .qm-card {{
        background: {SURFACE};
        border: 1px solid {SURFACE_BORDER};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
    }}
    .qm-card.qm-flagged {{ border-color: {WARNING}; }}
    .qm-card.qm-agree {{ border-color: {SUCCESS}; }}
    .qm-card h4 {{ margin: 0 0 0.2rem 0; font-size: 1.02rem; }}
    .qm-muted {{ color: {TEXT_MUTED}; font-size: 0.85rem; }}
    .qm-output-box {{
        border: 1px solid {SURFACE_BORDER};
        background: {SURFACE};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        line-height: 1.65;
    }}
    .qm-badge {{
        display: inline-block;
        border-radius: 999px;
        padding: 0.05rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 0.4rem;
    }}
    .qm-badge.qm-danger {{ background: {DANGER_BG}; color: {DANGER}; }}
    .qm-badge.qm-warning {{ background: {WARNING_BG}; color: {WARNING}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Shared render functions
# --------------------------------------------------------------------------- #
def render_verdict(record: ArbitrationRecord) -> None:
    v = record.verdict
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall score", f"{v.overall_score}/10")
    c2.metric("Confidence", f"{v.confidence:.0%}")
    c3.metric("Confirmed issues", len(v.confirmed_issues))
    c4.metric("Dismissed flags", len(v.dismissed_flags))

    if v.short_circuited:
        st.success("✅ Clean bill of health — all critics agreed, adjudication was short-circuited.")

    st.markdown(
        f'<p><b>Annotated output</b> &nbsp; '
        f'<span class="qm-badge qm-danger">🔴 confirmed issue</span>'
        f'<span class="qm-badge qm-warning">🟡 dismissed / low-confidence</span>'
        f'<span style="color:{SUCCESS};font-size:0.85rem;margin-left:0.4rem;">🟢 validated claim</span></p>',
        unsafe_allow_html=True,
    )
    annotated_html, unmatched = build_annotated_html(
        record.original_output, v.confirmed_issues, v.dismissed_flags, record.critic_runs
    )
    st.markdown(f'<div class="qm-output-box">{annotated_html}</div>', unsafe_allow_html=True)
    if unmatched:
        with st.expander(f"{len(unmatched)} additional flag(s) not locatable inline (e.g. completeness gaps reference the prompt)"):
            for note in unmatched:
                st.markdown(f"- {note}")

    st.markdown(f"**Assessment:** {v.summary}")

    col_confirmed, col_dismissed = st.columns(2)
    with col_confirmed:
        st.markdown(f'##### 🔴 Confirmed issues <span class="qm-muted">({len(v.confirmed_issues)})</span>', unsafe_allow_html=True)
        if not v.confirmed_issues:
            st.caption("None.")
        for issue in v.confirmed_issues:
            with st.expander(f"Severity {issue.severity}/5 — {issue.quote[:60]}"):
                st.write(f"**Problem:** {issue.problem}")
                st.write(f"**Evidence:** {issue.evidence}")
                st.write(f"**Source critic(s):** {', '.join(issue.source_critics)}")
    with col_dismissed:
        st.markdown(f'##### 🟡 Dismissed flags <span class="qm-muted">({len(v.dismissed_flags)})</span>', unsafe_allow_html=True)
        if not v.dismissed_flags:
            st.caption("None.")
        for flag in v.dismissed_flags:
            with st.expander(f"{flag.original_critic} — {flag.quote[:60]}"):
                st.write(f"**Problem raised:** {flag.problem}")
                st.write(f"**Why dismissed:** {flag.reasoning}")


def render_critic_panel(record: ArbitrationRecord) -> None:
    st.markdown("##### Critic comparison")
    disagreement_critics: set[str] = set()
    for d in record.disagreements:
        disagreement_critics.update(d.critics_involved)

    cols = st.columns(3)
    for col, run in zip(cols, record.critic_runs):
        with col:
            label = DIMENSION_LABEL.get(run.critic, run.critic)
            flagged = run.critic in disagreement_critics
            status = (
                f'<span class="qm-badge qm-warning">disagreement</span>'
                if flagged
                else f'<span style="color:{SUCCESS};font-size:0.78rem;">● agrees</span>'
            )
            card_class = "qm-flagged" if flagged else "qm-agree"

            if run.ok and run.report is not None:
                body = (
                    f'<div>Score: <b>{run.report.score}/5</b> &nbsp;|&nbsp; '
                    f'Confidence: <b>{run.report.confidence:.0%}</b></div>'
                    f'<div>Issues found: <b>{len(run.report.issues)}</b> &nbsp;|&nbsp; '
                    f'Validated: <b>{len(run.report.validated_claims)}</b></div>'
                    f'<p class="qm-muted" style="margin-top:0.5rem;">{run.report.summary}</p>'
                )
            else:
                body = f'<p style="color:{DANGER};">Failed: {run.error}</p>'

            st.markdown(
                f'<div class="qm-card {card_class}">'
                f'<h4>{label} {status}</h4>'
                f'<div class="qm-muted">{run.provider} / {run.model}</div>'
                f'<div style="margin-top:0.6rem;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if record.disagreements:
        st.markdown("###### Disagreements between critics")
        for d in record.disagreements:
            st.markdown(
                f'<div class="qm-card qm-flagged">'
                f'<b>{DISAGREEMENT_LABEL.get(d.type, d.type)}</b> '
                f'<span class="qm-muted">({", ".join(d.critics_involved)})</span>'
                f'<div style="margin-top:0.25rem;">{d.description}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No disagreements detected between critics.")


def run_and_store(output: str, prompt: str | None) -> ArbitrationRecord:
    record = run_arbitration(output, prompt)
    save_arbitration(settings.db_path, record)
    return record


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("⚖️ Quorum")
st.caption("Three independent critics evaluate every LLM output. An adjudicator resolves what they disagree on.")
st.caption(
    f"Provider mode: **{settings.provider_mode}** · "
    f"accuracy → {settings.accuracy_critic.provider}/{settings.accuracy_critic.model} · "
    f"logic → {settings.logic_critic.provider}/{settings.logic_critic.model} · "
    f"completeness → {settings.completeness_critic.provider}/{settings.completeness_critic.model}"
)
if settings.provider_mode == "mock":
    st.info(
        "Running in **mock** mode — critics are deterministic offline stand-ins, no API keys required. "
        "Set `ARBITRATION_PROVIDER_MODE=live` with real provider keys / a local Ollama for real model critiques.",
        icon="🧪",
    )

tab_arbitrate, tab_batch, tab_history, tab_analytics = st.tabs(["Arbitrate", "Batch", "History", "Analytics"])

with tab_arbitrate:
    # A form batches every field's latest value with the submit click into one
    # atomic rerun - without it, clicking the button right after typing can race
    # against Streamlit's text_area commit-on-blur and silently need a second click.
    with st.form("arbitrate_form"):
        prompt_input = st.text_area("Original prompt (optional)", height=80, key="single_prompt")
        output_input = st.text_area("LLM output to evaluate", height=180, key="single_output")
        submitted = st.form_submit_button("Run arbitration", type="primary")

    if submitted:
        if not output_input.strip():
            st.warning("Please provide an output to evaluate.")
        else:
            with st.spinner("Dispatching critics in parallel and adjudicating..."):
                st.session_state["last_record"] = run_and_store(output_input, prompt_input or None)

    if "last_record" in st.session_state:
        st.divider()
        render_verdict(st.session_state["last_record"])
        st.divider()
        render_critic_panel(st.session_state["last_record"])

with tab_batch:
    st.write(
        "Submit multiple outputs at once: upload a CSV with an `output` column (and optional `prompt` column), "
        "or paste entries below separated by a line containing only `---`."
    )
    with st.form("batch_form"):
        uploaded = st.file_uploader("CSV file", type=["csv"])
        pasted = st.text_area("...or paste outputs, separated by a line with just `---`", height=150)
        submitted_batch = st.form_submit_button("Run batch arbitration")

    if submitted_batch:
        items: list[tuple[str, str | None]] = []
        if uploaded is not None:
            df_in = pd.read_csv(uploaded)
            for _, row in df_in.iterrows():
                out = str(row.get("output", "")).strip()
                prm = row.get("prompt")
                prm = str(prm).strip() if isinstance(prm, str) and prm.strip() else None
                if out:
                    items.append((out, prm))
        elif pasted.strip():
            blocks = [b.strip() for b in pasted.split("\n---\n") if b.strip()]
            items = [(b, None) for b in blocks]

        if not items:
            st.warning("No inputs provided.")
        else:
            progress = st.progress(0.0)
            results = []
            for i, (out, prm) in enumerate(items):
                results.append(run_and_store(out, prm))
                progress.progress((i + 1) / len(items))
            st.session_state["batch_results"] = results

    if "batch_results" in st.session_state:
        results: list[ArbitrationRecord] = st.session_state["batch_results"]
        rows = [
            {
                "id": r.id,
                "output_excerpt": (r.original_output[:80] + "…") if len(r.original_output) > 80 else r.original_output,
                "overall_score": r.verdict.overall_score,
                "confirmed_issues": len(r.verdict.confirmed_issues),
                "confidence": r.verdict.confidence,
                "clean": r.verdict.short_circuited,
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        selected_id = st.selectbox("View full verdict for:", [r.id for r in results])
        if selected_id:
            selected = next(r for r in results if r.id == selected_id)
            render_verdict(selected)
            render_critic_panel(selected)

with tab_history:
    total = count_arbitrations(settings.db_path)
    st.write(f"{total} arbitration(s) recorded.")
    if total:
        records = list_arbitrations(settings.db_path, limit=200)
        rows = [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(timespec="seconds"),
                "output_excerpt": (r.original_output[:70] + "…") if len(r.original_output) > 70 else r.original_output,
                "overall_score": r.verdict.overall_score,
                "confidence": r.verdict.confidence,
                "confirmed_issues": len(r.verdict.confirmed_issues),
                "disagreements": len(r.disagreements),
            }
            for r in records
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        chosen = st.selectbox("Inspect arbitration:", [r.id for r in records])
        if chosen:
            record = next(r for r in records if r.id == chosen)
            render_verdict(record)
            render_critic_panel(record)

with tab_analytics:
    st.write("Meta-analysis of critic behavior across every arbitration run so far.")
    summary = analytics.agreement_summary(settings.db_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total arbitrations", summary["total_arbitrations"])
    c2.metric("Disagreement rate", f"{summary['disagreement_rate']:.0%}")
    c3.metric("Short-circuit rate", f"{summary['short_circuit_rate']:.0%}")
    c4.metric("Clean runs", summary["runs_clean"])

    st.markdown("##### Issues raised per critic")
    issue_counts = analytics.critic_issue_counts(settings.db_path)
    if not issue_counts.empty:
        st.bar_chart(issue_counts.set_index("critic")["issues_raised"], color=ACCENT)
        st.dataframe(issue_counts, use_container_width=True, hide_index=True)

    st.markdown("##### Overrule rate per critic (issues raised vs. dismissed by the adjudicator)")
    overrule = analytics.critic_overrule_rates(settings.db_path)
    if not overrule.empty:
        st.bar_chart(overrule.set_index("critic")["overrule_rate"], color=WARNING)
        st.dataframe(overrule, use_container_width=True, hide_index=True)

    st.markdown("##### Disagreement types")
    dtype_counts = analytics.disagreement_type_counts(settings.db_path)
    if not dtype_counts.empty:
        st.bar_chart(dtype_counts.set_index("type")["count"], color=ACCENT)
    else:
        st.caption("No disagreements recorded yet.")

    st.markdown("##### Critic failure rates")
    failures = analytics.failure_counts(settings.db_path)
    if not failures.empty:
        st.dataframe(failures, use_container_width=True, hide_index=True)
    else:
        st.caption("No critic failures recorded.")
