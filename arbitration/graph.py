"""LangGraph orchestration.

Graph shape (matches the project's architecture spec):

    START -> parse_input -> {accuracy_critic, logic_critic, completeness_critic}  (parallel fan-out)
          -> collect_critiques (fan-in)
          -> detect_disagreements
          -> [conditional] -> adjudicate | short_circuit_verdict | all_failed_verdict
          -> synthesize_verdict -> END
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from arbitration.adjudicator import run_adjudicator
from arbitration.config import load_settings
from arbitration.critics import run_critic
from arbitration.disagreement import all_critics_failed, detect_disagreements, is_clean_sweep
from arbitration.models import ArbitrationRecord, CriticRunResult, Disagreement, Verdict

CRITIC_KEYS = ("accuracy_critic", "logic_critic", "completeness_critic")


class ArbitrationState(TypedDict):
    original_prompt: str | None
    original_output: str
    critic_runs: Annotated[list[CriticRunResult], operator.add]
    disagreements: list[Disagreement]
    verdict: Verdict | None


def _parse_input(state: ArbitrationState) -> dict:
    output = (state["original_output"] or "").strip()
    if not output:
        raise ValueError("original_output must be a non-empty string")
    prompt = state["original_prompt"].strip() if state.get("original_prompt") else None
    return {"original_output": output, "original_prompt": prompt}


def _make_critic_node(critic_key: str):
    def _node(state: ArbitrationState) -> dict:
        settings = load_settings()
        config = getattr(settings, critic_key)
        result = run_critic(config, settings, state["original_prompt"], state["original_output"])
        return {"critic_runs": [result]}

    _node.__name__ = f"run_{critic_key}"
    return _node


def _collect_critiques(state: ArbitrationState) -> dict:
    # Fan-in point. LangGraph has already merged all three critic_runs updates
    # (via the `operator.add` reducer) before this node runs. Nothing to do -
    # this node exists to make the "critique collection" phase explicit.
    return {}


def _detect_disagreements(state: ArbitrationState) -> dict:
    return {"disagreements": detect_disagreements(state["critic_runs"])}


def _route_after_disagreements(state: ArbitrationState) -> str:
    critic_runs = state["critic_runs"]
    if all_critics_failed(critic_runs):
        return "all_failed"
    if is_clean_sweep(critic_runs, state["disagreements"]):
        return "short_circuit"
    return "adjudicate"


def _short_circuit_verdict(state: ArbitrationState) -> dict:
    ok_runs = [r for r in state["critic_runs"] if r.ok and r.report is not None]
    avg_score_5 = sum(r.report.score for r in ok_runs) / len(ok_runs)
    avg_confidence = sum(r.report.confidence for r in ok_runs) / len(ok_runs)
    verdict = Verdict(
        overall_score=round((avg_score_5 / 5) * 10),
        confidence=round(min(1.0, avg_confidence), 2),
        confirmed_issues=[],
        dismissed_flags=[],
        summary=(
            "All critics independently agreed this output is clean: no issues were raised on "
            "accuracy, logic, or completeness, and no disagreements were detected. Adjudication "
            "was short-circuited."
        ),
        short_circuited=True,
    )
    return {"verdict": verdict}


def _all_failed_verdict(state: ArbitrationState) -> dict:
    errors = "; ".join(f"{r.critic}: {r.error}" for r in state["critic_runs"] if r.error)
    verdict = Verdict(
        overall_score=1,
        confidence=0.0,
        confirmed_issues=[],
        dismissed_flags=[],
        summary=f"Every critic failed to return a report ({errors}); no verdict could be substantively formed.",
    )
    return {"verdict": verdict}


def _adjudicate(state: ArbitrationState) -> dict:
    settings = load_settings()
    verdict = run_adjudicator(
        settings.adjudicator,
        settings,
        state["original_prompt"],
        state["original_output"],
        state["critic_runs"],
        state["disagreements"],
    )
    return {"verdict": verdict}


def _synthesize_verdict(state: ArbitrationState) -> dict:
    """Final pass: append a graceful-degradation note when some (not all) critics failed."""
    verdict = state["verdict"]
    critic_runs = state["critic_runs"]
    failed = [r for r in critic_runs if not r.ok]

    if 0 < len(failed) < len(critic_runs):
        names = ", ".join(r.critic for r in failed)
        verdict = verdict.model_copy(
            update={
                "summary": (
                    verdict.summary
                    + f" Note: {names} did not return a report; this verdict has reduced "
                    f"confidence because it excludes that dimension."
                )
            }
        )
    return {"verdict": verdict}


def build_graph():
    graph = StateGraph(ArbitrationState)

    graph.add_node("parse_input", _parse_input)
    for key in CRITIC_KEYS:
        graph.add_node(key, _make_critic_node(key))
    graph.add_node("collect_critiques", _collect_critiques)
    graph.add_node("detect_disagreements", _detect_disagreements)
    graph.add_node("adjudicate", _adjudicate)
    graph.add_node("short_circuit_verdict", _short_circuit_verdict)
    graph.add_node("all_failed_verdict", _all_failed_verdict)
    graph.add_node("synthesize_verdict", _synthesize_verdict)

    graph.add_edge(START, "parse_input")
    for key in CRITIC_KEYS:
        graph.add_edge("parse_input", key)
        graph.add_edge(key, "collect_critiques")

    graph.add_edge("collect_critiques", "detect_disagreements")
    graph.add_conditional_edges(
        "detect_disagreements",
        _route_after_disagreements,
        {
            "adjudicate": "adjudicate",
            "short_circuit": "short_circuit_verdict",
            "all_failed": "all_failed_verdict",
        },
    )
    graph.add_edge("adjudicate", "synthesize_verdict")
    graph.add_edge("short_circuit_verdict", "synthesize_verdict")
    graph.add_edge("all_failed_verdict", "synthesize_verdict")
    graph.add_edge("synthesize_verdict", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_arbitration(original_output: str, original_prompt: str | None = None) -> ArbitrationRecord:
    final_state = get_graph().invoke(
        {
            "original_prompt": original_prompt,
            "original_output": original_output,
            "critic_runs": [],
            "disagreements": [],
            "verdict": None,
        }
    )
    return ArbitrationRecord(
        original_prompt=final_state["original_prompt"],
        original_output=final_state["original_output"],
        critic_runs=final_state["critic_runs"],
        disagreements=final_state["disagreements"],
        verdict=final_state["verdict"],
    )
