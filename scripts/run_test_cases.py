"""Phase 6 portfolio test cases: run four canned outputs through the pipeline and
print/persist their verdicts. Works fully offline in mock mode (the default).

Usage:  python scripts/run_test_cases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbitration.config import load_settings
from arbitration.graph import run_arbitration
from arbitration.storage import save_arbitration

TEST_CASES = [
    {
        "name": "factually_incorrect",
        "description": "A response with three planted factual errors.",
        "prompt": "List three facts about Napoleon, the Eiffel Tower, and Einstein.",
        "output": (
            "Here are three facts: Napoleon was famously short, standing just over five feet "
            "tall, which shaped his military reputation. The Eiffel Tower is located in London "
            "and draws millions of visitors each year. Einstein failed math as a student before "
            "becoming a physicist."
        ),
    },
    {
        "name": "logically_flawed",
        "description": "A response riddled with classic reasoning fallacies.",
        "prompt": "Should the city invest more in public parks?",
        "output": (
            "Everyone always supports more green space in their neighborhood. Either the city "
            "builds more parks or there is no way to improve resident wellbeing. Parks improve "
            "neighborhoods because it is good for neighborhoods. The city council held a budget "
            "meeting last spring. Therefore, the mayor should resign immediately."
        ),
    },
    {
        "name": "misses_the_point",
        "description": "Technically answers part of the question but ignores the rest.",
        "prompt": "What is the capital of France? Also, explain the historical reasons Paris was chosen over other French cities.",
        "output": "The capital of France is Paris.",
    },
    {
        "name": "genuinely_good",
        "description": "A clean, accurate, well-reasoned, complete response - should get a clean bill of health.",
        "prompt": "What causes rainbows to form?",
        "output": (
            "Rainbows form when sunlight is refracted, reflected, and dispersed inside water "
            "droplets in the atmosphere, splitting white light into its component colors. This "
            "is why rainbows typically appear opposite the sun during or after rainfall."
        ),
    },
]


def main() -> None:
    settings = load_settings()
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [f"# Test case verdicts (provider_mode={settings.provider_mode})\n"]
    results = {}

    for case in TEST_CASES:
        record = run_arbitration(case["output"], case["prompt"])
        save_arbitration(settings.db_path, record)
        results[case["name"]] = json.loads(record.model_dump_json())

        v = record.verdict
        print(f"\n=== {case['name']} ===")
        print(f"  {case['description']}")
        print(f"  overall_score={v.overall_score}/10  confidence={v.confidence:.0%}  "
              f"short_circuited={v.short_circuited}")
        print(f"  confirmed_issues={len(v.confirmed_issues)}  dismissed_flags={len(v.dismissed_flags)}  "
              f"disagreements={len(record.disagreements)}")
        for run in record.critic_runs:
            if run.ok:
                print(f"    - {run.critic}: score={run.report.score}/5 issues={len(run.report.issues)}")
            else:
                print(f"    - {run.critic}: FAILED ({run.error})")
        print(f"  summary: {v.summary}")

        summary_lines.append(f"## {case['name']}\n")
        summary_lines.append(f"*{case['description']}*\n")
        summary_lines.append(f"- **id**: `{record.id}`")
        summary_lines.append(f"- **overall_score**: {v.overall_score}/10")
        summary_lines.append(f"- **confidence**: {v.confidence:.0%}")
        summary_lines.append(f"- **short_circuited**: {v.short_circuited}")
        summary_lines.append(f"- **confirmed_issues**: {len(v.confirmed_issues)}")
        summary_lines.append(f"- **dismissed_flags**: {len(v.dismissed_flags)}")
        summary_lines.append(f"- **disagreements**: {len(record.disagreements)}")
        summary_lines.append(f"- **summary**: {v.summary}\n")

    (out_dir / "test_case_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "test_case_results.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nSaved full JSON to {out_dir / 'test_case_results.json'}")
    print(f"Saved markdown summary to {out_dir / 'test_case_results.md'}")
    print(f"All four runs were also persisted to {settings.db_path} - open the Streamlit UI's "
          f"History/Analytics tabs to explore them.")


if __name__ == "__main__":
    main()
