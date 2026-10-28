"""Run the saved scenario set against the live in-process LiveKit agent.

This command performs real model/tool calls. It requires LiveKit credentials and an
immutable VOICE_AGENT_VERSION. The tool-timeout scenario is marked not_applicable until
a dedicated fault-injection adapter is configured; it is not silently treated as a pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.evaluator import evaluate_trace
from evals.pixi_bridge import (
    BridgeRequestError,
    candidate_version,
    parse_candidate_operation,
    run_live_session,
)


SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"


def load_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            scenarios.append(json.load(handle))
    return scenarios


def _local_candidate_id(operation: dict[str, Any]) -> str:
    # Stable for one CLI invocation/configuration, but clearly not a Pixi database ID.
    import hashlib

    canonical = json.dumps(operation, sort_keys=True, separators=(",", ":"))
    return "runner-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


async def run_scenario(
    scenario: dict[str, Any],
    *,
    base_version: str,
    operation: dict[str, Any] | None,
    candidate_id: str | None,
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    name = str(scenario.get("name", "Unnamed scenario"))
    user_input = str(scenario.get("user_input", ""))
    turns = scenario.get("turns") or ([user_input] if user_input.strip() else [])
    variant = "candidate" if operation is not None else "baseline"
    version = (
        candidate_version(base_version, candidate_id or "local-runner-candidate", operation)
        if operation is not None
        else base_version
    )
    expected_tool = scenario.get("expected_tool")

    # A timeout test requires a controlled, per-scenario tool fault. Running it with
    # a normal tool would be a false test pass, so preserve the not_applicable status.
    fault = scenario.get("fault_injection")
    if "timeout" in name.casefold() and not fault:
        return {
            "scenario": name,
            "user_input": user_input,
            "expected_tool": expected_tool,
            "timestamp": timestamp,
            "variant": variant,
            "agent_version": version,
            "status": "not_applicable",
            "reason": "No explicit tool-timeout fault injection is configured.",
            "evaluation": {
                "outcome": "not_applicable",
                "passed": None,
                "failures": [],
                "inconclusive_reasons": [],
            },
        }

    if not turns:
        return {
            "scenario": name,
            "user_input": user_input,
            "expected_tool": expected_tool,
            "timestamp": timestamp,
            "variant": variant,
            "agent_version": version,
            "status": "inconclusive",
            "reason": "Scenario contains no user_input or turns.",
            "evaluation": {
                "outcome": "inconclusive",
                "passed": None,
                "failures": [],
                "inconclusive_reasons": ["Scenario contains no user input."],
            },
        }

    try:
        result = await run_live_session(
            turns=[str(turn) for turn in turns],
            candidate_id=candidate_id,
            operation=operation,
            agent_version=version,
        )
        trace = result["voice_trace"]
        tools_called = result["tools_called"]
        legacy_trace = {
            "scenario": name,
            "user_input": user_input or "\n".join(turns),
            "expected_tool": expected_tool,
            "timestamp": timestamp,
            "status": trace.get("status", "inconclusive"),
            "tool_called": tools_called[0] if tools_called else None,
            "tools_called": tools_called,
            "latency_ms": result.get("latency_ms"),
            "assistant_output": result.get("transcript", ""),
            "voice_trace": trace,
        }
        legacy_trace["evaluation"] = evaluate_trace(legacy_trace, scenario)
        return legacy_trace
    except BridgeRequestError as exc:
        return {
            "scenario": name,
            "user_input": user_input,
            "expected_tool": expected_tool,
            "timestamp": timestamp,
            "variant": variant,
            "agent_version": version,
            "status": "inconclusive",
            "reason": exc.detail,
            "evaluation": {
                "outcome": "inconclusive",
                "passed": None,
                "failures": [],
                "inconclusive_reasons": [exc.detail],
            },
        }
    except Exception as exc:  # noqa: BLE001 - preserve failures as inconclusive evidence
        return {
            "scenario": name,
            "user_input": user_input,
            "expected_tool": expected_tool,
            "timestamp": timestamp,
            "variant": variant,
            "agent_version": version,
            "status": "inconclusive",
            "reason": type(exc).__name__,
            "evaluation": {
                "outcome": "inconclusive",
                "passed": None,
                "failures": [],
                "inconclusive_reasons": ["LiveKit session failed; details are in local logs."],
            },
        }


async def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_version = os.getenv("VOICE_AGENT_VERSION", "").strip()
    if not base_version:
        raise SystemExit("Set VOICE_AGENT_VERSION to an immutable source/build identifier in .env.local.")
    operation = parse_candidate_operation(args.candidate_operation) if args.candidate_operation else None
    candidate_id = _local_candidate_id(operation) if operation is not None else None

    scenarios = load_scenarios()
    if args.scenario:
        scenarios = [scenario for scenario in scenarios if scenario.get("name") == args.scenario]
        if not scenarios:
            raise SystemExit(f"No scenario named {args.scenario!r} was found.")

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        label = str(scenario.get("name", "Unnamed scenario"))
        print(f"Running {label} ({'candidate' if operation else 'baseline'})…", flush=True)
        result = await run_scenario(
            scenario,
            base_version=base_version,
            operation=operation,
            candidate_id=candidate_id,
        )
        results.append(result)
        evaluation = result["evaluation"]
        print(
            f"  status={result['status']} outcome={evaluation.get('outcome', 'unknown')} "
            f"latency_ms={result.get('latency_ms', 'n/a')}",
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="Run only the exact scenario name from evals/scenarios.")
    parser.add_argument(
        "--candidate-operation",
        help="Optional allow-listed candidate operation, e.g. enable_tool:get_country_info.",
    )
    args = parser.parse_args()
    results = asyncio.run(run_all(args))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "latest.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = {
        "passed": sum(item["evaluation"].get("outcome") == "passed" for item in results),
        "failed": sum(item["evaluation"].get("outcome") == "failed" for item in results),
        "inconclusive": sum(item["evaluation"].get("outcome") == "inconclusive" for item in results),
        "not_applicable": sum(item["evaluation"].get("outcome") == "not_applicable" for item in results),
    }
    print(f"Saved {len(results)} result(s) to {output}")
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
