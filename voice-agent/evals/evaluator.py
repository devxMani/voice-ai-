"""Evaluate observable run evidence against a scenario's explicit checks."""

from __future__ import annotations

from typing import Any


def evaluate_trace(trace: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    inconclusive: list[str] = []

    if trace.get("status") != "completed":
        inconclusive.append("Scenario did not complete with a usable trace.")

    expected_tool = scenario.get("expected_tool")
    if expected_tool:
        actual_tools = trace.get("tools_called")
        if not isinstance(actual_tools, list):
            legacy_tool = trace.get("tool_called")
            actual_tools = [legacy_tool] if legacy_tool else []
        if expected_tool not in actual_tools:
            failures.append(
                f"Expected tool '{expected_tool}', observed {actual_tools or 'no tool calls'}."
            )

    max_latency = scenario.get("max_latency_ms")
    actual_latency = trace.get("latency_ms")
    if max_latency is not None:
        if actual_latency is None:
            inconclusive.append("Latency evidence is unavailable.")
        elif actual_latency > max_latency:
            failures.append(
                f"Latency {actual_latency:.1f}ms exceeded the {max_latency}ms limit."
            )

    outcome = "failed" if failures else "inconclusive" if inconclusive else "passed"
    return {
        "passed": outcome == "passed",
        "outcome": outcome,
        "failures": failures,
        "inconclusive_reasons": inconclusive,
    }
