def evaluate_trace(trace, scenario):
    failures = []

    # 1. Did the scenario actually run?
    if trace.get("status") != "completed":
        failures.append("Scenario did not complete")

    # 2. Did the expected tool run?
    expected_tool = scenario.get("expected_tool")
    actual_tool = trace.get("tool_called")

    if expected_tool and actual_tool != expected_tool:
        failures.append(
            f"Expected tool '{expected_tool}', got '{actual_tool}'"
        )

    # 3. Did it stay within latency budget?
    max_latency = scenario.get("max_latency_ms")
    actual_latency = trace.get("latency_ms")

    if (
        max_latency is not None
        and actual_latency is not None
        and actual_latency > max_latency
    ):
        failures.append(
            f"Latency {actual_latency}ms exceeded "
            f"limit {max_latency}ms"
        )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
    }
