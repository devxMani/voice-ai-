import json
from pathlib import Path
from datetime import datetime


SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"


def load_scenarios():
    scenarios = []

    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        with open(path, "r") as f:
            scenarios.append(json.load(f))

    return scenarios


def run_scenario(scenario):
    print(f"\nRunning: {scenario['name']}")
    print(f"User: {scenario['user_input']}")

    # Temporary placeholder.
    # We will connect this to the real voice agent next.
    trace = {
        "scenario": scenario["name"],
        "user_input": scenario["user_input"],
        "expected_tool": scenario.get("expected_tool"),
        "timestamp": datetime.now().isoformat(),
        "status": "not_run",
    }

    return trace


def main():
    scenarios = load_scenarios()

    print("=" * 50)
    print("Voice Reliability Lab")
    print("=" * 50)

    results = []

    for scenario in scenarios:
        trace = run_scenario(scenario)
        results.append(trace)

    RESULTS_DIR.mkdir(exist_ok=True)

    output = RESULTS_DIR / "latest.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Scenarios: {len(results)}")
    print(f"Results:   {output}")
    print("=" * 50)


if __name__ == "__main__":
    main()