from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest

from evals import pixi_bridge as bridge


@pytest.fixture
def live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOICE_AGENT_VERSION", "test-build-sha-1234567890")
    monkeypatch.setenv("VOICE_BRIDGE_TOKEN", "test-bridge-token-with-at-least-32-characters")


def base_request() -> dict[str, Any]:
    return {
        "protocol_version": 2,
        "scenario": {
            "id": "country-lookup",
            "name": "Japan country lookup",
            "turns": ["Tell me about Japan."],
            "assertions": [
                {"id": "mentions-japan", "kind": "contains", "expected_text": "Japan"}
            ],
        },
        "experiment": {"variant": "baseline", "candidate": None},
    }


def test_candidate_operations_are_explicit_and_allow_list_only() -> None:
    assert bridge.parse_candidate_operation("enable_tool:get_country_info") == {
        "kind": "enable_tool",
        "tool": "get_country_info",
    }
    assert bridge.parse_candidate_operation(" prompt_append: Answer in one sentence. ") == {
        "kind": "prompt_append",
        "text": "Answer in one sentence.",
    }
    with pytest.raises(bridge.BridgeRequestError, match="Unsupported candidate change"):
        bridge.parse_candidate_operation("run arbitrary Python from this string")
    with pytest.raises(bridge.BridgeRequestError, match="non-empty instruction text"):
        bridge.parse_candidate_operation("prompt_append:   ")


def test_candidate_version_is_stable_and_changes_with_candidate_configuration() -> None:
    operation = {"kind": "enable_tool", "tool": "get_country_info"}
    first = bridge.candidate_version("build-abc", "candidate-1", operation)
    assert first == bridge.candidate_version("build-abc", "candidate-1", operation)
    assert first != bridge.candidate_version("build-abc", "candidate-2", operation)
    assert first != bridge.candidate_version(
        "build-abc", "candidate-1", {"kind": "prompt_append", "text": "different"}
    )


def test_request_validation_rejects_missing_base_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOICE_AGENT_VERSION", raising=False)
    with pytest.raises(bridge.BridgeRequestError) as error:
        bridge.evaluate_bridge_request(base_request())
    assert error.value.status_code == 503


def test_request_validation_keeps_baseline_and_candidate_metadata(live_env: None) -> None:
    baseline = bridge.evaluate_bridge_request(base_request())
    assert baseline["agent_version"] == "test-build-sha-1234567890"
    assert baseline["variant"] == "baseline"
    assert baseline["candidate_id"] is None
    assert baseline["_execution"]["operation"] is None

    candidate = base_request()
    candidate["experiment"] = {
        "variant": "candidate",
        "candidate": {
            "id": "candidate-123",
            "hypothesis": "Country lookups become reliable.",
            "proposed_change": "enable_tool:get_country_info",
        },
    }
    result = bridge.evaluate_bridge_request(candidate)
    assert result["variant"] == "candidate"
    assert result["candidate_id"] == "candidate-123"
    assert result["agent_version"].startswith("test-build-sha-1234567890+candidate-")
    assert result["agent_version"] != baseline["agent_version"]
    assert result["_execution"]["operation"] == {
        "kind": "enable_tool",
        "tool": "get_country_info",
    }


def test_request_validation_rejects_wrong_protocol_and_unbounded_turns(live_env: None) -> None:
    payload = base_request()
    payload["protocol_version"] = 1
    with pytest.raises(bridge.BridgeRequestError, match="protocol_version 2"):
        bridge.evaluate_bridge_request(payload)

    payload = base_request()
    payload["scenario"]["turns"] = ["x"] * (bridge.MAX_TURNS + 1)
    with pytest.raises(bridge.BridgeRequestError, match="1–12"):
        bridge.evaluate_bridge_request(payload)


def test_trace_collector_records_text_output_tools_and_latency() -> None:
    collector = bridge.RequestTraceCollector()
    collector.begin_turn("Tell me about Japan.")
    collector.record_tool_latency("get_country_info", 1.25)
    function_call = SimpleNamespace(
        name="get_country_info",
        arguments='{"country":"Japan"}',
    )
    function_output = SimpleNamespace(output="Japan is an island country.", is_error=False)
    collector.on_function_tools_executed(
        SimpleNamespace(zipped=lambda: [(function_call, function_output)])
    )
    collector.on_conversation_item_added(
        SimpleNamespace(
            item=SimpleNamespace(
                role="assistant",
                content=["Japan is an island country."],
            )
        )
    )
    collector.finish_turn(420.0)
    data = collector.to_dict(agent_version="test-build", candidate_id=None)
    assert data["input_mode"] == "text"
    assert data["latency_basis"] == "text_input_to_completed_response"
    assert data["turns"][0]["user_input"] == "Tell me about Japan."
    assert data["turns"][0]["assistant_output"] == "Japan is an island country."
    assert data["turns"][0]["tools"][0]["name"] == "get_country_info"
    assert data["turns"][0]["tools"][0]["latency_ms"] == 1.25
    assert data["turns"][0]["latency"]["e2e_ms"] == 420.0


def test_http_bridge_requires_bearer_auth_and_returns_run_evidence(
    monkeypatch: pytest.MonkeyPatch,
    live_env: None,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run_live_session(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "transcript": "Japan is an island country in East Asia.",
            "latency_ms": 420.0,
            "tools_called": ["get_country_info"],
            "voice_trace": {"status": "completed", "input_mode": "text", "turns": []},
        }

    monkeypatch.setattr(bridge, "run_live_session", fake_run_live_session)
    server = bridge.ThreadingPixiServer(("127.0.0.1", 0), bridge.PixiBridgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/pixi/evaluate"
    try:
        body = json.dumps(base_request()).encode()
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer test-bridge-token-with-at-least-32-characters",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
            assert response.status == 200
        assert payload["protocol_version"] == 2
        assert payload["status"] == "completed"
        assert payload["agent_version"] == "test-build-sha-1234567890"
        assert payload["variant"] == "baseline"
        assert payload["candidate_id"] is None
        assert payload["transcript"] == "Japan is an island country in East Asia."
        assert payload["tools_called"] == ["get_country_info"]
        assert payload["voice_trace"]["input_mode"] == "text"
        assert calls[0]["turns"] == ["Tell me about Japan."]
        assert calls[0]["candidate_id"] is None

        unauthorized = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer wrong-token"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(unauthorized, timeout=5)
        assert error.value.code == 401
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bridge_rejects_candidate_without_allow_list_operation(live_env: None) -> None:
    payload = base_request()
    payload["experiment"] = {
        "variant": "candidate",
        "candidate": {
            "id": "candidate-123",
            "hypothesis": "Change something.",
            "proposed_change": "execute code",
        },
    }
    with pytest.raises(bridge.BridgeRequestError) as error:
        bridge.evaluate_bridge_request(payload)
    assert error.value.status_code == 422
