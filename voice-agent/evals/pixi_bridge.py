"""Local HTTP bridge for Pixi to evaluate this LiveKit AgentSession.

This is a scripted-text test harness around the real DefaultAgent, LiveKit Inference
LLM/TTS and its registered tools. It does not inject caller audio, so STT, acoustic
latency and interruption fields are intentionally null unless the runtime emits them.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from livekit.agents import AgentSession, TurnHandlingOptions, inference

from agent import DefaultAgent
from evals.trace import ToolTrace, TurnTrace, VoiceTrace

load_dotenv(".env.local")

logger = logging.getLogger("pixi-livekit-bridge")
MODEL_NAME = os.getenv("VOICE_AGENT_LLM_MODEL", "google/gemma-4-31b-it")
MAX_BODY_BYTES = 1_000_000
MAX_TURNS = 12
MAX_TURN_CHARS = 4_000
MAX_CANDIDATE_CHARS = 4_000


class BridgeRequestError(Exception):
    """Expected client/configuration error with a safe HTTP status."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def parse_candidate_operation(proposed_change: str) -> dict[str, Any]:
    """Parse an explicit allow-listed candidate operation. Never execute supplied code."""
    change = proposed_change.strip()
    if change == "enable_tool:get_country_info":
        return {"kind": "enable_tool", "tool": "get_country_info"}
    if change.startswith("prompt_append:"):
        prompt = change.partition(":")[2].strip()
        if not prompt:
            raise BridgeRequestError(422, "prompt_append requires non-empty instruction text.")
        if len(prompt) > MAX_CANDIDATE_CHARS:
            raise BridgeRequestError(422, "Candidate instruction text exceeds the 4000-character limit.")
        return {"kind": "prompt_append", "text": prompt}
    raise BridgeRequestError(
        422,
        "Unsupported candidate change. Use exactly 'enable_tool:get_country_info' "
        "or 'prompt_append: <instruction text>'. Arbitrary code changes are not executed.",
    )


def candidate_version(base_version: str, candidate_id: str, operation: dict[str, Any]) -> str:
    """Create an immutable identity for this base build plus candidate configuration."""
    canonical = json.dumps(
        {"base_version": base_version, "candidate_id": candidate_id, "operation": operation},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{base_version}+candidate-{digest}"


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    try:
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif getattr(part, "text", None):
                parts.append(str(part.text))
    except TypeError:
        return ""
    return " ".join(part for part in parts if part).strip()


class RequestTraceCollector:
    """Collect an isolated trace object for one HTTP request; does not write shared files."""

    def __init__(self) -> None:
        self.trace = VoiceTrace()
        self.trace_started_at = datetime.now(timezone.utc).isoformat()
        self.current_turn: TurnTrace | None = None
        self.turn_started_at: float | None = None
        self._tool_latencies: dict[str, list[float]] = {}

    def begin_turn(self, user_input: str) -> TurnTrace:
        turn = TurnTrace(user_input=user_input, status="running")
        self.trace.turns.append(turn)
        self.current_turn = turn
        self.turn_started_at = time.perf_counter()
        return turn

    def finish_turn(self, duration_ms: float) -> None:
        if self.current_turn is None:
            return
        # In this text-driven harness, this is text-input-to-completed-response time.
        # It is not an acoustic end-to-end latency or time-to-first-audio measurement.
        if self.current_turn.e2e_latency_ms is None:
            self.current_turn.e2e_latency_ms = duration_ms
        if self.current_turn.assistant_output.strip() and not self.current_turn.error:
            self.current_turn.status = "completed"
        elif self.current_turn.status == "running":
            self.current_turn.status = "inconclusive"

    def record_tool_latency(self, name: str, latency_ms: float) -> None:
        self._tool_latencies.setdefault(name, []).append(latency_ms)

    def _take_tool_latency(self, name: str) -> float | None:
        values = self._tool_latencies.get(name)
        return values.pop(0) if values else None

    def attach(self, session: AgentSession) -> None:
        session.on("conversation_item_added", self.on_conversation_item_added)
        session.on("function_tools_executed", self.on_function_tools_executed)
        session.on("agent_state_changed", self.on_agent_state_changed)
        session.on("metrics_collected", self.on_session_metrics)

    def on_conversation_item_added(self, event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None or not hasattr(item, "role"):
            return
        role = str(item.role).lower()
        text = _text_content(getattr(item, "content", None))
        if role == "user":
            if text and (
                self.current_turn is None or self.current_turn.user_input != text
            ):
                self.begin_turn(text)
            return
        if role != "assistant" or not text or self.current_turn is None:
            return
        # Tool-call messages often have no text; only save actual assistant prose.
        self.current_turn.assistant_output = text
        self.current_turn.status = "completed"

    def on_function_tools_executed(self, event: Any) -> None:
        if self.current_turn is None:
            return
        try:
            zipped = event.zipped()
        except (AttributeError, TypeError):
            return
        for function_call, function_output in zipped:
            raw_arguments = getattr(function_call, "arguments", None)
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            except (json.JSONDecodeError, TypeError):
                arguments = {"raw": str(raw_arguments)}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            name = str(getattr(function_call, "name", "unknown"))
            output = str(getattr(function_output, "output", "") or "")
            is_error = bool(getattr(function_output, "is_error", False))
            self.current_turn.tools.append(
                ToolTrace(
                    name=name,
                    arguments=arguments,
                    output=output,
                    latency_ms=self._take_tool_latency(name),
                    success=not is_error,
                    error=output if is_error else None,
                )
            )

    def on_agent_state_changed(self, event: Any) -> None:
        if self.current_turn is None or self.turn_started_at is None:
            return
        if str(getattr(event, "new_state", "")).lower() == "speaking":
            # This state event is meaningful only if LiveKit reports it in this
            # text-driven session. finish_turn() records complete-turn latency otherwise.
            self.current_turn.e2e_latency_ms = (
                time.perf_counter() - self.turn_started_at
            ) * 1000

    def on_session_metrics(self, event: Any) -> None:
        metrics = getattr(event, "metrics", event)
        name = type(metrics).__name__
        if self.current_turn is None:
            return
        if name == "STTMetrics":
            self.record_stt_metrics(metrics)
        elif name == "EOUMetrics":
            self.record_eou_metrics(metrics)
        elif name == "LLMMetrics":
            self.record_llm_metrics(metrics)
        elif name == "TTSMetrics":
            self.record_tts_metrics(metrics)

    def record_stt_metrics(self, metrics: Any) -> None:
        if self.current_turn is not None and getattr(metrics, "duration", None) is not None:
            self.current_turn.stt_processing_ms = float(metrics.duration) * 1000

    def record_eou_metrics(self, metrics: Any) -> None:
        if self.current_turn is not None and getattr(metrics, "end_of_utterance_delay", None) is not None:
            self.current_turn.eot_latency_ms = float(metrics.end_of_utterance_delay) * 1000

    def record_llm_metrics(self, metrics: Any) -> None:
        if self.current_turn is not None and getattr(metrics, "ttft", None) is not None:
            self.current_turn.llm_ttft_ms = float(metrics.ttft) * 1000

    def record_tts_metrics(self, metrics: Any) -> None:
        if self.current_turn is not None and getattr(metrics, "ttfb", None) is not None:
            self.current_turn.tts_ttfb_ms = float(metrics.ttfb) * 1000

    def to_dict(self, *, agent_version: str, candidate_id: str | None) -> dict[str, Any]:
        data = self.trace.to_dict()
        data.update(
            {
                "trace_started_at": self.trace_started_at,
                "input_mode": "text",
                "latency_basis": "text_input_to_completed_response",
                "agent_version": agent_version,
                "candidate_id": candidate_id,
            }
        )
        return data


def _make_session() -> AgentSession:
    return AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        stt_context_options={"keyterm_detection": {"enabled": True}},
        llm=inference.LLM(model=MODEL_NAME),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            language="en",
        ),
        expressive=True,
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            preemptive_generation={"enabled": True},
        ),
        vad=inference.VAD(),
    )


async def run_live_session(
    *,
    turns: list[str],
    candidate_id: str | None,
    operation: dict[str, Any] | None,
    agent_version: str,
) -> dict[str, Any]:
    collector = RequestTraceCollector()
    session = _make_session()
    agent = DefaultAgent(
        collector,
        enable_country_info=bool(operation and operation.get("kind") == "enable_tool"),
        prompt_append=str(operation["text"]) if operation and operation.get("kind") == "prompt_append" else None,
        suppress_greeting=True,
    )
    collector.attach(session)
    started = time.perf_counter()
    try:
        await session.start(agent=agent)
        for user_input in turns:
            collector.begin_turn(user_input)
            turn_started = time.perf_counter()
            result = await session.run(user_input=user_input, input_modality="text")
            duration_ms = (time.perf_counter() - turn_started) * 1000
            collector.finish_turn(duration_ms)
            exception = getattr(result, "exception", None)
            if callable(exception):
                exception = exception()
            if exception:
                if collector.current_turn is not None:
                    collector.current_turn.error = str(exception)
                    collector.current_turn.status = "inconclusive"
                raise RuntimeError("LiveKit AgentSession failed to complete a scripted turn.") from exception

        turns_trace = collector.trace.turns
        if not turns_trace or any(
            turn.status != "completed" or not turn.assistant_output.strip()
            for turn in turns_trace
        ):
            raise RuntimeError("The LiveKit session did not produce a complete traced assistant response.")

        transcript = "\n".join(turn.assistant_output for turn in turns_trace).strip()
        tools_called = list(
            dict.fromkeys(tool.name for turn in turns_trace for tool in turn.tools)
        )
        latencies = [
            float(turn.e2e_latency_ms)
            for turn in turns_trace
            if turn.e2e_latency_ms is not None
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000
        trace = collector.to_dict(agent_version=agent_version, candidate_id=candidate_id)
        trace["measured_turn_count"] = len(turns_trace)
        trace["scenario_elapsed_ms"] = elapsed_ms
        return {
            "transcript": transcript,
            "latency_ms": max(latencies) if latencies else elapsed_ms,
            "tools_called": tools_called,
            "trace": trace,
        }
    finally:
        await session.close()


def evaluate_bridge_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate Pixi protocol v2 request, then run a real in-process LiveKit session."""
    if payload.get("protocol_version") != 2:
        raise BridgeRequestError(400, "Only Pixi bridge protocol_version 2 is supported.")
    scenario = payload.get("scenario")
    experiment = payload.get("experiment")
    if not isinstance(scenario, dict) or not isinstance(experiment, dict):
        raise BridgeRequestError(400, "Request must include scenario and experiment objects.")
    turns = scenario.get("turns")
    if (
        not isinstance(turns, list)
        or not turns
        or len(turns) > MAX_TURNS
        or any(not isinstance(turn, str) or not turn.strip() or len(turn) > MAX_TURN_CHARS for turn in turns)
    ):
        raise BridgeRequestError(422, "Scenario turns must contain 1–12 non-empty text inputs, each <= 4000 characters.")

    variant = experiment.get("variant")
    if variant not in ("baseline", "candidate"):
        raise BridgeRequestError(422, "Experiment variant must be 'baseline' or 'candidate'.")
    candidate = experiment.get("candidate")
    candidate_id: str | None = None
    operation: dict[str, Any] | None = None
    base_version = os.getenv("VOICE_AGENT_VERSION", "").strip()
    if not base_version or len(base_version) > 150:
        raise BridgeRequestError(503, "VOICE_AGENT_VERSION must identify the immutable deployed/source build.")
    if variant == "baseline":
        if candidate is not None:
            raise BridgeRequestError(422, "Baseline experiments must set candidate to null.")
        agent_version = base_version
    else:
        if not isinstance(candidate, dict):
            raise BridgeRequestError(422, "Candidate experiments require a candidate proposal.")
        candidate_id = candidate.get("id")
        hypothesis = candidate.get("hypothesis")
        proposed_change = candidate.get("proposed_change")
        if not isinstance(candidate_id, str) or not candidate_id.strip() or len(candidate_id) > 36:
            raise BridgeRequestError(422, "Candidate id must be a non-empty identifier no longer than 36 characters.")
        if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 5000:
            raise BridgeRequestError(422, "Candidate hypothesis must be non-empty and <= 5000 characters.")
        if not isinstance(proposed_change, str) or not proposed_change.strip() or len(proposed_change) > 20_000:
            raise BridgeRequestError(422, "Candidate proposed_change must be non-empty and <= 20000 characters.")
        operation = parse_candidate_operation(proposed_change)
        agent_version = candidate_version(base_version, candidate_id, operation)

    return {
        "protocol_version": 2,
        "status": "completed",
        "transcript": "",
        "latency_ms": None,
        "tools_called": [],
        "provider": "LiveKit Agents",
        "model": MODEL_NAME,
        "agent_version": agent_version,
        "variant": variant,
        "candidate_id": candidate_id,
        "_execution": {"turns": turns, "operation": operation},
        "_execution_candidate_id": candidate_id,
    }


class PixiBridgeHandler(BaseHTTPRequestHandler):
    server_version = "VoiceReliabilityPixiBridge/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Do not log transcript bodies, bearer credentials, or upstream errors.
        logger.info("bridge request: " + format, *args)

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/pixi/evaluate":
            self._write_json(404, {"detail": "Not found."})
            return
        expected_token = os.getenv("VOICE_BRIDGE_TOKEN", "")
        supplied = self.headers.get("Authorization", "")
        expected_header = "Bearer " + expected_token
        if len(expected_token) < 32 or not hmac.compare_digest(supplied, expected_header):
            self._write_json(401, {"detail": "Missing or invalid bridge bearer token."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(411, {"detail": "Content-Length is required."})
            return
        if content_length <= 0:
            self._write_json(400, {"detail": "Request body is required."})
            return
        if content_length > MAX_BODY_BYTES:
            self._write_json(413, {"detail": "Request body exceeds the 1 MB bridge limit."})
            return
        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise BridgeRequestError(400, "Request body must be a JSON object.")
            envelope = evaluate_bridge_request(payload)
            execution = envelope.pop("_execution")
            candidate_id = envelope.pop("_execution_candidate_id")
            result = asyncio.run(
                run_live_session(
                    turns=execution["turns"],
                    candidate_id=candidate_id,
                    operation=execution["operation"],
                    agent_version=envelope["agent_version"],
                )
            )
            envelope.update(result)
            self._write_json(200, envelope)
        except BridgeRequestError as exc:
            self._write_json(exc.status_code, {"detail": exc.detail})
        except json.JSONDecodeError:
            self._write_json(400, {"detail": "Request body must be valid JSON."})
        except Exception:
            logger.exception("LiveKit bridge evaluation failed.")
            self._write_json(502, {"detail": "LiveKit AgentSession evaluation failed; no result was accepted."})


class ThreadingPixiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    host = os.getenv("VOICE_BRIDGE_HOST", "127.0.0.1")
    try:
        is_loopback = host.casefold() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise SystemExit("The Pixi bridge only binds to localhost/loopback; do not expose it publicly.")
    token = os.getenv("VOICE_BRIDGE_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("Set VOICE_BRIDGE_TOKEN to a random secret of at least 32 characters in .env.local.")
    version = os.getenv("VOICE_AGENT_VERSION", "").strip()
    if not version:
        raise SystemExit("Set VOICE_AGENT_VERSION to an immutable commit SHA or image/config digest.")
    port = int(os.getenv("VOICE_BRIDGE_PORT", "8787"))
    if not 1024 <= port <= 65535:
        raise SystemExit("VOICE_BRIDGE_PORT must be between 1024 and 65535.")
    server = ThreadingPixiServer((host, port), PixiBridgeHandler)
    logger.info("Pixi bridge listening only on %s:%s (protocol v2, text-input mode).", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping Pixi bridge.")
    finally:
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    main()
