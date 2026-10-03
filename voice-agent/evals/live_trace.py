import json
import time
from datetime import datetime
from pathlib import Path
import json
import time
from datetime import datetime
from pathlib import Path

from evals.trace import (
    VoiceTrace,
    TurnTrace,
    ToolTrace,
)

RESULTS_DIR = Path(__file__).parent / "results"


class LiveTraceCollector:
    def __init__(self):
        self.trace = VoiceTrace()

        self.current_turn = None

        self.trace_started_at = datetime.now().isoformat()

        self.active_tools = {}

        self.turn_started_at = None

        # Tracks when the agent started speaking.
        self.agent_speaking_started_at = None

        # Tracks the most recent interruption.
        self.active_interruption = None

    def attach(self, session):
        session.on(
            "user_input_transcribed",
            self.on_user_input_transcribed,
        )

        session.on(
            "conversation_item_added",
            self.on_conversation_item_added,
        )

        session.on(
            "agent_state_changed",
            self.on_agent_state_changed,
        )

        session.on(
            "user_state_changed",
            self.on_user_state_changed,
        )

        session.on(
            "function_tools_executed",
            self.on_function_tools_executed,
        )

        session.on(
            "overlapping_speech",
            self.on_overlapping_speech,
        )

        session.on(
            "agent_false_interruption",
            self.on_agent_false_interruption,
        )

        session.on(
            "speech_created",
            self.on_speech_created,
        )

        session.on(
            "close",
            self.on_close,
        )

        print("[TRACE] collector attached")

    # ---------------------------------------------------------
    # COMPONENT METRICS
    # ---------------------------------------------------------

    def record_stt_metrics(self, metrics):
        if self.current_turn is None:
            return

        self.current_turn.stt_processing_ms = (
            metrics.duration * 1000
        )

        print(
            "[TRACE] "
            f"STT processing="
            f"{self.current_turn.stt_processing_ms:.2f}ms"
        )

    def record_llm_metrics(self, metrics):
        if self.current_turn is None:
            return

        ttft_ms = metrics.ttft * 1000

        # Preserve the latest TTFT as the current canonical value.
        self.current_turn.llm_ttft_ms = ttft_ms

        print(
            "[TRACE] "
            f"LLM TTFT={ttft_ms:.2f}ms"
        )

    def record_tts_metrics(self, metrics):
        if self.current_turn is None:
            return

        ttfb_ms = metrics.ttfb * 1000

        self.current_turn.tts_ttfb_ms = ttfb_ms

        print(
            "[TRACE] "
            f"TTS TTFB={ttfb_ms:.2f}ms"
        )

    # ---------------------------------------------------------
    # SESSION
    # ---------------------------------------------------------

    def on_close(self, event):
        print("[TRACE] session closed — saving trace")
        self.save()

    # ---------------------------------------------------------
    # USER TRANSCRIPTION
    # ---------------------------------------------------------

    def on_user_input_transcribed(self, event):
        print(
            "[TRACE] "
            f"transcript final={event.is_final} "
            f"text={event.transcript!r}"
        )

        if not event.is_final:
            return

        self.current_turn = TurnTrace(
            user_input=event.transcript
        )

        self.trace.turns.append(
            self.current_turn
        )

        self.turn_started_at = time.perf_counter()

        print(
            "[TRACE] "
            f"new turn user={event.transcript!r}"
        )

        # If the previous agent speech was interrupted,
        # this new user turn represents recovery.
        if self.active_interruption is not None:
            self.active_interruption.recovered = True

            print(
                "[TRACE] "
                "interruption recovered by new user turn"
            )

            self.active_interruption = None

    # ---------------------------------------------------------
    # CONVERSATION ITEMS
    # ---------------------------------------------------------

    def on_conversation_item_added(self, event):
        item = event.item

        print(
            "[TRACE] "
            f"conversation item type="
            f"{type(item).__name__}"
        )

        if not hasattr(item, "role"):
            return

        print(
            "[TRACE] "
            f"role={item.role}"
        )

        if str(item.role).lower() != "assistant":
            return

        content = getattr(
            item,
            "content",
            None,
        )

        if not content:
            return

        texts = []

        for part in content:
            if hasattr(part, "text"):
                texts.append(part.text)

            elif isinstance(part, str):
                texts.append(part)

        if not texts:
            return

        if self.current_turn is None:
            return

        self.current_turn.assistant_output = (
            " ".join(texts)
        )

        print(
            "[TRACE] "
            f"assistant="
            f"{self.current_turn.assistant_output!r}"
        )

    # ---------------------------------------------------------
    # AGENT STATE
    # ---------------------------------------------------------

    def on_agent_state_changed(self, event):
        print(
            "[TRACE] "
            f"agent state "
            f"{event.old_state} -> "
            f"{event.new_state}"
        )

        if self.current_turn is None:
            return

        new_state = str(
            event.new_state
        ).lower()

        old_state = str(
            event.old_state
        ).lower()

        # Agent starts speaking.
        if new_state == "speaking":
            self.agent_speaking_started_at = (
                time.perf_counter()
            )

            if self.turn_started_at is not None:
                self.current_turn.e2e_latency_ms = (
                    time.perf_counter()
                    - self.turn_started_at
                ) * 1000

                print(
                    "[TRACE] "
                    f"E2E latency="
                    f"{self.current_turn.e2e_latency_ms:.2f}ms"
                )

        # Agent stops speaking.
        if (
            old_state == "speaking"
            and new_state != "speaking"
        ):
            if self.agent_speaking_started_at is not None:
                duration_ms = (
                    time.perf_counter()
                    - self.agent_speaking_started_at
                ) * 1000

                print(
                    "[TRACE] "
                    f"agent speech duration="
                    f"{duration_ms:.2f}ms"
                )

                if (
                    self.current_turn is not None
                    and self.current_turn.interruption
                    .interrupted
                ):
                    self.current_turn.interruption.\
                        agent_speech_duration_ms = (
                            duration_ms
                        )

            self.agent_speaking_started_at = None

    # ---------------------------------------------------------
    # USER STATE
    # ---------------------------------------------------------

    def on_user_state_changed(self, event):
        print(
            "[TRACE] "
            f"user state "
            f"{event.old_state} -> "
            f"{event.new_state}"
        )

    # ---------------------------------------------------------
    # SPEECH CREATED
    # ---------------------------------------------------------

    def on_speech_created(self, event):
        print(
            "[TRACE] "
            f"speech created "
            f"user_initiated={event.user_initiated} "
            f"source={event.source}"
        )

    # ---------------------------------------------------------
    # INTERRUPTION
    # ---------------------------------------------------------

    def on_overlapping_speech(self, event):
        print(
            "[TRACE] "
            "overlapping speech detected "
            f"is_interruption={event.is_interruption} "
            f"agent_ended={event.agent_ended} "
            f"detection_delay="
            f"{event.detection_delay * 1000:.2f}ms "
            f"probability={event.probability:.3f}"
        )

        if self.current_turn is None:
            return

        interruption = self.current_turn.interruption

        interruption.overlapping_speech = True

        if event.is_interruption:
            interruption.interrupted = True

            self.current_turn.interrupted = True

            interruption.detection_delay_ms = (
                event.detection_delay * 1000
            )

            self.active_interruption = interruption

            print(
                "[TRACE] "
                f"REAL INTERRUPTION "
                f"detection_delay="
                f"{interruption.detection_delay_ms:.2f}ms"
            )

        else:
            print(
                "[TRACE] "
                "overlap detected but not classified "
                "as interruption"
            )

    # ---------------------------------------------------------
    # FALSE INTERRUPTION
    # ---------------------------------------------------------

    def on_agent_false_interruption(self, event):
        print(
            "[TRACE] "
            f"false interruption "
            f"resumed={event.resumed}"
        )

        if self.current_turn is None:
            return

        interruption = self.current_turn.interruption

        interruption.false_interruption = True

        if event.resumed:
            interruption.recovered = True

        print(
            "[TRACE] "
            "false interruption recorded"
        )

    # ---------------------------------------------------------
    # TOOLS
    # ---------------------------------------------------------

    def on_function_tools_executed(self, event):
        print(
            "[TRACE] "
            f"function tools executed: "
            f"{len(event.function_calls)}"
        )

        for function_call, function_output in event.zipped():

            arguments = function_call.arguments

            try:
                arguments = json.loads(
                    arguments
                )

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                arguments = {
                    "raw": arguments
                }

            tool_trace = ToolTrace(
                name=function_call.name,
                arguments=arguments,
                output=function_output.output,
                success=not function_output.is_error,
                error=(
                    function_output.output
                    if function_output.is_error
                    else None
                ),
            )

            if self.current_turn is not None:
                self.current_turn.tools.append(
                    tool_trace
                )

            print(
                "[TRACE] "
                f"tool completed "
                f"name={function_call.name} "
                f"call_id={function_call.call_id} "
                f"success={not function_output.is_error} "
                f"output={function_output.output!r}"
            )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    def save(self):
        self.trace.status = "completed"

        RESULTS_DIR.mkdir(
            exist_ok=True
        )

        output = (
            RESULTS_DIR
            / "live_trace.json"
        )

        data = self.trace.to_dict()

        data[
            "trace_started_at"
        ] = self.trace_started_at

        with open(
            output,
            "w",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
            )

        print(
            f"[TRACE] saved → {output}"
        )

        return output
from evals.trace import VoiceTrace, TurnTrace, ToolTrace


RESULTS_DIR = Path(__file__).parent / "results"


class LiveTraceCollector:
    def __init__(self):
        self.trace = VoiceTrace()

        # Current conversation turn
        self.current_turn = None

        self.trace_started_at = datetime.now().isoformat()

        # call_id -> ToolTrace
        self.active_tools = {}

        # Used for measuring turn latency
        self.turn_started_at = None

    # ========================================================
    # ATTACH EVENTS
    # ========================================================

    def attach(self, session):
        session.on(
            "user_input_transcribed",
            self.on_user_input_transcribed,
        )

        session.on(
            "conversation_item_added",
            self.on_conversation_item_added,
        )

        session.on(
            "agent_state_changed",
            self.on_agent_state_changed,
        )

        session.on(
            "user_state_changed",
            self.on_user_state_changed,
        )

        session.on(
            "function_tools_executed",
            self.on_function_tools_executed,
        )

        session.on(
            "close",
            self.on_close,
        )

        print("[TRACE] collector attached")

    # ========================================================
    # METRICS
    # ========================================================
    def record_eou_metrics(self, metrics):
        if self.current_turn is None:
            return

        self.current_turn.eot_latency_ms = (
            metrics.end_of_utterance_delay * 1000
        )

        print(
            "[TRACE] "
            f"EOT latency="
            f"{self.current_turn.eot_latency_ms:.2f}ms "
            f"transcription_delay="
            f"{metrics.transcription_delay * 1000:.2f}ms "
            f"user_turn_completed_delay="
            f"{metrics.on_user_turn_completed_delay * 1000:.2f}ms"
        )
        
    def record_stt_metrics(self, metrics):
        if self.current_turn is None:
            return

        self.current_turn.stt_latency_ms = (
            metrics.duration * 1000
        )

        print(
            f"[TRACE] STT latency="
            f"{self.current_turn.stt_latency_ms:.2f}ms"
        )

    def record_llm_metrics(self, metrics):
        if self.current_turn is None:
            return

        self.current_turn.llm_ttft_ms = (
            metrics.ttft * 1000
        )

        print(
            f"[TRACE] LLM TTFT="
            f"{self.current_turn.llm_ttft_ms:.2f}ms"
        )

    def record_tts_metrics(self, metrics):
        if self.current_turn is None:
            return

        self.current_turn.tts_ttfb_ms = (
            metrics.ttfb * 1000
        )

        print(
            f"[TRACE] TTS TTFB="
            f"{self.current_turn.tts_ttfb_ms:.2f}ms"
        )

    # ========================================================
    # SESSION CLOSE
    # ========================================================

    def on_close(self, event):
        print("[TRACE] session closed — saving trace")
        self.save()

    # ========================================================
    # USER TRANSCRIPT
    # ========================================================

    def on_user_input_transcribed(self, event):
        print(
            f"[TRACE] transcript "
            f"final={event.is_final} "
            f"text={event.transcript!r}"
        )

        if event.is_final:
            self.current_turn = TurnTrace(
                user_input=event.transcript
            )

            self.trace.turns.append(
                self.current_turn
            )

            self.turn_started_at = time.perf_counter()

            print(
                f"[TRACE] new turn "
                f"user={event.transcript!r}"
            )

    # ========================================================
    # CONVERSATION ITEMS
    # ========================================================

    def on_conversation_item_added(self, event):
        item = event.item

        print(
            f"[TRACE] conversation item "
            f"type={type(item).__name__}"
        )

        if hasattr(item, "role"):
            print(f"[TRACE] role={item.role}")

            if str(item.role).lower() == "assistant":

                content = getattr(
                    item,
                    "content",
                    None,
                )

                if content:
                    texts = []

                    for part in content:
                        if hasattr(part, "text"):
                            texts.append(part.text)

                        elif isinstance(part, str):
                            texts.append(part)

                    if (
                        texts
                        and self.current_turn is not None
                    ):
                        self.current_turn.assistant_output = (
                            " ".join(texts)
                        )

                        # Mark the turn as completed
                        self.current_turn.status = "completed"

                        print(
                            f"[TRACE] assistant="
                            f"{self.current_turn.assistant_output!r}"
                        )

    # ========================================================
    # AGENT STATE
    # ========================================================

    def on_agent_state_changed(self, event):
        print(
            f"[TRACE] agent state "
            f"{event.old_state} -> {event.new_state}"
        )

        if self.current_turn is None:
            return

        # First approximation of E2E latency:
        #
        # final user transcript
        #        ↓
        # agent starts speaking
        #
        if str(event.new_state).lower() == "speaking":

            if self.turn_started_at is not None:

                self.current_turn.e2e_latency_ms = (
                    time.perf_counter()
                    - self.turn_started_at
                ) * 1000

                print(
                    f"[TRACE] E2E latency="
                    f"{self.current_turn.e2e_latency_ms:.2f}ms"
                )

    # ========================================================
    # USER STATE
    # ========================================================

    def on_user_state_changed(self, event):
        print(
            f"[TRACE] user state "
            f"{event.old_state} -> {event.new_state}"
        )

    # ========================================================
    # TOOL EXECUTION
    # ========================================================

    def on_function_tools_executed(self, event):
        print(
            f"[TRACE] function tools executed: "
            f"{len(event.function_calls)}"
        )

        for function_call, function_output in event.zipped():

            arguments = function_call.arguments

            try:
                arguments = json.loads(arguments)

            except (
                json.JSONDecodeError,
                TypeError,
            ):
                arguments = {
                    "raw": arguments
                }

            tool_trace = ToolTrace(
                name=function_call.name,
                arguments=arguments,
                output=function_output.output,
                success=not function_output.is_error,
                error=(
                    function_output.output
                    if function_output.is_error
                    else None
                ),
            )

            if self.current_turn is not None:
                self.current_turn.tools.append(
                    tool_trace
                )

            print(
                f"[TRACE] tool completed "
                f"name={function_call.name} "
                f"call_id={function_call.call_id} "
                f"success={not function_output.is_error} "
                f"output={function_output.output!r}"
            )

    # ========================================================
    # SAVE TRACE
    # ========================================================

    def save(self):
        self.trace.status = "completed"

        RESULTS_DIR.mkdir(
            exist_ok=True
        )

        output = (
            RESULTS_DIR
            / "live_trace.json"
        )

        data = self.trace.to_dict()

        data["trace_started_at"] = (
            self.trace_started_at
        )

        with open(output, "w") as f:
            json.dump(
                data,
                f,
                indent=2,
            )

        print(
            f"[TRACE] saved → {output}"
        )

        return output
