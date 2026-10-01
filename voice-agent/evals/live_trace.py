import json
import time
from datetime import datetime
from pathlib import Path

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

        # Useful for measuring the whole turn later
        self.turn_started_at = None

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

    def on_close(self, event):
        print("[TRACE] session closed — saving trace")
        self.save()

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

            self.trace.turns.append(self.current_turn)

            self.turn_started_at = time.perf_counter()

            print(
                f"[TRACE] new turn "
                f"user={event.transcript!r}"
            )

    def on_conversation_item_added(self, event):
        item = event.item

        print(
            f"[TRACE] conversation item "
            f"type={type(item).__name__}"
        )

        if hasattr(item, "role"):
            print(f"[TRACE] role={item.role}")

            if str(item.role).lower() == "assistant":
                content = getattr(item, "content", None)

                if content:
                    texts = []

                    for part in content:
                        if hasattr(part, "text"):
                            texts.append(part.text)

                        elif isinstance(part, str):
                            texts.append(part)

                    if texts and self.current_turn is not None:
                        self.current_turn.assistant_output = " ".join(texts)

                        print(
                            f"[TRACE] assistant="
                            f"{self.current_turn.assistant_output!r}"
                        )

    def on_agent_state_changed(self, event):
        print(
            f"[TRACE] agent state "
            f"{event.old_state} -> {event.new_state}"
        )

    def on_user_state_changed(self, event):
        print(
            f"[TRACE] user state "
            f"{event.old_state} -> {event.new_state}"
        )

    def on_function_tools_executed(self, event):
        print(
            f"[TRACE] function tools executed: "
            f"{len(event.function_calls)}"
        )

        for function_call, function_output in event.zipped():
            arguments = function_call.arguments

            try:
                arguments = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {"raw": arguments}

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
                self.current_turn.tools.append(tool_trace)

            print(
                f"[TRACE] tool completed "
                f"name={function_call.name} "
                f"call_id={function_call.call_id} "
                f"success={not function_output.is_error} "
                f"output={function_output.output!r}"
            )

    def save(self):
        self.trace.status = "completed"

        RESULTS_DIR.mkdir(exist_ok=True)

        output = RESULTS_DIR / "live_trace.json"

        data = self.trace.to_dict()

        data["trace_started_at"] = self.trace_started_at

        with open(output, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[TRACE] saved → {output}")

        return output