from dataclasses import dataclass, field


@dataclass
class ToolTrace:
    name: str
    arguments: dict = field(default_factory=dict)
    output: str | None = None
    latency_ms: float | None = None
    success: bool = True
    error: str | None = None


@dataclass
class InterruptionTrace:
    interrupted: bool = False
    overlapping_speech: bool = False
    false_interruption: bool = False

    detection_delay_ms: float | None = None
    agent_speech_duration_ms: float | None = None

    recovered: bool = False

    def to_dict(self):
        return {
            "interrupted": self.interrupted,
            "overlapping_speech": self.overlapping_speech,
            "false_interruption": self.false_interruption,
            "detection_delay_ms": self.detection_delay_ms,
            "agent_speech_duration_ms": self.agent_speech_duration_ms,
            "recovered": self.recovered,
        }


@dataclass
class TurnTrace:
    user_input: str = ""
    assistant_output: str = ""

    tools: list[ToolTrace] = field(default_factory=list)

    stt_processing_ms: float | None = None
    eot_latency_ms: float | None = None
    llm_ttft_ms: float | None = None
    tts_ttfb_ms: float | None = None
    e2e_latency_ms: float | None = None

    interruption: InterruptionTrace = field(
        default_factory=InterruptionTrace
    )

    interrupted: bool = False

    status: str = "running"
    error: str | None = None

    def to_dict(self):
        return {
            "user_input": self.user_input,
            "assistant_output": self.assistant_output,

            "tools": [
                {
                    "name": tool.name,
                    "arguments": tool.arguments,
                    "output": tool.output,
                    "latency_ms": tool.latency_ms,
                    "success": tool.success,
                    "error": tool.error,
                }
                for tool in self.tools
            ],

            "latency": {
                "stt_processing_ms": self.stt_processing_ms,
                "eot_latency_ms": self.eot_latency_ms,
                "llm_ttft_ms": self.llm_ttft_ms,
                "tts_ttfb_ms": self.tts_ttfb_ms,
                "e2e_ms": self.e2e_latency_ms,
            },

            "interruption": self.interruption.to_dict(),

            "interrupted": self.interrupted,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class VoiceTrace:
    turns: list[TurnTrace] = field(default_factory=list)

    status: str = "running"
    error: str | None = None

    def to_dict(self):
        return {
            "turns": [
                turn.to_dict()
                for turn in self.turns
            ],
            "status": self.status,
            "error": self.error,
        }