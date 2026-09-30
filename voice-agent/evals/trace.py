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
class VoiceTrace:
    user_input: str = ""
    assistant_output: str = ""

    tools: list[ToolTrace] = field(default_factory=list)

    stt_latency_ms: float | None = None
    llm_ttft_ms: float | None = None
    tts_ttfb_ms: float | None = None
    e2e_latency_ms: float | None = None

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
                "stt_ms": self.stt_latency_ms,
                "llm_ttft_ms": self.llm_ttft_ms,
                "tts_ttfb_ms": self.tts_ttfb_ms,
                "e2e_ms": self.e2e_latency_ms,
            },
            "interrupted": self.interrupted,
            "status": self.status,
            "error": self.error,
        }