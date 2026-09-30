import logging

from livekit.agents.metrics import (
    LLMMetrics,
    TTSMetrics,
    STTMetrics,
)

logger = logging.getLogger("voice-observability")


def log_stt_metrics(metrics: STTMetrics):
    logger.info(
        "[STT] duration=%.3fs audio_duration=%.3fs",
        metrics.duration,
        metrics.audio_duration,
    )


def log_llm_metrics(metrics: LLMMetrics):
    logger.info(
        "[LLM] ttft=%.3fs duration=%.3fs prompt_tokens=%s completion_tokens=%s",
        metrics.ttft,
        metrics.duration,
        metrics.prompt_tokens,
        metrics.completion_tokens,
    )


def log_tts_metrics(metrics: TTSMetrics):
    logger.info(
        "[TTS] ttfb=%.3fs duration=%.3fs audio_duration=%.3fs",
        metrics.ttfb,
        metrics.duration,
        metrics.audio_duration,
    )
    