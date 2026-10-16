
# Voice Reliability Lab

**A benchmark and observability harness for real-time voice AI agents.**

Voice agents fail in ways that text evals never see: the model answers correctly but takes four seconds to start speaking, the user interrupts and the agent talks over them, a tool times out mid-sentence. Voice Reliability Lab runs a real [LiveKit Agents](https://docs.livekit.io/agents/) voice agent, records every turn as a structured trace, and checks those traces against scenario specs, so reliability becomes something you can measure instead of something you judge by ear.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![LiveKit Agents](https://img.shields.io/badge/LiveKit-Agents-black) ![Status](https://img.shields.io/badge/status-active%20development-orange)

---

## Status

This project is under active development and is being grown into a general voice AI benchmark harness. This table says plainly what works today and what does not.

| Area | State |
|---|---|
| Voice agent (STT, LLM, TTS, VAD, turn detection, noise enhancement) | Working |
| Per-turn latency tracing: LLM time-to-first-token, TTS time-to-first-byte, end-to-end | Working (measured, see below) |
| Per-turn latency tracing: STT processing time, end-of-utterance delay | Handlers implemented; fields are empty in the committed sample trace, being verified |
| Tool-call tracing (name, arguments, output, latency, success) | Working |
| Interruption tracking (6 signals per turn) | Implemented |
| Scenario specs (4) and pass/fail evaluator (3 checks) | Implemented |
| Scenario runner that drives the live agent | **Placeholder.** It currently writes `not_run` results. Connecting it is the next milestone |
| `get_country_info` tool used by 3 of the 4 scenarios | Not implemented yet |
| Fault injection (tool timeouts, slow tools) | Planned |
| Multi-run aggregation (p50 / p95), provider comparison, reports | Planned |

---

## How it works

```
 caller audio
      │
      ▼
┌──────────────────────────── LiveKit room ────────────────────────────┐
│  VAD + turn detector + ai|coustics noise enhancement                  │
│                                                                       │
│  STT  Deepgram Nova-3  ──►  LLM  Gemma 4 31B (+ tools)  ──►  TTS      │
│                                                       Cartesia Sonic-3│
└───────────────┬───────────────────────────────────────────────────────┘
                │  metrics_collected events (STT, EOU, LLM, TTS)
                ▼
        LiveTraceCollector ──►  evals/results/live_trace.json
                                        │
                                        ▼
              Evaluator  ◄──  scenario specs (evals/scenarios/*.json)
              (completed? expected tool? within latency budget?)
```

The agent is a standard LiveKit pipeline. The harness attaches to the session's metric events and turns each user turn into a trace record.

### What is measured

| Metric | Field | Meaning |
|---|---|---|
| STT processing | `stt_processing_ms` | Time the speech-to-text stage took |
| End-of-utterance delay | `eot_latency_ms` | Delay before the turn detector decided the user finished |
| LLM time to first token | `llm_ttft_ms` | How long until the model starts responding |
| TTS time to first byte | `tts_ttfb_ms` | How long until audio starts streaming back |
| End-to-end latency | `e2e_ms` | User stops speaking to agent starts speaking |
| Tool calls | `tools[]` | `name`, `arguments`, `output`, `latency_ms`, `success`, `error` |
| Interruption signals | `interruption{}` | `interrupted`, `overlapping_speech`, `false_interruption`, `recovered`, `detection_delay_ms`, `agent_speech_duration_ms` |

---

## Sample measurement

From a live session of **3 turns** (`evals/results/live_trace.json`). This is a small sample, so treat it as a smoke test, not a benchmark.

| Metric | Mean | Range |
|---|---|---|
| LLM time to first token | 1,155 ms | 1,001 – 1,262 ms |
| TTS time to first byte | 183 ms | 180 – 185 ms |
| End-to-end latency | 1,897 ms | 1,636 – 2,388 ms |

All 3 turns finished inside the 3,000 ms budget used by the scenario specs. 1 tool call was traced and succeeded. Larger runs are part of the roadmap.

---

## Quickstart

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and a [LiveKit Cloud](https://cloud.livekit.io) project.

```bash
git clone https://github.com/devxMani/voice-ai-.git
cd voice-ai-/voice-agent

uv sync

# Credentials: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
cp .env.example .env.local      # the agent loads .env.local

uv run python src/agent.py download-files   # one-time model downloads
uv run python src/agent.py console          # talk to the agent in your terminal
uv run python src/agent.py dev              # run as a LiveKit worker
```

When a session ends, the collector writes the trace to `evals/results/live_trace.json`.

### Docker

A `Dockerfile` (Python 3.13, uv) is included for deployment.

---

## Scenarios and evaluation

A scenario is a JSON file in `evals/scenarios/`:

```json
{
  "name": "Tokyo time lookup",
  "user_input": "What time is it in Tokyo?",
  "expected_tool": "get_time",
  "expected_behavior": "Agent should call get_time with Tokyo and provide the current Tokyo time.",
  "max_latency_ms": 3000
}
```

`evals/evaluator.py` checks a trace against its scenario on three points:

1. **Completed:** the turn finished with status `completed`.
2. **Expected tool:** the agent called the tool the scenario expects.
3. **Latency budget:** the turn stayed within `max_latency_ms`.

Current scenarios: Japan country lookup, Tokyo time lookup, tool timeout, unknown country.

---

## Roadmap

The goal is a proper, provider-agnostic **voice AI benchmark harness**.

**1. Make the runner real**
- [ ] Drive the live agent from `evals/runner.py` with scripted caller turns and record real results
- [ ] Implement `get_country_info` so all four scenarios can execute
- [ ] Add fault injection: tool timeouts, slow tools, malformed tool output
- [ ] Merge the duplicate trace-collector code paths in `evals/live_trace.py`

**2. Richer measurements**
- [ ] Run N times per scenario and report p50 / p95 instead of single values
- [ ] Turn-taking and barge-in benchmarks: detection delay, recovery, false-interruption rate
- [ ] ASR accuracy: word error rate against reference transcripts
- [ ] TTS evaluation: time to first byte, streaming stability, and intelligibility checked by ASR round-trip
- [ ] Long-horizon conversations: multi-turn task completion and context retention over extended calls

**3. Benchmark harness**
- [ ] Swap STT, LLM, and TTS providers behind one config and compare them on identical scenarios
- [ ] Cost per call-minute alongside latency
- [ ] Regression suite in CI that fails when latency or pass rate degrades
- [ ] Shareable HTML report and a public scenario set

---

## Repository layout

```
voice-agent/
├── src/
│   ├── agent.py            # LiveKit voice agent, tools, session setup
│   └── observability.py    # logging and metric helpers
├── evals/
│   ├── scenarios/          # scenario specs (JSON)
│   ├── runner.py           # scenario runner (placeholder, being connected)
│   ├── evaluator.py        # pass/fail checks against a scenario
│   ├── live_trace.py       # live trace collector attached to the session
│   ├── trace.py            # trace data models (TurnTrace, ToolTrace, InterruptionTrace)
│   └── results/            # latest.json, live_trace.json
├── Dockerfile
└── pyproject.toml
```

---

## Built with

[LiveKit Agents](https://github.com/livekit/agents) (Python), [Deepgram](https://deepgram.com) Nova-3, [Cartesia](https://cartesia.ai) Sonic-3, Google Gemma 4 31B through LiveKit Inference, and [ai|coustics](https://ai-coustics.com) audio enhancement. The agent started from LiveKit's single-prompt agent starter.

## License

Add a license before accepting contributions (MIT is a common choice).
