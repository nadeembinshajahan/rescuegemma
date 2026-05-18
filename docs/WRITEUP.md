# RescueGemma — Technical Write-up

**Team:** StratoFirma Autonomy Labs (Nadeem Shajahan, Thomas Kuruvila)
**Primary Track:** Global Resilience · **Secondary:** Safety & Trust
**Special Technology Tracks targeted:** Ollama · llama.cpp · LiteRT
(one model, one Python package, three deployment surfaces — all on-device)

## The problem

When a building is on fire or has collapsed, the first question is always
the same and always human: *is there someone in there, and can we get them
out?* The people asking it — a firefighter at the door, a parent on the
lawn — do not speak in coordinates. And the environments where it matters
most have no cloud: no signal in a concrete stairwell, no time to wait on a
network.

## What we built

RescueGemma is a voice-commanded autonomous indoor **escort** agent. An
operator speaks a plain, panicked sentence — *"There's smoke downstairs, I
can't see — get Manu out, he's at the workbench."* — and the agent, running
entirely on-device, finds the trapped mobile person, then *leads him out*:
workbench → door → down the stairs. It scouts each path segment before
committing him to it, speaks to him aloud (its own words from the real
frame), and reassesses when he lags. No network. No cloud. No survivor
imagery ever leaves the device.

It chains three Gemma 4 capabilities in a loop only an LLM can close:

1. **Audio in (native).** The operator's voice → intent. No separate ASR.
2. **Function calling (native).** Typed calls (`takeoff`, `scout_ahead`,
   `speak_to_survivor`, `lead_to_exit`, `report_finding`, …) into a thin
   wrapper over PX4/MAVLink. Gemma is the planner; PX4 is the executor.
3. **Vision in.** Returned camera frames let Gemma *ground* its descriptions
   ("the doorway with boxes on the left") and choose its words for the
   person — the core safety property of an escort agent.

## One model: Gemma 4 E4B

The agent's entire intelligence is **Gemma 4 E4B** — the 4 B-effective
edge variant with **native audio**, **native vision**, and **native tool
calling** in a single checkpoint. One model. One forward path per turn.
No external ASR pipeline, no separate VLM, no caption model, no glue code
that fails first under field conditions.

E4B was designed for on-device deployment:

| Capability             | How                                            |
|------------------------|------------------------------------------------|
| Audio in               | Native encoder, 50% smaller than Gemma 3n's, 40 ms frames |
| Vision in              | Built-in tower, ingests inline JPEGs           |
| Function calling       | Native tool slot                               |
| Disk / memory (Q4)     | ~9.6 GB — fits on a Mac, fits on a Jetson Orin |
| Throughput on M-series | Real-time end-to-end for our 14-step mission   |

We deliberately chose **E4B over heavier Gemma 4 variants** (26B-A4B MoE,
31B dense) for this submission. E4B handles the agent's full workload
end-to-end on a Mac, demonstrates the on-device claim honestly, and is
the same checkpoint that ships to a Jetson Orin Nano via llama.cpp. The
larger variants are available as drop-in `--reasoning-model` overrides
for ground-station deployments.

## Architecture

```
operator voice
      │
      ▼
[Gemma 4 E4B · audio in] ──► transcript + intent ──┐
                                                   │
            ┌────── closed escort loop ────────────▼─┐
            ▼                                         │
   [Gemma 4 E4B · vision in + plan + ONE tool call]   │
            │                                         │
            ▼                                         │
       executor (sim or real PX4 flight bridge)       │
            │                                         │
            ▼                                         │
   observation + REAL camera frame ──────► next turn ─┘
            │
            ▼
   speak_to_survivor / report_finding / return_to_operator
```

The **executor** is swappable behind one contract (`schemas.ToolResult`):

- `EscortExecutor` — serves real frames from a phone walkthrough, tracks
  the person's follow state per segment, refuses `lead_to_exit` on any
  segment that has not been `scout_ahead`-ed first.
- `SimExecutor` — the original search scenario.
- `FlightBridge` (Thomas's MAVLink/PX4 wrapper) — same contract on the
  Jetson Orin Nano.

Because both satisfy the identical schema, **the planner code path in the
demo is byte-for-byte the one that runs on the aircraft.** Only flight
dynamics and the person's follow signal are simulated; perception and
reasoning are real.

## Deployment surfaces — same code, swap the host (Special Tech tracks)

The agent is one Python package; the model host is a swappable backend.
Same prompts, same TOOLS schema, same loop:

| Surface                | Backend                | Track            |
|------------------------|------------------------|------------------|
| Mac dev / demo         | `OllamaBackend` (Gemma 4 E4B via Ollama) | **Ollama** |
| Jetson Orin Nano edge  | `OllamaBackend` pointed at Ollama-on-llama.cpp on the aircraft | **llama.cpp** |
| Operator phone app     | Same TOOLS schema + prompts on a Google AI Edge **LiteRT** Gemma 4 build | **LiteRT** |
| Smoke / CPU CI         | `EscortScriptedBackend` (no model, no network, every line `[SCRIPTED]`) | n/a |

- **Ollama track:** the demo runs the entire mission via local `ollama`
  with `gemma4:e4b`. No network. No API key. `keep_alive=30m` so weights
  stay resident between turns on Apple Silicon Metal.
- **llama.cpp track:** Ollama is a llama.cpp wrapper. The same Python
  agent, the same `OllamaBackend`, just `OLLAMA_HOST` pointed at the
  Jetson Orin Nano running Ollama-on-llama.cpp. Resource-constrained
  edge, same agent.
- **LiteRT track:** the operator's mobile app — which surfaces the
  spoken lines and findings live — runs Gemma 4 on Google AI Edge's
  LiteRT runtime. Same TOOLS schema, same prompts, only the transport
  changes.

## Why Gemma 4 specifically

- **On-device / offline.** Demonstrated with the network physically
  unplugged. In a collapsed building there is no cloud — and survivor
  imagery must never leave the device. Privacy-preserving by construction.
- **Native multimodality + function calling in one checkpoint** removes
  the brittle ASR→LLM→planner glue that fails first under field
  conditions. One model, one failure mode to harden.
- **Apache 2.0** lets this ship in a real commercial SAR product.

## Mac efficiency notes

`OllamaBackend` is tuned for Apple Silicon out of the box:

- `keep_alive="30m"` so the model stays resident across agent turns —
  otherwise every other turn pays a weight-reload tax.
- `num_ctx=4096` — the agent loop is bounded; bigger context only
  inflates KV cache for no benefit.
- `num_gpu=-1` — push every layer Ollama can to Metal.
- One model in unified memory at a time. Nothing else hot.

In practice on an M-series Mac with `gemma4:e4b` Q4 the full 14-step
escort transcript finishes in well under a minute, end-to-end.

## Honesty about what is real

The Gemma 4 reasoning, the audio transcription, the vision grounding,
and the function calling are all real and all done by the same single
Gemma 4 E4B checkpoint. The camera frames are real footage from a phone
walkthrough. Only the airframe and the person's follow signal are
simulated for reproducibility. The scripted backend is included
**solely** as a no-GPU smoke test of the loop and is explicitly labelled
`[SCRIPTED]` on every line — never presented as model output.

## Failure modes & safety

- Lead-into-uninspected-space is a hard refusal in the executor: the
  planner *must* `scout_ahead` before `lead_to_exit` on any segment.
- When the person lags, the planner is required by the system prompt to
  hold, re-speak (reassuring, slower) and re-attempt — not blind retry.
- Ambiguous unsafe state → conservative `abort_and_hold`.
- Malformed model call → structural validation returns a correctable
  error to Gemma rather than crashing.
- If the local Ollama build does not yet support audio-in on Gemma 4
  E4B, the backend logs that *loudly* and proceeds with a clearly-
  labelled placeholder intent. We never silently fake the transcript.
- The model never receives survivor ground truth; it must earn the
  determination from the frame.

## Ethics & dual-use

This is a SAR system. The same autonomy has obvious dual-use; we state
that plainly. Our design choices — operator-in-the-loop reporting (the
drone informs, the human decides), on-device data that never leaves the
aircraft, and conservative abort defaults — are deliberate constraints
toward the civil-protection use case.
