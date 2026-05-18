# RescueGemma

**Voice-commanded, on-device indoor rescue agent — Gemma 4 hears the
operator, sees the building through a live phone camera, and leads a
trapped survivor out, step by step.**

---

## TL;DR

A stressed bystander says into their phone: *"There's smoke downstairs,
get Manu out — he's upstairs at the workbench."* On the same Mac, with
no network, a simulated drone (the phone's camera) takes off, climbs the
stairs, recognises the trapped person by sight, then *talks him out* in
a natural neural voice — narrating each step, scouting each segment
before committing him to it, and reassuring him when he hesitates. The
whole loop runs locally; the only "simulated" thing is flight dynamics.

## Why this matters

When a building is on fire or has collapsed, the first question is always
human: *is there someone in there, and can we get them out?* The people
asking it don't speak in coordinates, and the environments where it
matters most have no network. RescueGemma collapses three field-brittle
glue layers — ASR → planner → VLM — into a **single on-device Gemma 4
forward pass per turn**, with vision grounding from the operator's own
phone camera and a natural spoken response back to the survivor.

## What's real, what's simulated

| Real | Simulated |
|---|---|
| Operator's voice → Gemma 4 audio encoder → text intent | Flight dynamics |
| Per-turn JPEG from the operator's phone → Gemma 4 vision encoder | Path-segment graph |
| Native function calling, parsed via Gemma 4's `<\|tool_call\|>` markers | The survivor's follow signal |
| Kokoro 82M neural TTS for voice out (system → Manu) | |
| Auto-controlled phone LED based on frame luminance | |
| Closed agent loop, pure Python, no cloud | |

## Architecture

Two-phase mission, enforced by a deterministic executor state machine:

**PHASE 1 — FIND** (drone alone): repeated `scout_ahead` calls advance
the drone through the building. The moment Gemma sees a human silhouette
in the frame, it calls `confirm_survivor_located(description,
confidence)` — that vision-asserted call flips the mission state to
phase 2 immediately, no facial recognition or beard-check required.

**PHASE 2 — ESCORT** (Manu in tow): strict per-segment loop —
`scout_ahead → speak_to_survivor → lead_to_exit` — repeated for 5 path
segments. One segment is flagged "lags" to force a reassurance beat
(Gemma must `speak_to_survivor(tone=reassuring)` then re-attempt the
lead). At the end, `report_finding(type=egress_reached)` then
`return_to_operator`.

```
operator voice (push-to-talk SPACE in browser)
       │
       │  WebM/Opus → ffmpeg → 16 kHz mono wav
       ▼
[ Gemma 4 E2B · native audio encoder → text intent ]
       │
       ▼  per turn:
       ┌──────────────────────────────────────────────────────┐
       │  pull /shot.jpg from phone (the simulated drone)     │
       │  send [image, transcript-so-far, ask-next-action]    │
       │  parse <|tool_call>call:NAME{...}<tool_call|>         │
       │  EscortExecutor.<tool>(...) updates state machine    │
       │  Kokoro speaks if speak_to_survivor                  │
       │  publish to web UI (SSE)                             │
       └──────────────────────────────────────────────────────┘
       │
       ▼
   report_finding(egress_reached) → return_to_operator
```

## The runtime stack

We use **Google AI Edge's [LiteRT-LM](https://github.com/litert-community)**
runtime — the only local stack as of mid-2026 that exposes Gemma 4's
**audio + vision encoders end-to-end through a Python API**. Ollama and
LM Studio host the same GGUFs but don't forward audio inputs to the model
in their chat APIs. LiteRT-LM does.

We run **Gemma 4 E2B-it** (`litert-community/gemma-4-E2B-it-litert-lm`,
~2.6 GB) on Apple Silicon — Metal-accelerated for text + vision, CPU
for audio. Context bumped to 16 K tokens via `max_num_tokens=16384` so
a full ~25-step mission fits comfortably.

**TTS** is [Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M) via
[mlx-audio](https://github.com/Blaizzy/mlx-audio) — neural, naturalistic,
real-time on M-series. macOS `say` is a fallback.

**Camera** is any Android phone running
[IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam),
served over MJPEG. The phone is literally the drone; the operator walks
it through the building.

## Special Technology tracks targeted

The same Python package, with a swappable backend, deploys to:

| Track | Backend | How |
|---|---|---|
| **LiteRT** (primary) | `LiteRTBackend` | Gemma 4 E2B via Google AI Edge LiteRT-LM on Apple Silicon. Native audio + vision + tools. |
| **llama.cpp** | `OllamaBackend` | Same Python agent, pointed at Ollama-on-llama.cpp on a Jetson Orin Nano (llama.cpp build b8766+ has Gemma 4 audio Conformer support). Resource-constrained edge. |
| **Ollama** | `OllamaBackend` | Single Gemma 4 E4B via local `ollama serve` on Mac. |

## What we built (engineering highlights)

- **Single source of truth for tools** in `schemas.py` — the executor and
  the planner both consume the same JSON schema, so the demo's planner
  code path is byte-for-byte the one that would run on a real Jetson +
  PX4 aircraft. Only the executor body is swapped.
- **State-machine executor** that *refuses* out-of-order calls (speak
  before scout, lead before scout, duplicate scout) with a directive
  pointer at the exact next call the model should make. The mission
  cannot get into incoherent states.
- **Stuck-detector with deterministic override**: after 3 consecutive
  rejected calls (any kind), the backend asks the executor "what call
  *should* happen next?" and force-runs that. The mission progresses
  through the model's confusion.
- **Live web UI** (`webapp.py`): stdlib-only HTTP server, Server-Sent
  Events for the reasoning timeline, MJPEG passthrough for the camera,
  push-to-talk on SPACE, IP-Webcam LED control proxy, auto-torch
  controller polling `/shot.jpg` and toggling the phone LED from
  luminance. All in one file, no frameworks.
- **Per-turn vision**: every Gemma turn opens a fresh conversation
  seeded with the mission transcript + a freshly-pulled phone JPEG, so
  the model literally sees what the operator is pointing at *right now*
  for every decision.
- **Honest fallback story** when something on-device isn't ready yet:
  Ollama / LM Studio don't forward audio? We say so in the WRITEUP and
  route around them via LiteRT-LM. Free IP Webcam doesn't expose
  multi-camera selection? We brute-force the likely keys and log a
  clear "switch it manually in the app" if none accepted.

## Honest engineering caveats

This is research-grade, built in a hackathon timeframe. Things still rough:

- Gemma 4 E2B occasionally produces malformed tool calls — emits `scout`
  instead of `scout_ahead`, or wraps strings in nested `<\|"\|>'...'<\|"\|>`.
  Our parser is forgiving and the executor REJECTs are sharp; the
  larger E4B variant reduces these to near zero.
- Per-turn frame ingestion is real, but tool *responses* are text-only
  (LiteRT-LM doesn't accept multimodal tool replies yet), so the model
  re-grounds on a fresh frame next user turn instead of mid-tool-call.

The full source, run instructions, and the bundled demo runner are in
the public repository.

## License

Apache-2.0. The whole stack — model, runtime, TTS — is open and
self-hostable.
