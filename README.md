# RescueGemma 🚁

**Voice-commanded, on-device indoor rescue agent powered by Gemma 4.**
*Kaggle Gemma 4 Good Hackathon · StratoFirma Autonomy Labs · Apache-2.0*

An operator speaks a plain, panicked sentence. RescueGemma — running fully
on-device through **Google AI Edge's LiteRT-LM runtime** — hears it,
navigates a simulated drone upstairs to find a trapped survivor (Manu),
then leads him out segment by segment: scouting each path, speaking aloud
to him (Kokoro neural TTS), and reassessing when he lags.

The phone is the drone's camera. As you walk it around the building, Gemma
4 sees what the phone sees on **every turn** and grounds its decisions
in the current frame.

---

## What's real

| Component | On-device |
|---|---|
| Gemma 4 E2B native audio (operator's voice) | ✅ LiteRT-LM |
| Gemma 4 E2B vision (per-turn camera frame) | ✅ LiteRT-LM |
| Gemma 4 E2B tool calling (typed function calls) | ✅ LiteRT-LM (text-protocol) |
| Kokoro 82M neural TTS (system → Manu) | ✅ mlx-audio on Apple Silicon |
| Auto-LED from frame luminance | ✅ PIL + IP Webcam control API |
| MJPEG video from phone | ✅ IP Webcam (Android) |
| Closed agent loop | ✅ Pure Python, no network |

---

## Architecture

```
operator's voice (mic in browser, push-to-talk on SPACE)
       │
       ▼  WebM/Opus → wav (ffmpeg) → audio Conformer encoder
[ Gemma 4 E2B · audio in → text intent ]
       │
       ▼  text intent + sys prompt + tool list
       ┌── for each turn ────────────────────────────────────┐
       │   pull /shot.jpg from the phone camera              │
       │   send [image, transcript-so-far, ask-next-action]  │
       │   parse <|tool_call>call:NAME{...}<tool_call|>       │
       │   run EscortExecutor.<tool>(...)                    │
       │   speak via Kokoro if speak_to_survivor             │
       │   publish turn to web UI via SSE                    │
       └─────────────────────────────────────────────────────┘
       │
       ▼
       escort complete → return_to_operator
```

Two phases, enforced by the executor's state machine:

1. **FIND.** Drone takes off next to the operator. Manu is reported
   upstairs. Each `scout_ahead` advances the drone one step closer.
   When Gemma sees a human in frame it calls
   `confirm_survivor_located(description, confidence)`, flipping to phase 2.
2. **ESCORT.** Strict per-segment loop:
   `scout_ahead → speak_to_survivor → lead_to_exit`, with a lag-recovery
   beat (executor force-flags one segment as "lags" to demonstrate
   re-speak + re-attempt). 5 escort segments, executor refuses out-of-order
   calls.

---

## Quick start

### 0 · Install

You need Python 3.12+. The easiest path is to reuse the venv from
[parlor](https://github.com/fikrikarim/parlor), which already has the
right dependencies (`litert-lm`, `mlx-audio`, `numpy 2.4`, `Pillow`,
`soundfile`):

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor/src && uv sync
# then run rescuegemma's run_demo.py using parlor's venv python
```

Or set up your own venv:

```bash
uv venv -p 3.12
uv pip install \
  litert-lm \
  mlx-audio \
  soundfile \
  numpy \
  Pillow \
  huggingface_hub \
  "misaki[en]"
```

Other deps you'll want:

```bash
brew install ffmpeg               # for browser audio → wav transcode
# optional, for the Ollama fallback backend:
brew install ollama && ollama pull gemma4:e4b
```

### 1 · Phone setup (the "drone")

1. Install **IP Webcam** (Android, free) by Pavel Khlebovich.
2. Launch the app → "Start server". Note the URL (e.g. `192.168.1.42:8080`).
3. *(Optional, recommended)* In Settings → "Use camera" → pick the
   **ultra-wide** lens for a wider FOV.

### 2 · Run it

No-GPU smoke test (proves the loop end-to-end without any model):
```bash
python run_demo.py --scenario escort --backend scripted
```

The real, judged demo with Gemma 4 native audio + vision via LiteRT-LM:

```bash
python run_demo.py \
  --scenario escort \
  --backend litert \
  --web \
  --camera-url "http://<your-phone-ip>:8080/video" \
  --tts kokoro \
  --kokoro-voice af_heart \
  --litert-max-tokens 16384
```

Open the printed URL in a browser. You'll hear *"Voice ready"* if Kokoro
audio is working. Hold **SPACE** and say:

> *"There's smoke downstairs and I can't see — get Manu out, he's
> upstairs at the workbench."*

Release SPACE → the recording uploads → Gemma 4 transcribes via its
native audio encoder → mission begins. Walk the phone upstairs as the
agent scouts; when it sees a person it'll call `confirm_survivor_located`
and start the escort.

---

## Tools the agent has

Defined in [`rescuegemma/schemas.py`](rescuegemma/schemas.py).
The escort scenario exposes only this subset:

| Tool | Args | What it does |
|---|---|---|
| `takeoff` | `altitude_m` | Arm + ascend |
| `scout_ahead` | `segment` | Fly ahead, observe, return to Manu |
| `confirm_survivor_located` | `description`, `confidence` | Vision-asserted: "Manu is in this frame" → flips to escort phase |
| `speak_to_survivor` | `utterance`, `tone` | TTS line to the person (Kokoro voice out) |
| `lead_to_exit` | `segment`, `pace` | Move forward one segment at the person's pace |
| `report_finding` | `type`, `location`, `confidence`, `detail` | Structured finding to the operator |
| `return_to_operator` | `reason` | Mission complete |
| `abort_and_hold` | `reason` | Conservative bail-out |

The executor enforces a strict state machine — duplicate scouts, speaks
before scouting, leads before scouting, etc. all get hard REJECTs with
a directive pointing at the exact next call.

---

## Special Technology tracks targeted

- **Ollama** (`--backend ollama`) — single Gemma 4 E4B locally via Ollama
  (no audio in yet but vision + tools work).
- **llama.cpp** — same `OllamaBackend` pointed at Ollama-on-llama.cpp
  on a Jetson Orin Nano. Build b8766+ has Gemma 4 audio Conformer
  support ([PR #21421](https://github.com/ggml-org/llama.cpp/pull/21421)).
- **LiteRT** (`--backend litert`, default for the demo) — Google AI Edge's
  official runtime. The **only** local stack as of mid-2026 that exposes
  Gemma 4's audio + vision encoders end-to-end. This is the path the demo
  is judged on.

---

## File layout

```
.
├── README.md                          this file
├── LICENSE                            Apache-2.0
├── pyproject.toml                     package metadata
├── run_demo.py                        entry point
├── rescuegemma/
│   ├── __init__.py
│   ├── schemas.py                     function-calling contract (Gemma ⇄ executor)
│   ├── planner.py                     LiteRTBackend, OllamaBackend, GemmaBackend,
│   │                                  ScriptedBackends, run_mission
│   ├── escort_executor.py             two-phase find→escort body w/ state machine
│   ├── sim_executor.py                original search scenario body (kept working)
│   ├── tts.py                         Kokoro (mlx-audio) + macOS `say` + Null
│   └── webapp.py                      stdlib HTTP server + SSE + UI (single-file)
├── assets/
│   ├── egress.json                    5 find_segments + 5 escort_segments
│   ├── floorplan.json                 for the original search scenario
│   └── frames/                        (you drop phone walkthrough JPEGs here)
├── docs/
│   ├── WRITEUP.md                     technical write-up for the hackathon
│   ├── RECORDING_GUIDE.md             how to record your phone walkthrough
│   ├── overlay.html                   standalone reasoning-overlay player
│   └── DEVELOPMENT_NOTES.md           in-progress design notes
└── notebooks/
    └── rescuegemma_demo.py            Kaggle-runnable notebook
```

---

## Known issues / honest current state

This is research-grade. Things that work:
- ✅ Native Gemma 4 audio-in (operator's voice → text intent) via LiteRT-LM
- ✅ Per-turn vision (phone JPEG goes into every model call)
- ✅ Tool calling via Gemma's native `<|tool_call>` markers
- ✅ Kokoro neural TTS for voice out
- ✅ Two-phase mission (find → escort) with strict state-machine
- ✅ Stuck-detector with executor-suggested next-call override
- ✅ Auto-LED from frame luminance
- ✅ IP Webcam integration

Things that need iteration:
- ⚠ Gemma 4 E2B occasionally produces malformed tool calls (`scout` instead
  of `scout_ahead`, double quotes inside `<|"|>...<|"|>` envelopes).
  Parser is forgiving; executor REJECTs are sharp. Larger E4B model
  reduces this dramatically.
- ⚠ IP Webcam (free) doesn't expose multi-camera selection over HTTP —
  manual lens switch in the app's gear icon.
- ⚠ The `assets/frames/` are placeholders — the executor tolerates missing
  files. For a fully grounded judged demo, record a phone walkthrough
  per `docs/RECORDING_GUIDE.md`.

---

## Acknowledgements

- [parlor](https://github.com/fikrikarim/parlor) by Fikri Karim — the
  on-device voice + vision research preview we borrowed the LiteRT-LM +
  Kokoro pattern from.
- [Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M) — the neural TTS.
- [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) —
  turns any Android phone into the simulated drone.
- The llama.cpp + LiteRT-LM contributors for shipping Gemma 4 audio support
  weeks after the model dropped.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).
