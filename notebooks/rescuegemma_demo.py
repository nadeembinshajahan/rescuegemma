# RescueGemma — voice-commanded autonomous indoor search-and-rescue
# Kaggle Gemma 4 Good Hackathon · StratoFirma Autonomy Labs
#
# This notebook is runnable by judges with NO drone and NO GPU.
#
# HONEST STATEMENT OF WHAT IS REAL:
#   * The Gemma 4 reasoning, audio-in, vision-in, and function calling are REAL.
#   * The camera frames are REAL footage from a recorded building walkthrough.
#   * Only the flight DYNAMICS are simulated. The planner code path is the
#     identical one that runs on our Jetson/PX4 hexacopter (same schemas).
#   * A scripted backend is included ONLY as a no-GPU smoke test of the loop.
#     It is clearly labelled and is NOT the submission.
#
# Two ways to run:
#   A) backend="gemma"    -> real Gemma 4 26B (Mac/Colab GPU). The real demo.
#   B) backend="scripted"  -> proves the loop runs anywhere. Smoke test only.

# %% [markdown]
# ## 1. Setup
# %%
import sys, subprocess, os
# In Kaggle/Colab, install the package from the repo. Locally, it's importable.
if not os.path.exists("rescuegemma"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/stratofirma/rescuegemma.git"], check=False)

# %% [markdown]
# ## 2. The function-calling contract
# These are the drone primitives Gemma 4 calls. The SAME schema drives the
# simulator here and the real PX4 flight bridge on the aircraft.
# %%
from rescuegemma import TOOLS, SimExecutor, run_mission, ScriptedBackend
import json
print(json.dumps([t["name"] for t in TOOLS], indent=2))

# %% [markdown]
# ## 3. The building (real recorded walkthrough)
# `floorplan.json` maps real phone-walkthrough frames onto rooms. The agent
# is NEVER told which room holds the survivor — it must ground the operator's
# spoken description against the actual camera frames.
# %%
executor = SimExecutor("assets/floorplan.json", "assets/frames")

# %% [markdown]
# ## 4A. The real demo — Gemma 4 26B
# Uncomment on a machine with a Gemma 4 endpoint (our Mac runs 26B locally).
# `operator_call.wav` is a real recorded emergency utterance:
#   "There's a fire downstairs, my son may be in his room — the upstairs one
#    with the dinosaur posters. Find him and tell me if he's responsive."
# %%
# from rescuegemma import GemmaBackend
# backend = GemmaBackend(model="gemma-4-26b")  # set GEMMA_BASE_URL / GOOGLE_API_KEY
# turns = run_mission("assets/operator_call.wav", executor, backend)
# for t in turns:
#     print(t.call.name, t.call.arguments)
#     print("  ", t.result.observation)

# %% [markdown]
# ## 4B. Loop smoke test (no GPU, no model) — proves the architecture runs
# This is NOT the submission output. It demonstrates the closed loop executes
# and terminates anywhere, so judges can verify the architecture is real even
# without a GPU.
# %%
turns = run_mission("assets/operator_call.wav", executor, ScriptedBackend())
for i, t in enumerate(turns, 1):
    print(f"[{i}] {t.call.name}({t.call.arguments})")
    print(f"    -> {'OK' if t.result.ok else 'FAIL'}: {t.result.observation[:120]}")
    if t.result.frame_ref:
        print(f"    frame: {t.result.frame_ref}")

# %% [markdown]
# ## 5. Why Gemma 4 specifically
# - **Audio in (native):** operator speaks; no separate ASR. One model, on device.
# - **Vision in:** grounds "the room with the dinosaur posters" against the
#   actual frame instead of guessing — the core safety property.
# - **Function calling (native):** typed calls into PX4 primitives; the LLM is
#   the planner, PX4 is the executor, SLAM is the eyes.
# - **On-device / offline:** runs with the network physically unplugged. In a
#   collapsed building there is no cloud. Privacy-preserving by construction.
print("See docs/WRITEUP.md for the full technical narrative and ethics statement.")
