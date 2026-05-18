# RescueGemma — Code Plan for Claude Code

**Read this whole file before writing code.** This is a 3-hour live hackathon
submission. The goal is a *working, honest* escort demo, not a polished
codebase. Optimize for "runs once on real footage and produces a real
transcript," then stop.

---

## 0. Context you need

**What this is:** a voice-commanded autonomous indoor *escort* agent on
Gemma 4. An operator speaks; Gemma 4 (on-device, no network) finds a trapped
mobile person (Manu), then *leads him out* — workbench → door → down the
stairs — scouting each path segment before committing him to it, speaking to
him aloud, and reassessing when he lags.

**The honesty contract (do not violate this):**
- Gemma 4's audio-in, vision-in, function calling, reasoning = REAL.
- Camera frames = REAL footage from a phone walkthrough the user records.
- Only flight dynamics and the person's follow-signal are simulated.
- The planner code path in the demo MUST be identical to what would run on
  hardware (same schemas). Do not add demo-only shortcuts in the planner.
- A `ScriptedBackend` exists ONLY as a no-GPU smoke test. It is never the
  submission output and must stay clearly labelled.

**Judging reality:** story + working demo + writeup. A real rough escort
beats a polished fake. The rubric explicitly penalizes mocked-up demos.

---

## 1. Current repo state (already built — do NOT rewrite)

```
rescuegemma/
  __init__.py            exports: TOOLS, ToolCall, ToolResult, SimExecutor,
                          run_mission, GemmaBackend, ScriptedBackend, Turn
  schemas.py             function-calling contract. ALREADY EXTENDED for escort:
                          - FindingType has HAZARD_STAIRS/OBSTRUCTION/NARROW,
                            EGRESS_REACHED
                          - GuidanceTone enum (calm_directive/urgent/reassuring)
                          - TOOLS includes speak_to_survivor, scout_ahead,
                            lead_to_exit (plus the original search tools)
  sim_executor.py        SEARCH-scenario body (kept, not used by escort)
  escort_executor.py     ESCORT body. COMPLETE. class EscortExecutor.
                          Reads assets/egress.json, serves real frames per
                          segment, tracks follow-state, enforces
                          "scout before lead", has a one-time 'lags' beat.
  planner.py             run_mission loop + GemmaBackend + ScriptedBackend.
                          run_mission is generic over any executor with
                          .execute(ToolCall)->ToolResult. NOTE: GemmaBackend
                          and ScriptedBackend are currently SEARCH-oriented.
run_demo.py              entrypoint (currently wired for SEARCH/SimExecutor)
assets/
  egress.json            ESCORT path: workbench_to_door, door_to_stairtop
                          (follow_behaviour='lags'), stairs_descent. Frame
                          filenames are PLACEHOLDERS to be replaced post-record.
  floorplan.json         search scenario (ignore for escort)
  frames/                empty; user drops real frames here
docs/                    WRITEUP.md, RECORDING_GUIDE.md, video script PDF
notebooks/rescuegemma_demo.py
```

Verify this by reading the files first. If anything above is missing or
differs, trust the files, not this doc.

---

## 2. What you must build (in strict priority order)

### TASK 1 — Escort system prompt + escort-aware GemmaBackend  [CRITICAL]

`planner.py` currently has `SYSTEM_PROMPT` written for search. The escort
mission needs its own prompt. Do this without breaking the search path.

- Add `ESCORT_SYSTEM_PROMPT` (new constant). It must instruct Gemma to:
  1. Use `speak_to_survivor` to generate ITS OWN words from the current
     frame — never canned phrases. Short, plain, for a frightened person.
  2. ALWAYS `scout_ahead` a segment before `lead_to_exit` on it. Leading
     into uninspected space is forbidden.
  3. When `lead_to_exit` returns `follow_state: lagging`, do NOT retry
     blindly — hold, re-`speak_to_survivor` (reassuring, slower), then
     attempt the segment again.
  4. `report_finding` for each hazard it actually sees in a frame.
  5. End with `return_to_operator` once `egress_complete: true`.
- Make `GemmaBackend.__init__` accept `system_prompt: str` (default = the
  existing search prompt for backward compat). `run_demo.py` will pass the
  escort one.
- Keep `transcribe_intent` and `next_call` as-is structurally. They already
  do audio-in (operator wav) and vision-in (attach frame_ref) correctly.
  The frame the escort executor returns must be base64-attached on the next
  `next_call`, exactly as the search path already does.

### TASK 2 — Escort smoke backend  [HIGH]

Add `EscortScriptedBackend` to `planner.py`. Same purpose as
`ScriptedBackend`: prove the closed loop runs with NO GPU/model. It must
drive a correct escort sequence by reading the executor's last observation
(not a blind fixed list — the 'lags' beat means step count is unreliable):

```
takeoff
→ scout_ahead(workbench to door)
→ speak_to_survivor(...)            # "wait / come to me, keep right"
→ lead_to_exit(workbench_to_door)  # follows
→ scout_ahead(stairs)
→ speak_to_survivor(...)           # hazard warning
→ lead_to_exit(door_to_stairtop)   # returns LAGGING
→ speak_to_survivor(...)           # reassure, slower
→ lead_to_exit(door_to_stairtop)   # now follows (executor flips it)
→ lead_to_exit(stairs_descent)     # needs scout first → executor REFUSES →
                                     so: scout_ahead(descent) → lead → done
→ report_finding(egress_reached)
→ return_to_operator
```
Drive transitions off substrings in the last tool observation
(`"LAGGED"`, `"not scouted"`, `"egress complete"`, `"Spoke aloud"`). Cap at
~16 steps and `abort_and_hold` if exceeded so it can never infinite-loop.
Mark every thought line with `[SCRIPTED]`. This is a test harness.

### TASK 3 — run_demo.py: add an escort mode  [HIGH]

Add `--scenario {search,escort}` (default `escort`). When `escort`:
- build `EscortExecutor("assets/egress.json", "assets/frames")`
- backend `gemma` → `GemmaBackend(model=..., system_prompt=ESCORT_SYSTEM_PROMPT)`
- backend `scripted` → `EscortScriptedBackend()`
- write `docs/mission_transcript.json` (already done by the on_turn hook —
  ensure `extra.spoken`, `extra.tone`, `extra.follow_state`, `extra.finding`
  are all persisted; the video overlay needs `spoken`/`tone`).
Keep `search` working exactly as before.

### TASK 4 — Verify the smoke loop end to end  [HIGH]

```
python run_demo.py --scenario escort --backend scripted
```
Must terminate cleanly with: takeoff → scouts → speaks → a LAGGING then
recovery on the stairs → egress complete → return. No infinite loop. The
printed transcript should read like a coherent rescue. If it doesn't,
fix the executor/backend until it does. THIS IS THE GATE before touching
the real model.

### TASK 5 — HTML reasoning overlay  [MEDIUM — do only if Tasks 1–4 done]

`docs/overlay.html`, single file, no network, no build step. It reads a
hardcoded/inlined copy of `mission_transcript.json` (or fetches the local
file) and renders the right-hand "reasoning panel" from the video script:
for each turn, in sequence with a small delay, show
`AUDIO/VISION badge → thought → tool call (mono) → spoken line (quoted, with
tone)`. Dark theme, monospace for calls, large readable speech. The user
will screen-record this beside the room footage. Keep it ONE file, ~150
lines. No frameworks. A "Play" button that steps through turns is enough.

### TASK 6 — Real run support  [the actual submission moment]

Make sure this path works on the user's Mac with a local Gemma 4 endpoint:
```
export GEMMA_BASE_URL=http://localhost:<port>
export GOOGLE_API_KEY=...            # only if hosted
python run_demo.py --scenario escort --backend gemma \
    --audio assets/operator_call.wav
```
- Confirm `google-genai` import path and the function-calling request shape
  in `GemmaBackend.next_call` against the installed SDK version. If the SDK
  surface differs, ADAPT the call but keep the contract: one tool call back
  per turn, frame attached as inline image, tools = schemas.TOOLS.
- If audio-in via `transcribe_intent` fails on the local model, fall back to
  attaching the wav on the first `next_call` instead — but log clearly that
  it did, so the user knows. Never silently fake the transcript.

---

## 3. The data the user provides (parallel to your work)

User is recording NOW. They will deliver, into `assets/`:
- `frames/` — real jpgs from `ffmpeg -i walk.mp4 -vf fps=1 frames/seg_%03d.jpg`
- `operator_call.wav` — a real spoken: "There's smoke downstairs, I can't
  see — get Manu out, he's at the workbench."

Then they edit `assets/egress.json` so each segment's `scout_frames` /
`lead_frames` point at the real filenames. Your code must NOT hardcode frame
names — read them from egress.json (escort_executor already does this).
If frames are missing at runtime the executor tolerates it (returns the
bare filename); the smoke test still passes without any frames.

---

## 4. Hard rules / definition of done

- Search path still runs: `python run_demo.py --scenario search --backend scripted`.
- Escort smoke runs and terminates: Task 4 command, coherent transcript.
- `mission_transcript.json` contains, per turn: step, thought, call+args,
  ok, observation, and `spoken`/`tone` whenever speak_to_survivor fired.
- No infinite loops anywhere (hard step cap + abort fallback).
- ScriptedBackend / EscortScriptedBackend thoughts all prefixed `[SCRIPTED]`.
- No new dependencies except `google-genai` (already optional in
  pyproject.toml) and nothing for the HTML overlay.
- Do NOT "improve" the writeup, schemas docstrings, or search path. Time box.
- When Tasks 1–4 pass, STOP and tell the user to do the real run. Do not
  gold-plate. The remaining value is in their footage + the real model run,
  not more code.

## 5. Suggested order & rough budget (of your share, ~75 min)

1. Read schemas.py, escort_executor.py, planner.py, run_demo.py (5 min)
2. Task 1 escort prompt + GemmaBackend system_prompt arg (15)
3. Task 2 EscortScriptedBackend (15)
4. Task 3 run_demo escort mode (10)
5. Task 4 verify smoke loop, fix until coherent (10)
6. Task 5 overlay.html (15)
7. Task 6 sanity-check the real-run code path (5)
Then hand back to the user for footage + real run + edit + submit.
