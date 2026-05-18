# RescueGemma — Demo Video & Recording Guide
*(drones down → digital-twin demo. This is the higher-scoring version anyway.)*

The hackathon explicitly rewards the **story** and the **"wow"**, with the
video and writeup setting winners apart — not airframe footage. So the video's
star is **Gemma 4's reasoning**, shown on screen, against **real recorded
footage**. Nothing is faked; only the drone's body is simulated.

---

## Part 1 — Record the walkthrough (today, ~30 min)

You are the drone. Record a continuous phone video walking a building.

**Path:** entry → hallway → Room A (decoy, e.g. a study: desk, shelves, NO
posters) → Room B (the target: put up a **dinosaur poster** or a kid's
backpack, and stage a **survivor** — a volunteer lying still near a window,
or a clothed mannequin / pillow rig under a blanket).

**Camera discipline:**
- Hold the phone at ~1.2–1.5 m, lens forward, like a drone's view.
- Move *slowly*. No whip pans. Pause ~3 s at each doorway and in each room.
- One clear close-pass on the survivor (this becomes `kids_room_close.jpg`).
- Even lighting; turn lights on. Avoid backlit windows blowing out the survivor.

**Extract frames:**
```bash
ffmpeg -i walkthrough.mp4 -vf fps=1 assets/frames/frame_%03d.jpg
# then rename the good ones to match floorplan.json:
#   hallway_01.jpg hallway_02.jpg study_01.jpg study_02.jpg
#   kids_room_01.jpg kids_room_02.jpg kids_room_close.jpg
```
Edit `assets/floorplan.json` so each room's `frames` list points at the right
files and `center` roughly matches the layout you walked.

## Part 2 — Record the operator call (2 min)

One WAV, spoken with real urgency, in plain words a parent would use:

> "There's a fire downstairs — my son might still be upstairs in his room,
> it's the one with the dinosaur posters. Find him and tell me if he's okay."

Save as `assets/operator_call.wav`. Do **not** write a robotic command. The
whole point is Gemma turning panicked human speech into a correct mission.

## Part 3 — Run the real agent and screen-record it

On the Mac with Gemma 4 26B:
```bash
export GEMMA_BASE_URL=http://localhost:8000
python run_demo.py --backend gemma --audio assets/operator_call.wav
```
Screen-record the terminal (or build the simple overlay UI from
`docs/mission_transcript.json`). You want, on screen, in real time:
the transcript appearing → the plan forming → each tool call firing →
the real frame it's looking at → the moment it says *"this room has the
dinosaur posters"* → the close scan → *"person located, not responding"* →
report → return.

## Part 4 — The 3-minute edit (shot list)

| t | On screen | Voiceover / caption |
|---|-----------|--------------------|
| 0:00–0:15 | Black → the spoken call plays, waveform | The problem: a person, a building, no time, no signal. |
| 0:15–0:35 | Gemma transcribes; intent line appears | One on-device model hears the panic and understands it. |
| 0:35–1:10 | `search_floor` → `explore_room`; decoy room frame; "no posters — not it" | It doesn't guess. It *looks*. |
| 1:10–1:50 | Target room frame; "dinosaur posters — this is the room"; survivor in frame | Natural-language grounding against real vision. |
| 1:50–2:20 | `hover_and_scan` close frame → `report_finding` plain-language alert | The operator gets a human answer, not telemetry. |
| 2:20–2:40 | Network cable physically unplugged on camera; loop keeps running | No cloud. It works where the cloud doesn't. |
| 2:40–3:00 | Slate: same agent, Jetson on our hexacopter + 5 s old flight clip if any | This brain ships on real hardware. |

**The unplug shot is the single most persuasive 5 seconds.** It proves offline.

## Part 5 — What to say about the drones being down

Say nothing apologetic. The framing is: *"The reasoning core is hardware-
agnostic. We demonstrate it against a recorded environment so anyone can
reproduce it; the identical planner runs on our Jetson/PX4 platform via the
same function-calling contract."* That is true, and it's a strength, not an
excuse — it's literally what makes the Kaggle notebook reproducible.
