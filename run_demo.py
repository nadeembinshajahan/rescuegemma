"""
RescueGemma demo runner.

Two scenarios, same closed loop, four backends:
  * escort  — voice-commanded escort of a mobile trapped person (Manu)
              out of the building. Real frames from a phone walkthrough,
              real Gemma 4 reasoning, only flight dynamics simulated.
  * search  — original search-and-rescue scenario (kept working).

Usage:
    # Loop verification, no GPU / no model (proves the architecture runs):
    python run_demo.py --scenario escort --backend scripted

    # *** The submission default ***  — single Gemma 4 E4B via Ollama on Mac.
    # E4B has native audio + vision + tools, ~9.6 GB, runs fast on Apple Silicon:
    ollama pull gemma4:e4b
    python run_demo.py --scenario escort --backend ollama \
        --audio assets/operator_call.wav

    # Optional: split audio onto a smaller Gemma 4 E2B (e.g. on an 8GB Mac):
    ollama pull gemma4:e2b
    python run_demo.py --scenario escort --backend routed \
        --audio assets/operator_call.wav

    # google-genai SDK (hosted AI Studio):
    export GOOGLE_API_KEY=...
    python run_demo.py --scenario escort --backend gemma \
        --audio assets/operator_call.wav

Outputs:
    * A readable mission transcript to stdout.
    * docs/mission_transcript.json — drives the video overlay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from rescuegemma import (
    SimExecutor,
    EscortExecutor,
    run_mission,
    GemmaBackend,
    OllamaBackend,
    LMStudioBackend,
    LiteRTBackend,
    RoutedBackend,
    ScriptedBackend,
    EscortScriptedBackend,
    ESCORT_SYSTEM_PROMPT,
)
from rescuegemma import webapp as _webapp
from rescuegemma import tts as _tts


def _try_ipcam_ultrawide(camera_url: str, override: str | None = None) -> None:
    """Best-effort: switch the IP Webcam app on the operator's phone to the
    ultra-wide back lens. IP Webcam doesn't publish a clean public API for
    multi-camera selection (it varies by app version and phone), so we try
    a handful of /settings/ candidates. If none work, we log it loudly so
    the operator knows to switch manually in the app.

    ``override`` is a single ``key=value`` from ``--ipcam-init`` — if set
    we try only that pair.
    """
    if not camera_url:
        return
    from urllib.parse import urlparse
    import urllib.request as _ur
    u = urlparse(camera_url if "://" in camera_url else "http://" + camera_url)
    base = f"{u.scheme}://{u.hostname}:{u.port or 8080}"

    candidates: list[tuple[str, str]] = []
    if override and "=" in override:
        k, v = override.split("=", 1)
        candidates.append((k.strip(), v.strip()))
    else:
        # Most-likely keys across IP Webcam free + Pro + forks. Ordered by
        # what I've seen actually used in the wild.
        candidates = [
            ("back_camera",   "ultra_wide"),
            ("back_camera",   "ultrawide"),
            ("back_camera",   "wide"),       # some builds call it "wide"
            ("back_camera",   "1"),
            ("camera_id",     "1"),
            ("camera_id",     "ultra_wide"),
            ("backcam_id",    "1"),
            ("camera",        "1"),
            ("photo_camera",  "1"),
        ]

    print("[ipcam] trying to select ultra-wide back lens...")
    success: tuple[str, str] | None = None
    for k, v in candidates:
        url = f"{base}/settings/{k}?set={v}"
        try:
            req = _ur.Request(url, headers={"User-Agent": "RescueGemma/0.1"})
            with _ur.urlopen(req, timeout=2) as r:
                status = r.getcode()
                body = (r.read() or b"").decode("utf-8", "replace").strip()
        except Exception as e:
            print(f"[ipcam]   GET {url} → ERR {e}")
            continue
        # IP Webcam returns plain text. "Bad" / "Unknown" / "no such key"
        # indicate failure; "Set ok" or empty 200 indicate success.
        bad = any(s in body.lower() for s in ("bad", "unknown", "not found",
                                              "no such", "invalid"))
        print(f"[ipcam]   GET {url} → {status} {body[:80]!r}")
        if status == 200 and not bad:
            success = (k, v)
            break
    if success:
        print(f"[ipcam] ✓ active camera setting: {success[0]}={success[1]}")
    else:
        print("[ipcam] ⚠  Could not auto-switch lens. "
              "Open the IP Webcam app → tap the gear → 'Use camera' / "
              "'Back camera' and pick the ultra-wide lens manually.")


def _make_ipcam_frame_provider(camera_url: str):
    """Return a callable that fetches a fresh JPEG snapshot from the IP
    camera's ``/shot.jpg`` endpoint. Returns bytes or None on failure.
    Each invocation is a fresh HTTP GET — these are cheap (IP Webcam
    serves a single JPEG per request) and let the model see whatever the
    operator is pointing the phone at right now.
    """
    if not camera_url:
        return None
    from urllib.parse import urlparse
    import urllib.request as _ur
    u = urlparse(camera_url if "://" in camera_url else "http://" + camera_url)
    host = u.hostname or ""
    port = u.port or 8080
    base = f"{u.scheme}://{host}:{port}"
    shot_url = base + "/shot.jpg"
    def _provider() -> bytes | None:
        try:
            req = _ur.Request(shot_url,
                              headers={"User-Agent": "RescueGemma/0.1"})
            with _ur.urlopen(req, timeout=3) as r:
                return r.read()
        except Exception as e:
            print(f"[frame] {shot_url} failed: {e}")
            return None
    return _provider


def _maybe_transcode_to_wav(src_path: str) -> str:
    """If src_path is not already a wav, transcode it to 16kHz mono wav via
    ffmpeg next to the original. Returns the wav path on success, otherwise
    returns src_path unchanged (with a loud log so the user knows)."""
    if src_path.lower().endswith(".wav"):
        return src_path
    import shutil, subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print(f"[transcode] ffmpeg not on PATH — sending {src_path} as-is. "
              f"If Ollama rejects it, `brew install ffmpeg` and re-record.")
        return src_path
    dst = os.path.splitext(src_path)[0] + ".wav"
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-i", src_path,
           "-ac", "1", "-ar", "16000",
           "-sample_fmt", "s16",
           dst]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=20)
        if r.returncode == 0 and os.path.exists(dst):
            print(f"[transcode] {os.path.basename(src_path)} -> {os.path.basename(dst)} (16kHz mono wav)")
            return dst
        print(f"[transcode] ffmpeg failed ({r.returncode}): {r.stderr.decode(errors='replace')[:200]}")
    except Exception as e:
        print(f"[transcode] ffmpeg error: {e}")
    return src_path


def _fmt(turn) -> str:
    c = turn.call
    args = ", ".join(f"{k}={v!r}" for k, v in c.arguments.items())
    head = f"  thought : {turn.thought}"
    callln = f"  call    : {c.name}({args})"
    res = f"  result  : {'OK ' if turn.result.ok else 'FAIL '}{turn.result.observation}"
    frame = f"  frame   : {turn.result.frame_ref}" if turn.result.frame_ref else ""
    extras = []
    ex = turn.result.extra or {}
    if "spoken" in ex:
        extras.append(f"  spoken  : ({ex.get('tone')}) \"{ex['spoken']}\"")
    if "follow_state" in ex:
        extras.append(f"  follow  : {ex['follow_state']}")
    return "\n".join(x for x in [head, callln, res, frame, *extras] if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["search", "escort"], default="escort")
    ap.add_argument("--backend",
                    choices=["scripted", "litert", "ollama", "lmstudio",
                             "routed", "gemma"],
                    default="scripted",
                    help="litert = Gemma 4 E2B via LiteRT-LM (TRUE native "
                         "audio + vision; submission default). "
                         "ollama / lmstudio = GGUF, but neither forwards "
                         "audio to Gemma 4 today. scripted = no-model smoke.")
    ap.add_argument("--litert-model-path", default=None,
                    help="Path to a local .litertlm file. Default: "
                         "auto-download from Hugging Face "
                         "(litert-community/gemma-4-E2B-it-litert-lm).")
    ap.add_argument("--litert-max-tokens", type=int, default=16384,
                    help="LiteRT engine context size (max_num_tokens). "
                         "Default 16384 — bump up if you see "
                         "'Input token ids are too long' errors.")
    ap.add_argument("--lmstudio-host", default=None,
                    help="LM Studio base URL. Default http://localhost:1234 "
                         "or $LMSTUDIO_HOST.")
    ap.add_argument("--lmstudio-model", default="google/gemma-4-e4b",
                    help="LM Studio model id (see GET /v1/models).")
    ap.add_argument("--audio", default=None,
                    help="Operator voice clip. Defaults per scenario.")
    ap.add_argument("--floorplan", default="assets/floorplan.json")
    ap.add_argument("--egress", default="assets/egress.json")
    ap.add_argument("--frames", default="assets/frames")
    ap.add_argument("--reasoning-model", default="gemma4:e4b",
                    help="Ollama tag for the Gemma 4 reasoning model.")
    ap.add_argument("--audio-model", default="gemma4:e2b",
                    help="Ollama tag for the small Gemma 4 used for audio in "
                         "(--backend routed only).")
    ap.add_argument("--model", default="gemma-4-e4b",
                    help="(--backend gemma only) google-genai model name.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Cap on agent turns. Defaults: 12 search, 20 escort.")
    ap.add_argument("--web", action="store_true",
                    help="Serve a live web UI (video + reasoning panel) at "
                         "http://<host>:<port>/ and stream turns over SSE.")
    ap.add_argument("--web-port", type=int, default=8765)
    ap.add_argument("--camera-url", default="",
                    help="MJPEG / video URL of the phone IP camera. Passed "
                         "to the web UI so the user doesn't have to paste it.")
    ap.add_argument("--web-wait", type=float, default=2.5,
                    help="Seconds to wait after starting the web server, so "
                         "the user can open the browser before turns begin.")
    ap.add_argument("--ultrawide", action="store_true", default=True,
                    help="Try to auto-switch the IP Webcam back lens to "
                         "ultra-wide on startup. Default on.")
    ap.add_argument("--no-ultrawide", dest="ultrawide", action="store_false",
                    help="Disable the ultra-wide auto-switch.")
    ap.add_argument("--ipcam-init", default=None,
                    help='Override the lens selection with one exact '
                         '"key=value" pair (e.g. "back_camera=ultra_wide" or '
                         '"camera_id=2"). Sent as /settings/<key>?set=<value>.')
    ap.add_argument("--auto-torch", default="on", choices=["on", "off"],
                    help="Auto-toggle the phone LED from camera luminance "
                         "(when --camera-url is set).")
    ap.add_argument("--torch-dark", type=float, default=55.0,
                    help="Mean luminance (0-255) below which the LED turns on.")
    ap.add_argument("--torch-bright", type=float, default=105.0,
                    help="Mean luminance (0-255) above which the LED turns off.")
    ap.add_argument("--tts", default="auto",
                    choices=["auto", "kokoro", "say", "piper", "off"],
                    help="Voice-out backend. 'auto' picks Kokoro (neural) "
                         "if available, else Piper, else macOS say.")
    ap.add_argument("--kokoro-voice", default="af_heart",
                    help="Kokoro voice id. af_heart=warm female (default), "
                         "af_bella, am_michael, am_adam, bf_emma, bm_lewis.")
    ap.add_argument("--step-delay", type=float, default=3.5,
                    help="Seconds to wait after each agent turn so the demo "
                         "can be physically performed (you move the phone "
                         "between commands). After speak_to_survivor the "
                         "wait is extended to let TTS finish playing.")
    ap.add_argument("--speak-pad", type=float, default=0.7,
                    help="Extra seconds added to the step delay after a "
                         "speak_to_survivor, on top of an estimate of how "
                         "long the line takes to say at the chosen TTS rate.")
    ap.add_argument("--tts-voice", default="auto",
                    help="macOS `say` voice name, or 'auto' to pick the most "
                         "natural installed voice. Try: say -v ?")
    ap.add_argument("--tts-rate", type=int, default=175,
                    help="macOS `say` rate in words/minute.")
    ap.add_argument("--piper-model", default=None,
                    help="Path to a Piper voice .onnx (optional, neural TTS).")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    frames = os.path.join(here, args.frames)

    if args.scenario == "escort":
        executor = EscortExecutor(os.path.join(here, args.egress), frames)
        if args.backend == "litert":
            backend = LiteRTBackend(
                model_path=args.litert_model_path,
                system_prompt=ESCORT_SYSTEM_PROMPT,
                tool_names=LiteRTBackend.ESCORT_TOOL_NAMES,
                frame_provider=_make_ipcam_frame_provider(args.camera_url),
                max_num_tokens=args.litert_max_tokens,
            )
        elif args.backend == "lmstudio":
            backend = LMStudioBackend(model=args.lmstudio_model,
                                      host=args.lmstudio_host,
                                      system_prompt=ESCORT_SYSTEM_PROMPT)
        elif args.backend == "routed":
            backend = RoutedBackend(reasoning_model=args.reasoning_model,
                                    audio_model=args.audio_model,
                                    system_prompt=ESCORT_SYSTEM_PROMPT)
        elif args.backend == "ollama":
            backend = OllamaBackend(model=args.reasoning_model,
                                    system_prompt=ESCORT_SYSTEM_PROMPT)
        elif args.backend == "gemma":
            backend = GemmaBackend(model=args.model,
                                   system_prompt=ESCORT_SYSTEM_PROMPT)
        else:
            backend = EscortScriptedBackend()
        default_audio = "assets/operator_call.wav"
        max_steps = args.max_steps or 40
    else:
        executor = SimExecutor(os.path.join(here, args.floorplan), frames)
        if args.backend == "litert":
            backend = LiteRTBackend(model_path=args.litert_model_path)
        elif args.backend == "lmstudio":
            backend = LMStudioBackend(model=args.lmstudio_model,
                                      host=args.lmstudio_host)
        elif args.backend == "routed":
            backend = RoutedBackend(reasoning_model=args.reasoning_model,
                                    audio_model=args.audio_model)
        elif args.backend == "ollama":
            backend = OllamaBackend(model=args.reasoning_model)
        elif args.backend == "gemma":
            backend = GemmaBackend(model=args.model)
        else:
            backend = ScriptedBackend()
        default_audio = "assets/operator_call.wav"
        max_steps = args.max_steps or 12

    print("=" * 70)
    print(f"RescueGemma mission  |  scenario={args.scenario}  backend={args.backend}")
    print("=" * 70)

    # Voice out (system -> Manu). Load BEFORE we block on the operator's
    # SPACE-bar recording so the audio probe ("Voice ready") fires while
    # the operator still has time to fix Mac sound settings.
    voice = _tts.make_voice(
        mode=args.tts,
        voice_name=args.tts_voice,
        rate_wpm=args.tts_rate,
        piper_model=args.piper_model,
        kokoro_voice=args.kokoro_voice,
    )
    print(f"[tts] backend = {voice.backend_name}")

    # Optional live web UI (video feed + reasoning timeline + voice in).
    if args.web:
        # Where uploaded operator clips land.
        audio_dir = os.path.join(here, "assets", "live")
        _webapp.configure_audio_dir(audio_dir)
        _webapp.start_server(port=args.web_port)
        _webapp.publish_reset()
        host = _webapp.local_ip()
        from urllib.parse import quote
        url = f"http://localhost:{args.web_port}/"
        if args.camera_url:
            url += "?camera=" + quote(args.camera_url, safe="")
        print(f"\n  🌐  Live UI:   {url}")
        print(f"  🌐  LAN:       http://{host}:{args.web_port}/")
        print(f"  📱  Camera:    {args.camera_url or '(paste an MJPEG URL in the UI)'}")
        # Switch to ultra-wide lens on the phone (best-effort).
        if args.camera_url and (args.ultrawide or args.ipcam_init):
            _try_ipcam_ultrawide(args.camera_url, override=args.ipcam_init)
        # Auto-LED from camera luminance, if camera URL is known.
        if args.camera_url and args.auto_torch == "on":
            _webapp.start_auto_torch(
                camera_base=args.camera_url,
                dark_threshold=args.torch_dark,
                bright_threshold=args.torch_bright,
            )
            print(f"  💡  Auto-LED:  on (dark<{args.torch_dark:.0f}, bright>{args.torch_bright:.0f}, ~2s poll)")
        # If --audio was explicitly given, run with that. Otherwise, block
        # until the operator's voice comes in over the browser (push-to-talk).
        audio_arg = args.audio
        if audio_arg and os.path.exists(os.path.join(here, audio_arg)):
            print(f"  🎙  Audio:     {audio_arg} (preloaded; mission auto-starts)")
            print(f"  ⏳  Mission starts in {args.web_wait:.1f}s — open the page now.\n")
            import time as _t
            _t.sleep(max(0.0, args.web_wait))
        else:
            print("  🎙  Audio:     hold SPACE in the browser to record operator voice.")
            print("  ⏳  Waiting for operator voice over the browser…\n")
            _webapp.publish_status("waiting for operator voice")
            live_audio = _webapp.TRIGGER.wait()
            if live_audio:
                # MediaRecorder gives webm/opus; Ollama's audio handler wants
                # PCM wav. Transcode 16 kHz mono via ffmpeg.
                live_audio = _maybe_transcode_to_wav(live_audio)
                # IMPORTANT: assign back so the audio var below picks it up.
                args.audio = os.path.relpath(live_audio, here)
                print(f"  🎙  Received: {args.audio}")
        _webapp.publish_status("mission starting")

    # Compute the final audio path AFTER any trigger override above.
    audio = os.path.join(here, args.audio or default_audio)
    if args.web and not os.path.exists(audio):
        print(f"[run_demo] warning: audio file not found at {audio}")

    transcript = []

    def _turn_to_json(turn, step):
        ex = turn.result.extra or {}
        return {
            "step": step,
            "thought": turn.thought,
            "call": {"name": turn.call.name, "args": turn.call.arguments},
            "ok": turn.result.ok,
            "observation": turn.result.observation,
            "pose": turn.result.pose,
            "frame": turn.result.frame_ref,
            "finding": ex.get("finding"),
            "spoken": ex.get("spoken"),
            "tone": ex.get("tone"),
            "follow_state": ex.get("follow_state"),
            "segment": ex.get("segment"),
            "egress_complete": ex.get("egress_complete"),
            # The actual phone JPEG bytes Gemma saw this turn, base64'd.
            "live_frame_b64": ex.get("live_frame_b64"),
        }

    def on_turn(turn):
        n = len(transcript) + 1
        print(f"\n[step {n}]")
        print(_fmt(turn))
        rec = _turn_to_json(turn, n)
        transcript.append(rec)
        if args.web:
            _webapp.publish_turn(rec)
        # Voice OUT (system -> Manu). Speak whenever Gemma chose to.
        spoken = rec.get("spoken")
        if spoken:
            voice.speak(spoken, tone=rec.get("tone"))
        # Pacing: hold the loop so the operator can physically move the
        # phone (= drone POV) between commands. Speech needs longer so
        # Kokoro finishes playing before the next step starts.
        if args.step_delay > 0:
            wait = float(args.step_delay)
            if spoken:
                # Rough estimate: Kokoro at ~14 chars/sec.
                est = len(spoken) / 14.0 + float(args.speak_pad)
                wait = max(wait, est)
            import time as _t
            _t.sleep(wait)

    # Capture the operator's intent (so the web UI shows what the model heard).
    if args.web:
        try:
            intent = backend.transcribe_intent(audio)
            _webapp.HUB.set_meta(intent=intent)
            _webapp.publish_status("intent transcribed", intent=intent)
            print(f"\n[intent] {intent}\n")
            # Patch the backend so run_mission doesn't re-transcribe.
            backend.transcribe_intent = lambda _p, _i=intent: _i  # type: ignore
        except Exception as e:
            print(f"[run_demo] intent transcription failed: {e}")

    run_mission(audio, executor, backend, max_steps=max_steps, on_turn=on_turn)
    if args.web:
        _webapp.publish_status("mission complete")

    out = os.path.join(here, "docs", "mission_transcript.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(transcript, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"Mission complete. {len(transcript)} steps.")
    print(f"Transcript -> {out}  (feeds the video overlay)")
    print("=" * 70)

    if args.web:
        print(f"\n  🌐  UI still live at http://localhost:{args.web_port}/")
        print("      Ctrl-C to quit.\n")
        try:
            import time as _t
            while True:
                _t.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
