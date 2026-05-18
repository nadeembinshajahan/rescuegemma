"""
RescueGemma — live web app server.

A tiny stdlib HTTP server (no new dependencies) that:

  * Serves a single-page reasoning + video UI at ``/``.
  * Streams agent turns to the browser over Server-Sent Events at ``/events``.
  * Optionally proxies an MJPEG stream from a phone IP-camera at ``/proxy``
    (used only if the browser can't talk to the camera directly because of
    CORS / mixed content).

Designed to run alongside ``run_demo.py`` in one process. The mission
publishes turns into ``HUB``; every connected browser sees them in real
time, with the live MJPEG from the phone alongside.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import socket
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional


class _Hub:
    """One in-memory pub/sub for the running mission. Each SSE client has
    its own queue; events are fanned out to every subscriber."""

    def __init__(self) -> None:
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = {}

    def set_meta(self, **kwargs: Any) -> None:
        with self._lock:
            self._meta.update(kwargs)

    def history_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"meta": dict(self._meta), "turns": list(self._history)}

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._subs.add(q)
            q.put({"type": "snapshot",
                   "meta": dict(self._meta),
                   "turns": list(self._history)})
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            if event.get("type") == "turn":
                self._history.append(event["turn"])
            elif event.get("type") == "reset":
                self._history.clear()
            for q in list(self._subs):
                q.put(event)


HUB = _Hub()


class _MissionTrigger:
    """One-slot mailbox used by ``run_demo.py`` to wait for a live operator
    voice clip from the browser. The UI POSTs ``/audio`` (the wav/webm
    blob) and the server stuffs the saved path here; ``wait()`` returns it.

    Re-armable: each ``wait()`` consumes one trigger, then a new recording
    can fire the next mission.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._slot: Optional[str] = None

    def fire(self, audio_path: str) -> None:
        with self._cv:
            self._slot = audio_path
            self._cv.notify_all()

    def wait(self, timeout: Optional[float] = None) -> Optional[str]:
        with self._cv:
            if self._slot is None:
                self._cv.wait(timeout=timeout)
            path = self._slot
            self._slot = None
            return path

    def reset(self) -> None:
        with self._cv:
            self._slot = None


TRIGGER = _MissionTrigger()


class AutoTorch:
    """Polls a single JPEG snapshot from the IP camera, measures mean
    luminance, and toggles the phone's LED through the IP-Webcam HTTP
    control API. Hysteresis prevents flapping when luminance hovers
    near the threshold.

    Designed for low overhead: one HTTP GET every ``poll_s`` seconds
    plus a small PIL decode and one mean(). Runs in its own daemon
    thread; ``pause()`` / ``resume()`` let the user manually override
    (e.g. when they tap the LED button in the UI).
    """

    def __init__(self,
                 camera_base: str,
                 dark_threshold: float = 55.0,
                 bright_threshold: float = 105.0,
                 poll_s: float = 2.0,
                 hysteresis_dark: int = 2,
                 hysteresis_bright: int = 3,
                 shot_path: str = "/shot.jpg"):
        self.base = _normalize_ipcam_base(camera_base)
        self.dark = dark_threshold
        self.bright = bright_threshold
        self.poll_s = poll_s
        self.h_dark = hysteresis_dark
        self.h_bright = hysteresis_bright
        self.shot_path = shot_path
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._on: Optional[bool] = None      # known state of phone LED
        self._dark_count = 0
        self._bright_count = 0
        self._last_lum: Optional[float] = None
        self._t: Optional[threading.Thread] = None

    # ---- public API ----
    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, name="auto-torch",
                                   daemon=True)
        self._t.start()
        self._publish(state="started")

    def stop(self) -> None:
        self._stop.set()
        self._publish(state="stopped")

    def pause(self) -> None:
        self._paused.set()
        self._publish(state="paused")

    def resume(self) -> None:
        self._paused.clear()
        # Reset counters so a manual flip doesn't snap back instantly.
        self._dark_count = 0
        self._bright_count = 0
        self._publish(state="resumed")

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def status(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "running": bool(self._t and self._t.is_alive()),
            "paused": self.paused,
            "led_on": self._on,
            "luminance": self._last_lum,
            "thresholds": {"dark": self.dark, "bright": self.bright},
        }

    # ---- inner loop ----
    def _run(self) -> None:
        try:
            from PIL import Image  # noqa: F401
        except Exception as e:
            print(f"[auto-torch] PIL not available ({e}). Disabling auto-LED.")
            return
        # Quick reachability check.
        time.sleep(0.5)
        while not self._stop.is_set():
            if not self._paused.is_set():
                try:
                    lum = self._sample_luminance()
                    if lum is not None:
                        self._last_lum = lum
                        self._decide(lum)
                except Exception as e:
                    # Don't crash the thread on transient camera errors.
                    print(f"[auto-torch] sample error: {e}")
            self._stop.wait(self.poll_s)

    def _sample_luminance(self) -> Optional[float]:
        from PIL import Image
        import io
        target = self.base + self.shot_path
        req = urllib.request.Request(
            target, headers={"User-Agent": "RescueGemmaAutoTorch/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                blob = r.read()
        except Exception as e:
            print(f"[auto-torch] fetch failed ({target}): {e}")
            return None
        try:
            with Image.open(io.BytesIO(blob)) as im:
                # Downsample for speed; we only need the mean.
                im.thumbnail((96, 64))
                gray = im.convert("L")
                px = gray.tobytes()
                if not px:
                    return None
                return sum(px) / len(px)
        except Exception as e:
            print(f"[auto-torch] decode failed: {e}")
            return None

    def _decide(self, lum: float) -> None:
        # Hysteresis: count consecutive dark/bright samples.
        if lum < self.dark:
            self._dark_count += 1
            self._bright_count = 0
        elif lum > self.bright:
            self._bright_count += 1
            self._dark_count = 0
        else:
            # In the dead zone — don't change state, but bleed counters.
            self._dark_count = max(0, self._dark_count - 1)
            self._bright_count = max(0, self._bright_count - 1)

        new_state: Optional[bool] = None
        if self._dark_count >= self.h_dark and self._on is not True:
            new_state = True
        elif self._bright_count >= self.h_bright and self._on is not False:
            new_state = False

        if new_state is not None:
            self._apply(new_state, lum)

    def _apply(self, on: bool, lum: float) -> None:
        action = "/enabletorch" if on else "/disabletorch"
        try:
            with urllib.request.urlopen(self.base + action, timeout=3) as r:
                r.read()
        except Exception as e:
            print(f"[auto-torch] LED {'on' if on else 'off'} failed: {e}")
            return
        self._on = on
        print(f"[auto-torch] LED {'ON' if on else 'OFF'} (lum={lum:.1f})")
        self._publish(state="changed")

    def _publish(self, state: str = "tick") -> None:
        HUB.publish({
            "type": "torch",
            "state": state,
            "auto": True,
            "paused": self.paused,
            "led_on": self._on,
            "luminance": self._last_lum,
        })


def _normalize_ipcam_base(s: str) -> str:
    s = (s or "").strip()
    if "://" not in s:
        s = "http://" + s
    u = urllib.parse.urlparse(s)
    host = u.hostname or ""
    port = u.port or 8080
    return f"{u.scheme}://{host}:{port}"


# Singleton used by run_demo.py; set up in start_server caller.
AUTO_TORCH: Optional[AutoTorch] = None


def start_auto_torch(camera_base: str, **kwargs: Any) -> AutoTorch:
    """Spin up the auto-torch controller. Idempotent — replaces any prior."""
    global AUTO_TORCH
    if AUTO_TORCH is not None:
        AUTO_TORCH.stop()
    AUTO_TORCH = AutoTorch(camera_base=camera_base, **kwargs)
    AUTO_TORCH.start()
    return AUTO_TORCH

# Where uploaded operator clips land. Set by ``configure_audio_dir``.
_AUDIO_DIR: pathlib.Path = pathlib.Path("/tmp")


def configure_audio_dir(path: str) -> None:
    global _AUDIO_DIR
    _AUDIO_DIR = pathlib.Path(path)
    _AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def publish_turn(turn: Any) -> None:
    """Push one agent turn to all connected browsers."""
    HUB.publish({"type": "turn", "turn": _to_jsonable(turn)})


def publish_status(message: str, **extra: Any) -> None:
    HUB.publish({"type": "status", "message": message, **extra})


def publish_reset() -> None:
    HUB.publish({"type": "reset"})


def _to_jsonable(o: Any) -> Any:
    if is_dataclass(o):
        return _to_jsonable(asdict(o))
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    return o


class _Handler(BaseHTTPRequestHandler):
    server_version = "RescueGemma/0.1"

    def log_message(self, fmt, *a):
        return

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/audio":
            self._serve_audio_upload(parsed)
            return
        self.send_error(404, "not found")

    def _serve_audio_upload(self, parsed):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self.send_error(400, "empty body")
            return
        ctype = (self.headers.get("Content-Type") or "").lower()
        ext = "webm"
        if "wav" in ctype: ext = "wav"
        elif "ogg" in ctype: ext = "ogg"
        elif "mp4" in ctype or "m4a" in ctype: ext = "m4a"
        elif "webm" in ctype: ext = "webm"
        # Optional ?ext= override
        params = urllib.parse.parse_qs(parsed.query)
        if "ext" in params:
            ext = params["ext"][0]
        data = self.rfile.read(length)
        ts = int(time.time() * 1000)
        out_path = _AUDIO_DIR / f"operator_{ts}.{ext}"
        try:
            with open(out_path, "wb") as fh:
                fh.write(data)
        except Exception as e:
            self.send_error(500, f"write failed: {e}")
            return
        TRIGGER.fire(str(out_path))
        HUB.publish({"type": "status",
                     "message": f"audio captured ({len(data)//1024} KB)"})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({
            "ok": True, "path": str(out_path), "bytes": len(data),
        }).encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._serve_index(parsed)
            return
        if path == "/events":
            self._serve_sse()
            return
        if path == "/transcript":
            self._serve_transcript()
            return
        if path == "/proxy":
            self._serve_mjpeg_proxy(parsed)
            return
        if path == "/control":
            self._serve_control(parsed)
            return
        self.send_error(404, "not found")

    # ---- index ----------------------------------------------------------

    def _serve_index(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        camera = params.get("camera", [""])[0]
        html = INDEX_HTML.replace("__CAMERA_URL__", _attr_escape(camera))
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- SSE ------------------------------------------------------------

    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        q = HUB.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                    self._write_event(ev)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.unsubscribe(q)

    def _write_event(self, ev):
        self.wfile.write(b"data: ")
        self.wfile.write(json.dumps(ev).encode("utf-8"))
        self.wfile.write(b"\n\n")
        self.wfile.flush()

    # ---- transcript snapshot --------------------------------------------

    def _serve_transcript(self):
        body = json.dumps(HUB.history_snapshot()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    # ---- MJPEG proxy (CORS fallback) ------------------------------------

    def _serve_mjpeg_proxy(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        target = params.get("url", [""])[0]
        if not target:
            self.send_error(400, "missing ?url=")
            return
        try:
            req = urllib.request.Request(target,
                                         headers={"User-Agent": "RescueGemmaProxy/0.1"})
            upstream = urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            self.send_error(502, f"upstream: {e}")
            return
        content_type = upstream.headers.get("Content-Type",
                                            "multipart/x-mixed-replace")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = upstream.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            try:
                upstream.close()
            except Exception:
                pass


    # ---- IP Webcam control proxy ----------------------------------------

    # IP Webcam (Android) exposes a simple HTTP control API on the same port
    # the MJPEG stream uses. We proxy a small whitelist of actions so the
    # browser can drive the phone (LED, focus, snapshot) without CORS.
    _IPCAM_ACTIONS = {
        "led_on":     "/enabletorch",
        "led_off":    "/disabletorch",
        "focus":      "/focus",
        "nofocus":    "/nofocus",
        "photoaf":    "/photoaf",
        "shot":       "/shot.jpg",
        "status":     "/status.json",
        "ptz_in":     "/ptz?zoom=in",
        "ptz_out":    "/ptz?zoom=out",
    }

    def _serve_control(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        action = params.get("action", [""])[0]
        base = params.get("base", [""])[0]
        if not action:
            self.send_error(400, "need ?action=")
            return
        # Local actions: control the AutoTorch instance, no IP-camera call.
        if action in ("auto_pause", "auto_resume", "auto_status"):
            self._handle_auto_action(action)
            return
        if not base:
            self.send_error(400, "need ?base= for that action")
            return
        if action not in self._IPCAM_ACTIONS:
            self.send_error(400, f"unknown action {action!r}; "
                                 f"allowed: {sorted(self._IPCAM_ACTIONS)}")
            return
        # A manual LED toggle pauses auto-mode so the user's choice sticks.
        if AUTO_TORCH is not None and action in ("led_on", "led_off"):
            AUTO_TORCH.pause()
        # Normalize base — accept "192.168.x:8080", "host:port/video", full URL.
        target = self._ipcam_base(base) + self._IPCAM_ACTIONS[action]
        try:
            req = urllib.request.Request(
                target,
                headers={"User-Agent": "RescueGemmaControl/0.1"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                body = r.read()
                upstream_ct = r.headers.get("Content-Type", "text/plain")
                status = r.getcode()
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": False, "action": action, "target": target,
                "error": str(e),
            }).encode("utf-8"))
            return
        # For non-image actions, return a tiny JSON envelope so the UI can
        # show a toast. For /shot.jpg pass the bytes through.
        if action == "shot":
            self.send_response(status)
            self.send_header("Content-Type", upstream_ct)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({
            "ok": 200 <= status < 300, "action": action,
            "status": status,
            "body": body.decode("utf-8", errors="replace")[:512],
        }).encode("utf-8"))

    def _handle_auto_action(self, action: str):
        if AUTO_TORCH is None:
            self._json({"ok": False, "error": "auto-torch not running"})
            return
        if action == "auto_pause":
            AUTO_TORCH.pause()
        elif action == "auto_resume":
            AUTO_TORCH.resume()
        self._json({"ok": True, "status": AUTO_TORCH.status()})

    def _json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _ipcam_base(s: str) -> str:
        s = s.strip()
        if "://" not in s:
            s = "http://" + s
        u = urllib.parse.urlparse(s)
        host = u.hostname or ""
        port = u.port or 8080
        return f"{u.scheme}://{host}:{port}"


def _attr_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace('"', "&quot;")
             .replace("<", "&lt;").replace(">", "&gt;"))


def start_server(port: int = 8765) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=srv.serve_forever, name="rescue-web", daemon=True)
    t.start()
    return srv


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Single-page UI. Modern dark theme, glass cards, live SSE timeline, MJPEG.
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RescueGemma · live</title>
<style>
  :root {
    --bg-0:#06080c; --bg-1:#0c1118; --bg-2:#11171f;
    --line:#1c2330; --line-hi:#283242;
    --ink:#e7ecf3; --dim:#8a93a0; --mute:#5a6471;
    --accent:#7dd3fc; --accent-2:#22d3ee;
    --good:#86efac; --warn:#fbbf24; --bad:#f87171; --speak:#fde68a;
    --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  }
  *{box-sizing:border-box}
  html,body{height:100%;margin:0}
  body{
    background:
      radial-gradient(1200px 600px at 80% -10%, rgba(34,211,238,.06), transparent 60%),
      radial-gradient(900px 500px at -10% 110%, rgba(125,211,252,.05), transparent 60%),
      var(--bg-0);
    color:var(--ink); font-family:var(--sans);
    -webkit-font-smoothing:antialiased; overflow:hidden;
  }
  /* ---------- top bar ---------- */
  header{
    display:flex; align-items:center; gap:16px;
    padding:14px 22px; border-bottom:1px solid var(--line);
    backdrop-filter: blur(8px);
    background: linear-gradient(180deg, rgba(12,17,24,.85), rgba(12,17,24,.55));
    position:relative; z-index:5;
  }
  .brand{display:flex; align-items:center; gap:10px; font-weight:600; letter-spacing:.02em}
  .brand .logo{
    width:26px; height:26px; border-radius:7px;
    background: conic-gradient(from 210deg, #22d3ee, #7dd3fc, #a78bfa, #22d3ee);
    box-shadow: 0 0 24px rgba(34,211,238,.4) inset, 0 0 18px rgba(125,211,252,.18);
    position:relative;
  }
  .brand .logo::after{
    content:""; position:absolute; inset:5px;
    border-radius:5px; background:var(--bg-0);
  }
  .brand small{color:var(--dim); font-weight:400; margin-left:6px}
  .live{
    display:inline-flex; align-items:center; gap:8px;
    padding:5px 10px 5px 8px; border-radius:999px;
    border:1px solid rgba(248,113,113,.35); background:rgba(248,113,113,.08);
    font-size:11px; font-weight:600; color:#fecaca;
    letter-spacing:.12em; text-transform:uppercase;
  }
  .live .dot{
    width:8px; height:8px; border-radius:50%; background:#f87171;
    box-shadow:0 0 0 0 rgba(248,113,113,.7); animation: pulse 1.4s infinite;
  }
  @keyframes pulse{
    0%{box-shadow:0 0 0 0 rgba(248,113,113,.55)}
    70%{box-shadow:0 0 0 10px rgba(248,113,113,0)}
    100%{box-shadow:0 0 0 0 rgba(248,113,113,0)}
  }
  .grow{flex:1}
  .url-bar{
    display:flex; align-items:center; gap:6px;
    background:var(--bg-1); border:1px solid var(--line);
    border-radius:10px; padding:4px 4px 4px 12px;
    min-width:360px;
  }
  .url-bar input{
    flex:1; background:transparent; border:0; outline:0;
    color:var(--ink); font:13px var(--mono);
  }
  .url-bar input::placeholder{color:var(--mute)}
  .btn{
    background:var(--accent); color:#06121b; border:0;
    padding:7px 14px; border-radius:8px; cursor:pointer;
    font-weight:600; font-size:12.5px; letter-spacing:.04em;
    transition:transform .05s ease, filter .15s ease;
  }
  .btn:hover{filter:brightness(1.1)}
  .btn:active{transform:translateY(1px)}
  .btn.secondary{
    background:var(--bg-2); color:var(--ink); border:1px solid var(--line);
  }
  .btn.ghost{
    background:transparent; color:var(--ink); border:1px solid var(--line);
    padding:7px 10px;
  }
  .btn.ghost:hover{ border-color: var(--line-hi) }
  .btn.torch-on{
    background: linear-gradient(180deg, #fde68a, #facc15);
    color:#3b2a06; border:1px solid rgba(0,0,0,.15);
    box-shadow: 0 0 22px rgba(250,204,21,.45), 0 0 8px rgba(250,204,21,.25);
  }
  .btn.torch-on:hover{ filter: brightness(1.05) }
  .toast{
    position:fixed; right:18px; top:64px; z-index:50;
    background: var(--bg-2); border:1px solid var(--line);
    border-radius:10px; padding:10px 14px; font-size:13px;
    color:var(--ink); box-shadow: 0 20px 50px rgba(0,0,0,.4);
    opacity:0; transform: translateY(-6px);
    transition: opacity .18s ease, transform .18s ease;
    pointer-events:none; max-width: 320px;
  }
  .toast.on{ opacity:1; transform: none }
  .toast.err{ border-color: rgba(248,113,113,.4); color:#fecaca }
  .toast.ok{ border-color: rgba(134,239,172,.4); color:#bbf7d0 }

  /* auto torch pill */
  .auto-pill{
    display:inline-flex; align-items:center; gap:6px;
    font:11.5px var(--mono); color:var(--dim); padding:6px 10px;
  }
  .auto-pill .auto-dot{
    width:8px; height:8px; border-radius:50%;
    background: var(--mute); transition: background .2s ease;
  }
  #auto-btn.armed .auto-dot{ background: var(--good); box-shadow: 0 0 12px rgba(134,239,172,.55); }
  #auto-btn.paused .auto-dot{ background: var(--mute); }
  #auto-btn.changed .auto-dot{ animation: blip .8s ease-out 1; }
  @keyframes blip{
    0%{ box-shadow: 0 0 0 0 rgba(125,211,252,.55) }
    100%{ box-shadow: 0 0 0 12px rgba(125,211,252,0) }
  }

  /* mic button + recording state */
  #mic-btn{ display:inline-flex; align-items:center; gap:8px; }
  .mic-dot{
    width:10px; height:10px; border-radius:50%;
    background: var(--mute); box-shadow: 0 0 0 0 rgba(0,0,0,0);
    transition: background .15s ease;
  }
  #mic-btn.armed .mic-dot{ background: var(--accent) }
  #mic-btn.rec{
    border-color: rgba(248,113,113,.55); background: rgba(248,113,113,.08);
    color:#fecaca;
  }
  #mic-btn.rec .mic-dot{
    background:#f87171;
    animation: micpulse .9s infinite;
  }
  @keyframes micpulse{
    0%{box-shadow:0 0 0 0 rgba(248,113,113,.55)}
    70%{box-shadow:0 0 0 9px rgba(248,113,113,0)}
    100%{box-shadow:0 0 0 0 rgba(248,113,113,0)}
  }
  .kbd{
    display:inline-block; padding:1px 5px; border-radius:4px;
    background:var(--bg-0); border:1px solid var(--line-hi);
    font:11px var(--mono); color:var(--dim);
  }
  .meta{font:12px var(--mono); color:var(--dim)}
  .meta b{color:var(--ink); font-weight:600}

  /* ---------- main split ---------- */
  main{
    display:grid; grid-template-columns: minmax(0, 1.55fr) minmax(360px, .8fr);
    gap:14px; padding:14px 18px;
    height: calc(100vh - 56px);
  }
  @media (max-width: 980px){
    main{grid-template-columns:1fr; height:auto}
  }
  .panel{
    background: linear-gradient(180deg, rgba(17,23,31,.85), rgba(12,17,24,.85));
    border:1px solid var(--line); border-radius:14px;
    overflow:hidden;
    box-shadow: 0 1px 0 rgba(255,255,255,.02) inset, 0 24px 60px rgba(0,0,0,.35);
  }

  /* ---------- video panel ---------- */
  #video-panel{
    display:flex; flex-direction:column; min-width:0;
  }
  .video-head{
    display:flex; align-items:center; gap:12px;
    padding:10px 14px; border-bottom:1px solid var(--line);
    font:12px var(--mono); color:var(--dim);
  }
  .video-head .pill{
    display:inline-flex; align-items:center; gap:6px;
    background:var(--bg-1); border:1px solid var(--line);
    padding:3px 9px; border-radius:999px; color:var(--ink);
  }
  .video-head .pill.on{ border-color: rgba(134,239,172,.4); color:#bbf7d0 }
  .video-head .pill.warn{ border-color: rgba(248,113,113,.4); color:#fecaca }
  .video-frame{
    position:relative; flex:1; min-height:240px;
    background: radial-gradient(800px 400px at 50% 20%, #0c1118 0%, #06080c 75%);
  }
  .video-frame img, .video-frame video{
    position:absolute; inset:0; width:100%; height:100%;
    object-fit:contain; background:#000;
  }
  .video-empty{
    position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    flex-direction:column; gap:10px; color:var(--mute); font:13px var(--sans);
    text-align:center; padding:0 24px;
  }
  .video-empty code{ font:12.5px var(--mono); color:var(--dim) }
  .speak-overlay{
    position:absolute; left:24px; right:24px; bottom:24px;
    padding:14px 18px; border-radius:12px;
    background: linear-gradient(180deg, rgba(0,0,0,.35), rgba(0,0,0,.65));
    border:1px solid rgba(253,230,138,.35);
    backdrop-filter: blur(10px);
    color:var(--speak); font-size:18px; line-height:1.45; font-style:italic;
    opacity:0; transform: translateY(8px); transition:opacity .25s ease, transform .25s ease;
    pointer-events:none;
  }
  .speak-overlay.on{opacity:1; transform:none}
  .speak-overlay b{
    display:inline-block; font-size:10px; letter-spacing:.18em; text-transform:uppercase;
    padding:2px 7px; border-radius:999px; color:#1a120a;
    background:var(--speak); margin-right:10px; vertical-align:middle;
    font-style:normal; font-weight:700;
  }
  .step-overlay{
    position:absolute; top:14px; left:14px;
    display:flex; align-items:center; gap:8px;
    padding:6px 10px; border-radius:999px;
    background: rgba(6,8,12,.6); border:1px solid var(--line);
    font:11.5px var(--mono); color:var(--dim);
    backdrop-filter: blur(6px);
  }
  .step-overlay b{color:var(--ink)}

  /* ---------- reasoning panel ---------- */
  #reason-panel{
    display:flex; flex-direction:column; min-height:0;
  }
  .r-head{
    padding:12px 16px; border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px;
  }
  .r-head h2{
    margin:0; font-size:12px; letter-spacing:.16em; text-transform:uppercase;
    color:var(--dim); font-weight:600;
  }
  .r-head .count{
    margin-left:auto; font:12px var(--mono); color:var(--dim);
  }
  #intent{
    padding:10px 16px 14px; border-bottom:1px solid var(--line);
    font-size:13.5px; line-height:1.5; color:#cbd5e1;
  }
  #intent .label{
    display:block; font:10.5px var(--mono); color:var(--dim);
    letter-spacing:.14em; text-transform:uppercase; margin-bottom:4px;
  }
  #intent.empty{color:var(--mute)}
  #timeline{
    flex:1; overflow-y:auto; padding:6px 4px 16px;
    scroll-behavior:smooth;
  }
  #timeline::-webkit-scrollbar{width:8px}
  #timeline::-webkit-scrollbar-thumb{background:var(--line-hi); border-radius:4px}
  .turn{
    margin:8px 12px; padding:12px 14px;
    background:var(--bg-2); border:1px solid var(--line); border-radius:12px;
    opacity:0; transform: translateY(6px);
    transition: opacity .22s ease, transform .22s ease, border-color .22s ease;
    min-width:0; overflow-wrap:anywhere;
  }
  .turn > *{ min-width:0; }
  .turn.on{opacity:1; transform:none}
  .turn.live{ border-color: rgba(125,211,252,.45); box-shadow: 0 0 0 3px rgba(125,211,252,.08) }
  .turn .meta-row{
    display:flex; align-items:center; gap:8px; flex-wrap:wrap;
    margin-bottom:6px;
  }
  .step-tag{
    font:11px var(--mono); color:var(--dim);
    background:var(--bg-1); border:1px solid var(--line);
    padding:2px 7px; border-radius:999px;
  }
  .badge{
    display:inline-block; padding:2px 8px; border-radius:999px;
    font:10.5px var(--sans); font-weight:600;
    letter-spacing:.08em; text-transform:uppercase;
    background:var(--bg-1); color:var(--ink);
    border:1px solid var(--line);
    white-space:nowrap; flex-shrink:0;
  }
  .step-tag{ white-space:nowrap; flex-shrink:0; }
  .turn-frame{
    margin:8px 0 0; display:block; width:100%;
    max-height:240px; border-radius:8px; border:1px solid var(--line);
    background:#000; object-fit:cover;
  }
  .badge.audio   { background:#0f243a; color:#bfe1ff; border-color:#1e3a5c }
  .badge.vision  { background:#231a3a; color:#d6c0ff; border-color:#3a2a5c }
  .badge.tone-urgent          { background:#3a1717; color:#ffd2d2; border-color:#5c1d1d }
  .badge.tone-calm_directive  { background:#143524; color:#c8f5d6; border-color:#1d4f30 }
  .badge.tone-reassuring      { background:#3a2e14; color:#ffe9b0; border-color:#5c4a1d }
  .badge.follow-lagging       { background:#3a2e14; color:#ffe9b0; border-color:#5c4a1d }
  .badge.follow-following     { background:#143524; color:#c8f5d6; border-color:#1d4f30 }
  .badge.fail                 { background:#3a1717; color:#ffd2d2; border-color:#5c1d1d }
  .thought{
    font-size:13.5px; line-height:1.5; color:var(--ink); margin:2px 0 8px;
  }
  .call{
    font-family:var(--mono); font-size:12.5px; line-height:1.55;
    background:var(--bg-0); border:1px solid var(--line); border-radius:8px;
    padding:8px 12px; color:var(--good);
    white-space:pre-wrap; overflow-wrap:anywhere; word-break:normal;
    max-width:100%; min-width:0;
    tab-size:2;
  }
  .call .arg-line{ display:block; padding-left: 1.4ch; }
  .call .name{ color:var(--accent) }
  .call .arg-k{ color:#a5b4fc }
  .call .arg-v{ color:#fde68a }
  .spoke{
    margin-top:8px; padding:9px 12px; border-left:3px solid var(--speak);
    background: rgba(253,230,138,.04); color:var(--speak);
    font-size:13.5px; line-height:1.5; font-style:italic; border-radius:0 8px 8px 0;
  }
  .obs{
    margin-top:8px; font:12px var(--sans); line-height:1.55;
    color:var(--dim);
  }
  .obs b{color:var(--ink)}
  .finding{
    margin-top:6px; font-size:12.5px; color:var(--warn);
  }
  .empty-state{
    padding:30px 18px; color:var(--mute); text-align:center;
    font-size:13px; line-height:1.5;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="logo"></span>
    RescueGemma <small>· live</small>
  </div>
  <span class="live"><span class="dot"></span>live</span>
  <div class="grow"></div>
  <div class="url-bar">
    <input id="camera-url" type="text" placeholder="http://192.168.x.x:8080/video  (IP Webcam / MJPEG URL)">
    <button id="connect-btn" class="btn">Connect</button>
  </div>
  <button id="torch-btn" class="btn ghost" title="Toggle phone LED (IP Webcam) — manual toggle pauses auto">
    <span id="torch-icon">🔦</span> <span id="torch-label">LED</span>
  </button>
  <button id="auto-btn" class="btn ghost auto-pill" title="Auto-toggle LED from frame luminance">
    <span class="auto-dot"></span> <span id="auto-label">auto</span> · <span id="auto-lum">—</span>
  </button>
  <button id="mic-btn" class="btn ghost" title="Hold to speak (operator voice in)">
    <span id="mic-dot" class="mic-dot"></span><span id="mic-label">Hold to speak</span>
  </button>
  <div class="meta"><span id="conn-status">waiting…</span></div>
</header>

<main>
  <section id="video-panel" class="panel">
    <div class="video-head">
      <span class="pill" id="cam-pill">camera: <b id="cam-state">not connected</b></span>
      <span class="pill on"><span id="sse-state">agent stream</span></span>
      <span style="flex:1"></span>
      <span id="ts" class="meta"></span>
    </div>
    <div class="video-frame" id="video-frame">
      <div class="step-overlay">step <b id="step-now">—</b> · <span id="seg-now">awaiting takeoff</span></div>
      <div class="video-empty" id="video-empty">
        <div style="font-size:14px; color:var(--ink); font-weight:500">
          Paste your phone's MJPEG URL above and hit Connect
        </div>
        <div>Most IP-camera apps expose this as <code>http://&lt;phone-ip&gt;:8080/video</code>.</div>
        <div>If the browser blocks it (mixed content / CORS), tick <label style="cursor:pointer"><input type="checkbox" id="use-proxy" style="vertical-align:middle"> use server-side proxy</label>.</div>
      </div>
      <img id="video-img" style="display:none" alt="">
      <div class="speak-overlay" id="speak-overlay">
        <b id="speak-tone">CALM</b>
        <span id="speak-text"></span>
      </div>
    </div>
  </section>

  <section id="reason-panel" class="panel">
    <div class="r-head">
      <h2>Gemma 4 · Reasoning</h2>
      <span class="count" id="turn-count">0 turns</span>
    </div>
    <div id="intent" class="empty">
      <span class="label">Operator intent</span>
      <span id="intent-text">awaiting audio…</span>
    </div>
    <div id="timeline">
      <div class="empty-state" id="empty-state">
        No agent turns yet. Start the mission with<br>
        <code style="color:var(--dim); font-family:var(--mono)">python run_demo.py --backend ollama --web</code>
      </div>
    </div>
  </section>
</main>

<div id="toast" class="toast"></div>

<script>
(function(){
  const CAMERA_URL_BOOT = "__CAMERA_URL__";
  const $ = (id) => document.getElementById(id);
  const VISION_TOOLS = new Set(["scout_ahead","explore_room","hover_and_scan","goto_waypoint","lead_to_exit"]);

  // ---------------- toast ----------------
  const toast = $("toast");
  let toastTimer = null;
  function showToast(msg, kind){
    toast.className = "toast on " + (kind || "");
    toast.textContent = msg;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("on"), 2200);
  }

  // ---------------- camera ----------------
  const camInput = $("camera-url");
  const camBtn = $("connect-btn");
  const camPill = $("cam-pill");
  const camState = $("cam-state");
  const camImg = $("video-img");
  const camEmpty = $("video-empty");
  const useProxy = $("use-proxy");

  function setCamState(text, kind){
    camState.textContent = text;
    camPill.classList.remove("on","warn");
    if (kind) camPill.classList.add(kind);
  }

  function connectCamera(url){
    if (!url) { setCamState("not connected", null); camImg.style.display="none"; camEmpty.style.display="flex"; return; }
    const src = useProxy.checked
      ? `/proxy?url=${encodeURIComponent(url)}`
      : url;
    camImg.onload = () => { setCamState("streaming", "on"); camEmpty.style.display="none"; camImg.style.display="block"; };
    camImg.onerror = () => { setCamState("error · try proxy", "warn"); };
    setCamState("connecting…", null);
    camImg.src = src + (src.includes("?") ? "&" : "?") + "_t=" + Date.now();
  }

  camBtn.onclick = () => connectCamera(camInput.value.trim());
  camInput.addEventListener("keydown", e => { if (e.key === "Enter") camBtn.click(); });
  if (CAMERA_URL_BOOT) { camInput.value = CAMERA_URL_BOOT; connectCamera(CAMERA_URL_BOOT); }

  // ---------------- torch (IP Webcam LED) ----------------
  const torchBtn = $("torch-btn");
  const torchLabel = $("torch-label");
  let torchOn = false;
  let torchBusy = false;

  function cameraBase(){
    const raw = (camInput.value || "").trim();
    if (!raw) return "";
    // Strip path & query; keep host:port.
    try {
      const u = new URL(raw.includes("://") ? raw : "http://" + raw);
      return `${u.protocol}//${u.hostname}:${u.port || "8080"}`;
    } catch { return ""; }
  }

  async function setTorch(on){
    const base = cameraBase();
    if (!base){ showToast("Set the camera URL first", "err"); return; }
    if (torchBusy) return;
    torchBusy = true;
    const action = on ? "led_on" : "led_off";
    try {
      const r = await fetch(`/control?action=${action}&base=${encodeURIComponent(base)}`);
      const j = await r.json();
      if (j.ok){
        torchOn = on;
        torchBtn.classList.toggle("torch-on", on);
        torchLabel.textContent = on ? "LED on" : "LED";
        showToast(on ? "Phone LED on" : "Phone LED off", "ok");
      } else {
        showToast(`LED toggle failed (${j.status || "??"}): ${j.error || j.body || ""}`.slice(0,160), "err");
      }
    } catch (e){
      showToast("LED toggle network error", "err");
    } finally {
      torchBusy = false;
    }
  }
  torchBtn.onclick = () => setTorch(!torchOn);

  // ---------------- auto torch state pill ----------------
  const autoBtn = $("auto-btn");
  const autoLabel = $("auto-label");
  const autoLum = $("auto-lum");

  function paintAutoPill(s){
    autoBtn.classList.remove("armed","paused","changed");
    if (s.paused){
      autoBtn.classList.add("paused");
      autoLabel.textContent = "auto · off";
    } else {
      autoBtn.classList.add("armed");
      autoLabel.textContent = "auto · " + (s.led_on === true ? "LED on" : s.led_on === false ? "LED off" : "watching");
    }
    if (typeof s.luminance === "number") autoLum.textContent = "lum " + s.luminance.toFixed(0);
    if (typeof s.led_on === "boolean") {
      torchOn = !!s.led_on;
      torchBtn.classList.toggle("torch-on", torchOn);
      torchLabel.textContent = torchOn ? "LED on" : "LED";
    }
  }

  async function fireAutoAction(act){
    try {
      const r = await fetch(`/control?action=${act}`);
      const j = await r.json();
      if (j.ok && j.status) paintAutoPill(j.status);
    } catch {}
  }
  autoBtn.onclick = () => {
    // Click toggles pause/resume.
    if (autoBtn.classList.contains("paused")) fireAutoAction("auto_resume");
    else fireAutoAction("auto_pause");
  };

  // ---------------- operator voice in (push-to-talk on SPACE) ----------------
  const micBtn = $("mic-btn");
  const micLabel = $("mic-label");
  let mediaStream = null;
  let recorder = null;
  let recordedChunks = [];
  let isRecording = false;
  let armRequested = false;

  function setMicState(s){
    micBtn.classList.remove("armed","rec");
    if (s === "armed"){ micBtn.classList.add("armed"); micLabel.innerHTML = 'Hold <span class="kbd">Space</span>'; }
    else if (s === "rec"){ micBtn.classList.add("rec"); micLabel.textContent = "Recording…"; }
    else { micLabel.innerHTML = 'Hold <span class="kbd">Space</span> to speak'; }
  }
  setMicState("idle");

  async function ensureStream(){
    if (mediaStream) return mediaStream;
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
      });
      setMicState("armed");
      return mediaStream;
    } catch (e){
      showToast("Mic permission denied", "err");
      throw e;
    }
  }

  function pickMime(){
    const cands = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const m of cands){
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
    }
    return "";
  }

  async function startRecording(){
    if (isRecording) return;
    try { await ensureStream(); } catch { return; }
    recordedChunks = [];
    const mime = pickMime();
    try {
      recorder = mime ? new MediaRecorder(mediaStream, { mimeType: mime })
                      : new MediaRecorder(mediaStream);
    } catch (e){
      showToast("MediaRecorder unsupported in this browser", "err");
      return;
    }
    recorder.ondataavailable = (ev) => { if (ev.data && ev.data.size) recordedChunks.push(ev.data); };
    recorder.onstop = onRecordingStop;
    isRecording = true;
    setMicState("rec");
    recorder.start(100); // 100 ms timeslices keep latency low
  }

  function stopRecording(){
    if (!isRecording || !recorder) return;
    isRecording = false;
    setMicState("armed");
    try { recorder.stop(); } catch {}
  }

  async function onRecordingStop(){
    if (!recordedChunks.length){
      showToast("No audio captured", "err");
      return;
    }
    const blob = new Blob(recordedChunks, { type: recorder.mimeType || "audio/webm" });
    if (blob.size < 1024){
      showToast("Clip too short", "err");
      return;
    }
    showToast(`Uploading ${(blob.size/1024).toFixed(1)} KB…`, "ok");
    const ext = (recorder.mimeType || "").includes("ogg") ? "ogg"
              : (recorder.mimeType || "").includes("mp4") ? "m4a"
              : "webm";
    try {
      const r = await fetch(`/audio?ext=${ext}`, {
        method: "POST",
        headers: { "Content-Type": recorder.mimeType || "audio/webm" },
        body: blob,
      });
      const j = await r.json();
      if (j.ok){
        showToast("Audio sent · mission armed", "ok");
        // Best-effort: light up the intent box so the user sees it landed.
        intentBox.classList.remove("empty");
        intentText.textContent = "(transcribing…)";
      } else {
        showToast("Upload failed", "err");
      }
    } catch (e){
      showToast("Upload network error", "err");
    }
  }

  // SPACE = push-to-talk. Ignore if user is typing in an input.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space") return;
    const a = document.activeElement;
    if (a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA")) return;
    if (e.repeat) return;
    e.preventDefault();
    startRecording();
  });
  document.addEventListener("keyup", (e) => {
    if (e.code !== "Space") return;
    const a = document.activeElement;
    if (a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA")) return;
    e.preventDefault();
    stopRecording();
  });
  // Click-and-hold on the button works too (mouse + touch).
  micBtn.addEventListener("mousedown", (e) => { e.preventDefault(); startRecording(); });
  micBtn.addEventListener("mouseup",   (e) => { e.preventDefault(); stopRecording();  });
  micBtn.addEventListener("mouseleave",(e) => { if (isRecording) stopRecording(); });
  micBtn.addEventListener("touchstart",(e) => { e.preventDefault(); startRecording(); }, {passive:false});
  micBtn.addEventListener("touchend",  (e) => { e.preventDefault(); stopRecording();  }, {passive:false});
  // Pre-arm the mic on first user gesture so the first real recording is instant.
  document.addEventListener("click", () => { ensureStream().catch(()=>{}); }, { once: true });

  // ---------------- SSE / reasoning ----------------
  const tl = $("timeline");
  const empty = $("empty-state");
  const intentBox = $("intent");
  const intentText = $("intent-text");
  const speakOverlay = $("speak-overlay");
  const speakText = $("speak-text");
  const speakTone = $("speak-tone");
  const stepNow = $("step-now");
  const segNow = $("seg-now");
  const turnCountEl = $("turn-count");
  const sseState = $("sse-state");
  const tsEl = $("ts");

  setInterval(() => tsEl.textContent = new Date().toLocaleTimeString(), 1000);

  let turns = [];
  let speakTimer = null;

  function escapeHTML(s){
    return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  }
  function renderArg(k, v){
    const val = typeof v === "string"
      ? `"${escapeHTML(v)}"`
      : escapeHTML(JSON.stringify(v));
    return `<span class="arg-k">${escapeHTML(k)}</span>=<span class="arg-v">${val}</span>`;
  }
  function callHTML(call){
    if (!call) return "";
    const argsObj = call.args || call.arguments || {};
    const entries = Object.entries(argsObj);
    const name = `<span class="name">${escapeHTML(call.name)}</span>`;
    if (entries.length === 0) return `${name}()`;
    // Estimate single-line width; if too long, switch to multi-line.
    const flat = entries.map(([k,v]) =>
      k + "=" + (typeof v === "string" ? `"${v}"` : JSON.stringify(v))).join(", ");
    if (flat.length <= 56 && entries.length <= 2) {
      return `${name}(${entries.map(([k,v]) => renderArg(k,v)).join(", ")})`;
    }
    // Multi-line, one arg per line, indented.
    const body = entries.map(([k,v]) =>
      `<span class="arg-line">${renderArg(k,v)},</span>`).join("");
    return `${name}(\n${body})`;
  }

  function showSpoken(text, tone){
    if (!text){ speakOverlay.classList.remove("on"); return; }
    speakText.textContent = `"${text}"`;
    speakTone.textContent = (tone || "speak").replace("_"," ");
    speakOverlay.classList.add("on");
    if (speakTimer) clearTimeout(speakTimer);
    speakTimer = setTimeout(() => speakOverlay.classList.remove("on"), 6500);
  }

  function renderTurn(t, opts = {live:false}){
    const el = document.createElement("div");
    el.className = "turn" + (opts.live ? " live" : "");
    const args = t.call?.args || t.call?.arguments || {};
    const isAudio = t.step === 1;
    const isVision = !!t.frame || VISION_TOOLS.has(t.call?.name);
    const badges = [];
    if (isAudio)  badges.push('<span class="badge audio">audio in</span>');
    if (isVision) badges.push('<span class="badge vision">vision in</span>');
    if (!t.ok)    badges.push('<span class="badge fail">refused</span>');
    if (t.tone)   badges.push(`<span class="badge tone-${escapeHTML(t.tone)}">${escapeHTML(t.tone.replace("_"," "))}</span>`);
    if (t.follow_state) badges.push(`<span class="badge follow-${escapeHTML(t.follow_state)}">${escapeHTML(t.follow_state)}</span>`);
    const speak = t.spoken ? `<div class="spoke">"${escapeHTML(t.spoken)}"</div>` : "";
    const finding = t.finding ? `<div class="finding">finding → ${escapeHTML(t.finding.type)} @ ${escapeHTML(t.finding.location)}</div>` : "";

    el.innerHTML = `
      <div class="meta-row">
        <span class="step-tag">step ${escapeHTML(t.step)}</span>
        ${badges.join(" ")}
      </div>
      <div class="thought">${escapeHTML(t.thought || "(no thought)")}</div>
      <div class="call">${callHTML(t.call)}</div>
      ${speak}
      <div class="obs"><b>obs:</b> ${escapeHTML(t.observation || "")}</div>
      ${finding}
    `;
    tl.appendChild(el);
    requestAnimationFrame(() => el.classList.add("on"));
    el.scrollIntoView({behavior: "smooth", block: "end"});

    // sidebars
    stepNow.textContent = t.step;
    if (t.segment) segNow.textContent = t.segment;
    if (t.spoken) showSpoken(t.spoken, t.tone);
  }

  function ingestSnapshot(snap){
    // clear existing
    tl.querySelectorAll(".turn").forEach(n => n.remove());
    empty.style.display = snap.turns?.length ? "none" : "block";
    turns = snap.turns || [];
    if (snap.meta?.intent) { intentBox.classList.remove("empty"); intentText.textContent = snap.meta.intent; }
    turns.forEach(t => renderTurn(t));
    turnCountEl.textContent = `${turns.length} turn${turns.length===1?"":"s"}`;
  }

  function ingestTurn(t){
    empty.style.display = "none";
    // strip 'live' off the previous most-recent
    tl.querySelectorAll(".turn.live").forEach(n => n.classList.remove("live"));
    turns.push(t);
    renderTurn(t, {live:true});
    turnCountEl.textContent = `${turns.length} turn${turns.length===1?"":"s"}`;
  }

  function connectSSE(){
    sseState.textContent = "agent stream · connecting";
    const es = new EventSource("/events");
    es.onopen = () => sseState.textContent = "agent stream · live";
    es.onerror = () => sseState.textContent = "agent stream · reconnecting";
    es.onmessage = (ev) => {
      let data; try { data = JSON.parse(ev.data); } catch { return; }
      if (data.type === "snapshot"){
        ingestSnapshot(data);
      } else if (data.type === "turn"){
        ingestTurn(data.turn);
      } else if (data.type === "status"){
        if (data.intent){ intentBox.classList.remove("empty"); intentText.textContent = data.intent; }
        if (data.message){ sseState.textContent = "agent stream · " + data.message; }
      } else if (data.type === "torch"){
        paintAutoPill(data);
        if (data.state === "changed") {
          autoBtn.classList.add("changed");
          setTimeout(() => autoBtn.classList.remove("changed"), 900);
          showToast(data.led_on ? "Auto: LED on (dark)" : "Auto: LED off (bright)", "ok");
        }
      } else if (data.type === "reset"){
        tl.querySelectorAll(".turn").forEach(n => n.remove());
        turns = []; turnCountEl.textContent = "0 turns";
        empty.style.display = "block";
        intentBox.classList.add("empty"); intentText.textContent = "awaiting audio…";
        stepNow.textContent = "—"; segNow.textContent = "awaiting takeoff";
        speakOverlay.classList.remove("on");
      }
    };
  }
  connectSSE();
  $("conn-status").textContent = "ready · " + location.host;
})();
</script>
</body>
</html>
"""
