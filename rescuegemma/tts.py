"""
RescueGemma — low-latency TTS for voice-out (system -> Manu).

Voice IN (operator -> system) is Gemma 4 E4B's native audio encoder.
Voice OUT (system -> trapped person) is this module: whenever the agent
fires ``speak_to_survivor``, the utterance is spoken aloud, in a
background thread, so the agent loop is never blocked.

Backends, picked in order:

1. **MacSayTTS** (default on macOS). Uses Apple's built-in ``say``
   command, which is a thin shell over AVSpeechSynthesizer. Zero install,
   on-device, ~100-250 ms to first audio on Apple Silicon. Voice can be
   any installed system voice (run ``say -v ?`` to list).
2. **PiperTTS** (optional upgrade). If ``piper-tts`` is importable and a
   voice model path is configured, uses Piper for neural-quality speech.
   Still local, still fast, just better.
3. **NullTTS** (fallback). No-op; logs the line that would have been
   spoken. Used on Linux/Windows without an explicit backend, or when
   the user passes ``--tts off``.

Calls are queued. If a new line arrives while the previous is still
playing we interrupt: a rescue agent should not say sentence N over
sentence N-1. The lag-recovery path also depends on the latest line
being the audible one.
"""

from __future__ import annotations

import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Optional


class _Backend:
    name = "null"
    def speak(self, text: str) -> None: ...
    def stop(self) -> None: ...


class NullTTS(_Backend):
    name = "null"

    def __init__(self, log: bool = True):
        self._log = log

    def speak(self, text: str) -> None:
        if self._log:
            print(f"[tts:null] would speak → {text}")


class MacSayTTS(_Backend):
    """Drives macOS ``say``. One subprocess per utterance; we kill any
    in-flight process when a new utterance arrives so the agent never
    stacks lines on top of each other."""

    name = "macos-say"

    # Picked for an emergency rescue tone: calm, clear, authoritative.
    # Order:
    #   1. Premium / Enhanced voices — genuinely Siri-quality if installed.
    #   2. Samantha — the classic US female; calm, reliable, well-tested.
    #   3. Daniel — calm British male, good "pilot voice" feel.
    #   4. The newer "personality" voices (Reed/Sandy/Flo) — last, because
    #      they're divisive on this user's ear. Available via --tts-voice
    #      override for anyone who likes them.
    _PREFERRED = [
        # Premium voices (need to be downloaded in System Settings).
        "Ava (Premium)", "Allison (Premium)", "Zoe (Premium)",
        "Evan (Premium)", "Tom (Premium)", "Samantha (Premium)",
        "Ava (Enhanced)", "Allison (Enhanced)", "Samantha (Enhanced)",
        # Classic stock voices — calm and natural enough for the rescue tone.
        "Samantha", "Daniel", "Karen", "Moira",
        # Newer "personality" voices — fallback only.
        "Sandy (English (US))", "Flo (English (US))",
        "Reed (English (US))", "Reed (English (UK))",
    ]

    def __init__(self, voice: str = "auto", rate_wpm: int = 175):
        # rate 175 wpm — clearer for a frightened person under smoke, and
        # close to a calm directive cadence.
        self.voice = voice if voice and voice != "auto" else self._auto_pick()
        self.rate = rate_wpm
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        print(f"[tts:say] voice = {self.voice!r}")

    @classmethod
    def _auto_pick(cls) -> str:
        try:
            out = subprocess.check_output(["say", "-v", "?"],
                                          stderr=subprocess.DEVNULL,
                                          timeout=3).decode("utf-8", "replace")
        except Exception:
            return "Samantha"
        # Parse "Voice Name           lang_TAG    # Hello! ..."
        installed: list[str] = []
        for line in out.splitlines():
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue
            # Split into the name part vs the locale comment. The locale
            # column starts with en_/de_/etc. — name is everything before
            # the locale token.
            parts = line.split()
            # Find the locale index (matches lang_REGION)
            loc_i = None
            for i, p in enumerate(parts):
                if len(p) >= 5 and p[2] == "_":
                    loc_i = i; break
            if loc_i is None:
                continue
            name = " ".join(parts[:loc_i]).strip()
            if name:
                installed.append(name)
        installed_set = set(installed)
        for pref in cls._PREFERRED:
            if pref in installed_set:
                return pref
        return "Samantha"

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.stop()
        cmd = ["say", "-v", self.voice, "-r", str(self.rate), text]
        try:
            with self._lock:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        except FileNotFoundError:
            print("[tts:say] `say` binary not found — falling back to log only.")
            print(f"[tts:say] would speak → {text}")
        except Exception as e:
            print(f"[tts:say] launch failed: {e}")

    def stop(self) -> None:
        with self._lock:
            p = self._proc
            self._proc = None
        if p is None:
            return
        try:
            p.send_signal(signal.SIGTERM)
        except Exception:
            pass


class KokoroTTS(_Backend):
    """Neural TTS via Kokoro-82M (mlx-audio on Apple Silicon). Voice
    quality is in a different league from ``say`` — natural, expressive,
    multi-voice. Uses Metal under the hood via MLX.

    Synthesizes a sentence to a numpy PCM array, then plays it through
    ``afplay`` (macOS) or ``aplay`` (Linux) so the agent loop is not
    blocked. New utterances pre-empt the in-flight clip.

    Voices: ``af_heart`` (warm female, default), ``af_bella``,
    ``af_sarah``, ``am_michael``, ``am_adam``, ``bf_emma``, ``bm_lewis``.
    For an emergency-rescue agent talking to a frightened person, the
    warm voices (``af_heart``, ``bf_emma``) work best.
    """

    name = "kokoro"

    def __init__(self,
                 voice: str = "af_heart",
                 speed: float = 1.05,
                 model_repo: str = "mlx-community/Kokoro-82M-bf16"):
        try:
            from mlx_audio.tts.generate import load_model  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Kokoro TTS needs mlx-audio. Run rescuegemma from the parlor "
                f"venv (it already has it). Original error: {e}")
        try:
            import soundfile  # type: ignore  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"soundfile required for Kokoro PCM playback: {e}")
        self.voice = voice
        self.speed = speed
        print(f"[tts:kokoro] loading {model_repo} (voice={voice})")
        self._model = load_model(model_repo)
        self.sample_rate = getattr(self._model, "sample_rate", 24000)
        # Warmup so the first real call doesn't take a model-init hit.
        try:
            list(self._model.generate(text="Ready.", voice=voice, speed=1.0))
        except Exception:
            pass
        self._lock = threading.Lock()
        self._player: Optional[subprocess.Popen] = None
        self._player_cmd = "afplay" if sys.platform == "darwin" else "aplay"
        if not shutil.which(self._player_cmd):
            raise RuntimeError(f"audio player {self._player_cmd!r} not on PATH")

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.stop()
        print(f"[tts:kokoro] 🔊 PLAYING: {text!r}")
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
        import tempfile
        try:
            results = list(self._model.generate(
                text=text, voice=self.voice, speed=self.speed))
            pcm = np.concatenate([np.array(r.audio) for r in results])
        except Exception as e:
            print(f"[tts:kokoro] synth failed: {e}")
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            wav_path = fh.name
        try:
            sf.write(wav_path, pcm, self.sample_rate)
        except Exception as e:
            print(f"[tts:kokoro] wav write failed: {e}")
            return
        # Capture player stderr so silent audio failures (muted output,
        # missing device, codec issues) actually surface.
        try:
            with self._lock:
                self._player = subprocess.Popen(
                    [self._player_cmd, wav_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            # Briefly poll the player; if it exited non-zero very fast,
            # surface the error.
            try:
                rc = self._player.wait(timeout=0.4)
                err = (self._player.stderr.read() or b"").decode("utf-8", "replace") if self._player.stderr else ""
                if rc != 0:
                    print(f"[tts:kokoro] {self._player_cmd} exited rc={rc} "
                          f"immediately: {err.strip()!r} — try `say -v ?` "
                          f"and check System Settings → Sound output.")
            except subprocess.TimeoutExpired:
                # Still playing — that's what we want; leave it alone.
                pass
        except Exception as e:
            print(f"[tts:kokoro] play failed: {e}")

    def stop(self) -> None:
        with self._lock:
            p = self._player
            self._player = None
        if p:
            try: p.send_signal(signal.SIGTERM)
            except Exception: pass


class PiperTTS(_Backend):
    """Neural TTS via piper-tts. Optional; only used if the package and a
    voice model are present. Streams to a system audio sink via afplay
    on macOS or aplay on Linux for the lowest possible latency.
    """

    name = "piper"

    def __init__(self, model_path: str):
        try:
            from piper import PiperVoice  # type: ignore
        except Exception as e:
            raise RuntimeError(f"piper-tts not importable: {e}")
        self._voice = PiperVoice.load(model_path)
        self._lock = threading.Lock()
        self._player_proc: Optional[subprocess.Popen] = None
        self._player = "afplay" if sys.platform == "darwin" else "aplay"
        if not shutil.which(self._player):
            raise RuntimeError(f"audio player {self._player!r} not on PATH")

    def speak(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.stop()
        import tempfile, wave
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            wav_path = fh.name
        with wave.open(wav_path, "wb") as wf:
            self._voice.synthesize(text, wf)
        try:
            with self._lock:
                self._player_proc = subprocess.Popen(
                    [self._player, wav_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            print(f"[tts:piper] play failed: {e}")

    def stop(self) -> None:
        with self._lock:
            p = self._player_proc
            self._player_proc = None
        if p:
            try: p.send_signal(signal.SIGTERM)
            except Exception: pass


# ---------------------------------------------------------------------------
# Public API: a queue-backed, non-blocking voice for the agent loop.
# ---------------------------------------------------------------------------

class Voice:
    """Thread-safe TTS facade. ``speak()`` returns immediately; a worker
    thread drains a queue and drives the selected backend. New utterances
    interrupt the in-flight one (latest line wins).
    """

    def __init__(self, backend: _Backend, queue_max: int = 4,
                 startup_probe: bool = True):
        self._backend = backend
        self._q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="rescue-tts",
                                   daemon=True)
        self._t.start()
        if startup_probe:
            print("[tts] 🔊 audio probe — you should hear 'voice ready' now. "
                  "If you do not, check macOS System Settings → Sound → "
                  "Output, and verify volume is unmuted.")
            try:
                # Synthesize a short test phrase synchronously so we surface
                # any errors before the mission begins.
                self._backend.speak("Voice ready.")
            except Exception as e:
                print(f"[tts] startup probe failed: {e}")
        # Tone overrides we apply on the way in (see ``speak`` below).
        self._tone_rate = {
            "urgent":         200,   # faster, sharper
            "calm_directive": 175,
            "reassuring":     155,   # slower, softer
        }

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "name", "unknown")

    def speak(self, text: str, tone: str | None = None) -> None:
        if not text:
            return
        # Drop any older queued lines; only the most recent matters.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        # Optional per-tone rate override (only meaningful for MacSayTTS).
        if isinstance(self._backend, MacSayTTS) and tone in self._tone_rate:
            self._backend.rate = self._tone_rate[tone]
        try:
            self._q.put_nowait(text)
        except queue.Full:
            # Worst-case: drop. Better than blocking the agent loop.
            print("[tts] queue full; dropped utterance")

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._backend.stop()
        except Exception:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                return
            try:
                self._backend.speak(text)
            except Exception as e:
                print(f"[tts] backend speak error: {e}")
            # No explicit wait here — speak() is non-blocking on `say` /
            # piper; the next utterance will pre-empt the audio via stop().


def make_voice(mode: str = "auto",
               voice_name: str = "auto",
               rate_wpm: int = 175,
               piper_model: str | None = None,
               kokoro_voice: str = "af_heart") -> Voice:
    """Build a Voice with the best available backend for the host.

    mode:
      * ``"auto"``  — Kokoro if available, else Piper if model provided,
                      else macOS ``say``, else null.
      * ``"kokoro"`` — neural Kokoro via mlx-audio (best quality on Mac).
      * ``"say"``   — force macOS ``say``.
      * ``"piper"`` — require Piper + ``piper_model``.
      * ``"off"`` / ``"null"`` — log-only, no sound.
    """
    mode = (mode or "auto").lower()
    if mode in ("off", "null"):
        return Voice(NullTTS(log=True))
    if mode == "kokoro":
        try:
            return Voice(KokoroTTS(voice=kokoro_voice))
        except Exception as e:
            print(f"[tts] Kokoro unavailable ({e}); falling back to null.")
            return Voice(NullTTS())
    if mode == "piper":
        if not piper_model:
            print("[tts] --tts piper requires --piper-model; falling back to null.")
            return Voice(NullTTS())
        return Voice(PiperTTS(piper_model))
    if mode == "say":
        if sys.platform != "darwin" or not shutil.which("say"):
            print("[tts] macOS `say` unavailable; falling back to null.")
            return Voice(NullTTS())
        return Voice(MacSayTTS(voice=voice_name, rate_wpm=rate_wpm))
    # auto: prefer Kokoro (genuinely natural), then Piper, then say.
    try:
        backend = KokoroTTS(voice=kokoro_voice)
        return Voice(backend)
    except Exception as e:
        print(f"[tts:auto] Kokoro unavailable ({e}); trying Piper.")
    if piper_model:
        try:
            return Voice(PiperTTS(piper_model))
        except Exception as e:
            print(f"[tts:auto] Piper unavailable ({e}); trying macOS say.")
    if sys.platform == "darwin" and shutil.which("say"):
        return Voice(MacSayTTS(voice=voice_name, rate_wpm=rate_wpm))
    print("[tts:auto] no on-device TTS available; using null backend.")
    return Voice(NullTTS())
