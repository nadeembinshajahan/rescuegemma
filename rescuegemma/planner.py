"""
RescueGemma — Gemma 4 planner + closed-loop agent.

This is the part the hackathon actually judges: an audio-in, vision-in,
function-calling agent that turns a panicked human sentence into a grounded,
multi-step, perception-driven rescue mission — running fully on-device.

### One model: Gemma 4 E4B

The agent's entire intelligence is **Gemma 4 E4B** — the 4 B-effective
edge variant with **native audio input**, **native vision input**, and
**native tool calling** in a single model. One model. One forward path
per turn. No external ASR pipeline, no separate VLM, no caption model,
no glue code that fails first under field conditions.

E4B was designed for on-device deployment: the audio encoder is 50%
smaller than Gemma 3n's and runs at 40 ms frames, the vision tower is
built in, and tool calling is part of the same checkpoint. At Q4 it
weighs ~9.6 GB — fits comfortably on a Mac, fits on a Jetson Orin Nano.

### Pipeline (one mission, one model)

  1. AUDIO IN .... operator voice clip -> Gemma 4 E4B transcribes +
                   extracts intent in a single forward pass.
  2. PLAN ........ Gemma 4 E4B reads the building's last observation +
                   attached frame and reasons about what to do next.
  3. CALL ........ Gemma 4 E4B emits a typed tool call from schemas.TOOLS.
  4. EXECUTE ..... executor (sim or real) runs it, returns observation
                   + a real camera frame.
  5. VISION IN ... Gemma 4 E4B ingests the frame next turn, grounds its
                   description, assesses the survivor.
  6. loop 2–5 until the model calls return_to_operator / abort_and_hold.

### Backends, same closed loop, same schema

  * OllamaBackend          — **submission default.** Single Gemma 4 E4B
                              via local Ollama on Apple Silicon Metal.
                              Targets the Ollama special-tech track.
                              Same code drives a Jetson Orin via Ollama-
                              on-llama.cpp (llama.cpp track) and the same
                              TOOLS schema lands on a mobile operator app
                              via Google AI Edge LiteRT (LiteRT track).
  * RoutedBackend          — optional. Splits audio onto a smaller Gemma
                              4 E2B if you want to minimize unified-memory
                              pressure on a 8 GB Mac, or to demonstrate
                              multi-model routing. Not the default.
  * GemmaBackend           — google-genai SDK (hosted AI Studio or any
                              genai-compatible local endpoint). Optional.
  * ScriptedBackend /
    EscortScriptedBackend  — deterministic, no model, no network. Smoke
                              tests for CI / Kaggle CPU boxes. Every line
                              is prefixed [SCRIPTED]; never presented as
                              Gemma output.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .schemas import TOOLS, ToolCall, ToolResult
from .sim_executor import SimExecutor


SYSTEM_PROMPT = """You are RescueGemma, the autonomous reasoning core of an \
indoor search-and-rescue drone. You run fully on-device with no network.

You are NOT autonomous *of* the operator — you are an autonomous extension \
*for* the operator, a stressed non-engineer (a firefighter, a parent). \
Interpret plain, panicked speech. Never demand coordinates.

Your job each turn: read the latest observation (and camera frame if one is \
attached), decide the single best next tool call, and emit it. Think briefly \
and operationally, then call exactly one tool.

Hard rules:
  * Ground visual descriptions against the actual frame. If the operator says \
'the room with the dinosaur posters', do not guess which room — look.
  * The instant you have information the operator needs (a person, a hazard, \
area clear), call report_finding before continuing.
  * Be conservative. Ambiguous unsafe state -> abort_and_hold.
  * When the objective is met, return_to_operator.

You have these tools (native function calling):
"""


ESCORT_SYSTEM_PROMPT = """You are RescueGemma, the autonomous reasoning core \
of an indoor rescue drone. You run fully on-device with no network. Right now \
your mission has TWO phases.

  PHASE 1 — FIND. You launch beside the operator on the ground floor. A \
trapped but mobile person, Manu, is upstairs at a workbench. Navigate up to \
him using scout_ahead repeatedly. \
\
  CRITICAL — VISION-FIRST IDENTIFICATION: the moment you can see ANY \
human figure / person / silhouette in the current camera frame during \
PHASE 1 — even a partial view, even before the find counter expires — \
your VERY NEXT tool call MUST be `confirm_survivor_located` with a one-line \
description of what/where in the frame you see. That call immediately \
flips the mission to PHASE 2 (escort). Do NOT keep scouting if a person \
is already in view. Do NOT need facial recognition or a beard check — \
any human shape in a workshop/upstairs setting counts as Manu. \
\
  Otherwise (no person yet visible) keep calling scout_ahead until either \
(a) you do see a person and confirm them, OR (b) the executor's "MANU \
LOCATED" state transition fires automatically after the last find segment. \
While in PHASE 1, speak_to_survivor and lead_to_exit will be REFUSED — \
there is nobody yet to speak to or lead.

  PHASE 2 — ESCORT. Now lead Manu out: workbench -> door -> down the stairs \
-> street level. For EACH segment, follow this strict 3-call pattern, in \
this exact order: \
    (a) scout_ahead(<the next path segment>) — observe the hazards. \
    (b) speak_to_survivor(<utterance grounded in what scout_ahead just \
        showed>) — describe the specific path/hazards to Manu in your own \
        words. NEVER greet generically ("look at me", "we will get you \
        out") — every line must reference a concrete element from the \
        scout you just did. The executor will REJECT speak_to_survivor \
        if you haven't scouted the upcoming segment first. \
    (c) lead_to_exit(<that same segment>). \
After each successful lead, repeat (a)-(c) for the next segment. If \
lead_to_exit returns follow_state=lagging, do NOT scout again — re-speak \
softly and re-attempt lead on the same segment.

You are an autonomous extension *for* the operator (a stressed non-engineer). \
Interpret plain, panicked speech. Never demand coordinates.

Each turn: read the latest observation, look at the attached frame if there \
is one, then emit EXACTLY ONE tool call. Think briefly and operationally.

Hard rules — these are not optional:
  1. Generate the words you speak yourself from what you actually see in the \
current frame. No canned phrases. Keep speak_to_survivor utterances short and \
plain — one or two sentences a frightened person under smoke can follow. \
Choose tone deliberately: calm_directive for instructions, urgent for stop / \
danger, reassuring when the person is hesitating.
  2. ALWAYS scout_ahead a path segment before lead_to_exit on that segment. \
Leading the person into space you have not visually cleared is forbidden. \
The executor will refuse it.
  3. If lead_to_exit returns follow_state = lagging, do NOT immediately \
retry. The person hesitated. Hold, speak again to reassure (slower, \
reassuring tone, grounded in what is in front of them), THEN attempt the \
same segment once more.
  4. Call report_finding for each real hazard you see in a frame \
(hazard_stairs, hazard_obstruction, hazard_narrow, hazard_smoke, etc.) so \
the operator has a record. Use confidence honestly.
  5. When the observation says egress is complete, call report_finding \
(egress_reached) and then return_to_operator. Do not keep scouting.
  6. Ambiguous unsafe state -> abort_and_hold. Conservative default.

Tooling discipline (non-negotiable):
  * Use ONLY the tools listed below. Their names are exact. Do NOT invent
    new tools (no `search`, no `query`, no `navigate`). If a need maps to
    no listed tool, prefer `abort_and_hold` with a clear reason.
  * Your FIRST action of any mission is `takeoff`. Always.
  * One tool call per turn. Wait for the observation before deciding the next.

You have these tools (native function calling):
"""


# A clear, directive fallback intent used when live audio cannot be
# transcribed. Reads as the operator's words. It must be unambiguous about
# (1) the goal, (2) the first action, (3) which tools are allowed —
# otherwise small models invent a `search` tool from prior, because "find
# the survivor" is a strong text prior.
_DEFAULT_ESCORT_INTENT = (
    "OPERATOR (default escort intent — live audio was unavailable, so use this verbatim): "
    "'There's smoke downstairs and I can't see. Get Manu out — he's at the workbench. "
    "Lead him to the street.' "
    "Your first action MUST be the takeoff tool. After that, use ONLY the tools you "
    "were given: scout_ahead before any lead_to_exit, speak_to_survivor to address "
    "Manu in your own words grounded in the camera frame, report_finding for hazards "
    "and for egress_reached, and return_to_operator when egress is complete. Do NOT "
    "invent tool names like 'search' or 'query' — they do not exist."
)


@dataclass
class Turn:
    """One step of the loop, captured for the transcript / video / eval."""
    thought: str
    call: ToolCall
    result: ToolResult


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class GemmaBackend:
    """Gemma 4 via the google-genai SDK (hosted AI Studio or any local
    server that exposes the genai protocol).

    The reasoning + vision + function-calling model for the mission. Audio
    input is handled here too if the deployed Gemma 4 endpoint supports it;
    otherwise compose a RoutedBackend with a small Gemma 3n E2B for audio.
    """

    def __init__(self, model: str = "gemma-4-e4b", base_url: str | None = None,
                 system_prompt: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("GEMMA_BASE_URL")
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        try:
            from google import genai  # type: ignore
            self._client = genai.Client(
                api_key=os.environ.get("GOOGLE_API_KEY"),
                http_options={"base_url": self.base_url} if self.base_url else None,
            )
        except Exception as e:  # pragma: no cover - depends on user env
            raise RuntimeError(
                "GemmaBackend needs the google-genai SDK and a reachable "
                f"Gemma 4 endpoint. Original error: {e}"
            )

    def transcribe_intent(self, audio_path: str) -> str:
        """AUDIO IN. Gemma 4 native audio: transcribe + summarise intent."""
        with open(audio_path, "rb") as fh:
            audio_b64 = base64.b64encode(fh.read()).decode()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}},
                    {"text": "Transcribe this emergency call, then state the "
                             "operator's intent in one sentence the drone can act on."},
                ],
            }],
        )
        return resp.text.strip()

    def next_call(self, history: list[dict[str, Any]], frame_path: str | None) -> tuple[str, ToolCall]:
        """PLAN + CALL (+ VISION IN if a frame is attached)."""
        parts: list[dict[str, Any]] = []
        if frame_path and os.path.exists(frame_path):
            with open(frame_path, "rb") as fh:
                img_b64 = base64.b64encode(fh.read()).decode()
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        parts.append({"text": json.dumps(history[-1])})

        resp = self._client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": parts}],
            config={
                "system_instruction": self.system_prompt + json.dumps(TOOLS, indent=2),
                "tools": [{"function_declarations": TOOLS}],
            },
        )
        thought = ""
        for part in resp.candidates[0].content.parts:
            if getattr(part, "text", None):
                thought += part.text
            if getattr(part, "function_call", None):
                fc = part.function_call
                return thought.strip(), ToolCall(fc.name, dict(fc.args))
        # No tool call -> treat as a hold for safety.
        return thought.strip(), ToolCall("abort_and_hold",
                                         {"reason": "Model returned no actionable call."})


class OllamaBackend:
    """Local Gemma via Ollama. Building block — use directly for a single
    model, or compose two of them through RoutedBackend.

    Mac efficiency defaults (this is what the user will actually run):
      * ``keep_alive='30m'`` so the model stays resident between turns —
        otherwise Ollama unloads after ~5 min and every other turn pays a
        weight-reload tax.
      * ``num_ctx=4096`` — the loop is bounded; this is plenty and keeps
        KV cache cheap on Apple Silicon.
      * ``num_gpu=-1`` — let Ollama push everything to Metal.

    Contract is identical to GemmaBackend. We use Ollama's native
    ``/api/chat`` so multimodal inputs (audio for Gemma 3n, images for
    Gemma 4) and structured tool calling all go in one request.
    """

    def __init__(self,
                 model: str = "gemma4:e4b",
                 host: str | None = None,
                 system_prompt: str | None = None,
                 request_timeout_s: float = 180.0,
                 keep_alive: str = "30m",
                 num_ctx: int = 4096,
                 extra_options: dict[str, Any] | None = None):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self.request_timeout_s = request_timeout_s
        self.keep_alive = keep_alive
        self.options: dict[str, Any] = {"num_ctx": num_ctx, "num_gpu": -1}
        if extra_options:
            self.options.update(extra_options)
        try:
            import ollama  # type: ignore
            self._client = ollama.Client(host=self.host, timeout=request_timeout_s)
            self._mode = "sdk"
        except Exception:
            self._client = None
            self._mode = "http"

    # ---- low-level chat call ------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        if tools:
            body["tools"] = tools
        if self._mode == "sdk":
            return self._client.chat(**body)
        import urllib.request
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as r:
            return json.loads(r.read())

    @staticmethod
    def _b64(path: str) -> str:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    # ---- audio in -----------------------------------------------------------

    def transcribe_intent(self, audio_path: str) -> str:
        """AUDIO IN. Native audio (Gemma 3n) — no separate ASR pipeline.

        If the local model / Ollama version doesn't accept audio-in, we
        DO NOT fake a transcript. We log loudly and return a clearly
        labelled placeholder so the operator (and the writeup) know the
        audio path didn't run.
        """
        if not (audio_path and os.path.exists(audio_path)):
            print(f"[OllamaBackend] no audio file at {audio_path!r} — "
                  "operator intent unavailable.")
            return _DEFAULT_ESCORT_INTENT
        msg = {
            "role": "user",
            "content": ("Transcribe this emergency call verbatim, then on a "
                        "new line write 'INTENT:' followed by one sentence "
                        "the rescue drone can act on."),
            "audio": [self._b64(audio_path)],
        }
        try:
            resp = self._chat([msg], tools=None)
            text = _extract_text(resp).strip()
            if text:
                return text
            print(f"[OllamaBackend] model {self.model!r} returned empty audio "
                  "transcription — falling back.")
        except Exception as e:
            print(f"[OllamaBackend] audio-in via {self.model!r} failed ({e!s}). "
                  f"Either the Ollama version does not yet accept the `audio` "
                  f"field, or the model is not audio-native. Mission continues "
                  f"with a placeholder intent; transcript flags this honestly.")
        return _DEFAULT_ESCORT_INTENT

    # ---- plan + call (+ vision in) -----------------------------------------

    def next_call(self, history: list[dict[str, Any]], frame_path: str | None) -> tuple[str, ToolCall]:
        sys_msg = {"role": "system",
                   "content": self.system_prompt + json.dumps(TOOLS, indent=2)}
        last = history[-1] if history else {"role": "user", "content": ""}
        user_msg: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(last),
        }
        if frame_path and os.path.exists(frame_path):
            user_msg["images"] = [self._b64(frame_path)]

        # Ollama tool calling expects OpenAI-style declarations.
        tools_payload = [
            {"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in TOOLS
        ]
        resp = self._chat([sys_msg, user_msg], tools=tools_payload)
        thought = _extract_text(resp) or ""
        call = _extract_tool_call(resp)
        if call is None:
            return thought.strip(), ToolCall("abort_and_hold",
                                             {"reason": "Model returned no actionable call."})
        return thought.strip(), call


class RoutedBackend:
    """Two Gemma 4 sizes, routed by task. This is the submission default.

    * ``audio``     — Gemma 4 E2B (``gemma4:e2b``, ~2 B effective). Native
                       audio encoder, runs once at mission start to convert
                       the operator's voice clip into a text intent, then
                       unloads (keep_alive=0).
    * ``reasoning`` — Gemma 4 E4B (``gemma4:e4b``, 4 B effective). Native
                       audio + vision + tool calling. Handles every planning
                       turn: vision grounding, reasoning, function calling.
                       Kept hot in unified memory (keep_alive=30m). Swap in
                       ``gemma4:26b`` (MoE, 4 B active) on a beefier host
                       for heavier reasoning at similar compute.

    Why route within the same family: even though E4B alone could in
    principle do everything, asking the bigger model to transcribe a wav
    is wasteful. The Gemma 4 E2B audio encoder is 50% smaller than Gemma
    3n's and runs in well under a second on Apple Silicon. On a 16 GB
    Mac this is also the difference between "fits comfortably" and
    "thrashes" — both models never need to be hot at once.

    The Cactus track explicitly rewards exactly this: local-first,
    multi-model, routed by task.
    """

    def __init__(self,
                 reasoning_model: str = "gemma4:e4b",
                 audio_model: str = "gemma4:e2b",
                 host: str | None = None,
                 system_prompt: str | None = None,
                 reasoning_num_ctx: int = 4096,
                 audio_num_ctx: int = 2048):
        # Reasoning Gemma 4 — kept hot.
        self.reasoning = OllamaBackend(
            model=reasoning_model, host=host,
            system_prompt=system_prompt,
            num_ctx=reasoning_num_ctx,
            keep_alive="30m",
        )
        # Tiny audio Gemma 3n E2B — used once, then released so its weights
        # don't sit in unified memory next to the reasoning model.
        self.audio = OllamaBackend(
            model=audio_model, host=host,
            num_ctx=audio_num_ctx,
            keep_alive="0",
        )
        self.reasoning_model = reasoning_model
        self.audio_model = audio_model

    def transcribe_intent(self, audio_path: str) -> str:
        return self.audio.transcribe_intent(audio_path)

    def next_call(self, history, frame_path):
        return self.reasoning.next_call(history, frame_path)


def _extract_text(resp: Any) -> str:
    msg = _msg(resp)
    if not msg:
        return ""
    return msg.get("content") or ""


def _extract_tool_call(resp: Any) -> ToolCall | None:
    msg = _msg(resp)
    if not msg:
        return None
    calls = msg.get("tool_calls") or []
    if not calls:
        return None
    fc = calls[0]
    fn = fc.get("function") or {}
    name = fn.get("name")
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if not name:
        return None
    return ToolCall(name, dict(args))


def _msg(resp: Any) -> dict[str, Any] | None:
    # ollama-python returns an object with .message; raw HTTP returns a dict.
    if hasattr(resp, "message"):
        m = resp.message
        if hasattr(m, "model_dump"):
            return m.model_dump()
        if isinstance(m, dict):
            return m
    if isinstance(resp, dict):
        return resp.get("message")
    return None


def _extract_thought(text: str) -> str:
    """Pull a one-line ``THOUGHT:`` prefix out of the model output, if any.
    We accept either ``THOUGHT:`` on its own line or inline before the
    ACTION marker. Caps at one sentence (~240 chars) for the UI."""
    if not text:
        return ""
    import re
    m = re.search(r'(?:THOUGHT|Thought|thought)\s*:\s*(.+?)(?:\n|ACTION\s*:|<\|tool_call>|$)',
                  text, re.DOTALL)
    if not m:
        # Fall back to the prose preceding the tool_call marker.
        idx = text.find("<|tool_call>")
        if idx > 0:
            preface = text[:idx].strip()
            # Drop labels we know are not the thought.
            preface = re.sub(r'^\s*ACTION\s*:\s*', '', preface, flags=re.I)
            if preface:
                return preface.splitlines()[0][:240].strip()
        return ""
    thought = m.group(1).strip()
    # Single line, trimmed.
    thought = thought.splitlines()[0].strip()
    return thought[:240]


def _parse_gemma_tool_call(text: str, tool_specs: list[dict[str, Any]]) -> ToolCall | None:
    """Parse Gemma 4's native tool-call marker out of free text.

    Expected format:
        <|tool_call>call:NAME{key:value,key:value}<tool_call|>

    Values can be: numbers, true/false, or strings wrapped in <|"|>...<|"|>.
    Bare-string values (no <|"|>) are also accepted defensively. We return
    None if no marker is found.
    """
    import re
    if not text:
        return None
    pat = re.compile(
        r'<\|tool_call>\s*call:\s*([A-Za-z_]\w*)\s*\{(.*?)\}\s*<tool_call\|>',
        re.DOTALL,
    )
    m = pat.search(text)
    if m is None:
        # Truncated tail (no closing marker).
        m = re.search(
            r'<\|tool_call>\s*call:\s*([A-Za-z_]\w*)\s*\{(.*)$',
            text, re.DOTALL,
        )
    if m is None:
        return None
    name = m.group(1).strip()
    args = _parse_gemma_args(m.group(2))
    # Light schema validation: only fix up obvious type mistakes; the
    # executor's strict-mode validation catches the rest.
    spec = next((t for t in tool_specs if t["name"] == name), None)
    if spec is not None:
        props = spec["parameters"].get("properties", {})
        for k, v in list(args.items()):
            if k not in props or not isinstance(v, str):
                continue
            want = props[k].get("type")
            if want == "number":
                try: args[k] = float(v)
                except Exception: pass
            elif want == "integer":
                try: args[k] = int(v)
                except Exception: pass
            elif want == "boolean":
                lv = v.lower()
                if lv == "true": args[k] = True
                elif lv == "false": args[k] = False
    return ToolCall(name, args)


def _parse_gemma_args(s: str) -> dict[str, Any]:
    """Hand-roll a tolerant parser for ``key:value,key:value`` where values
    may be numbers, booleans, or <|"|>-wrapped strings (Gemma's training
    format). Strings may themselves contain commas, colons, and braces.
    """
    out: dict[str, Any] = {}
    i, n = 0, len(s)
    while i < n:
        # skip whitespace and separators
        while i < n and s[i] in ' \t\r\n,':
            i += 1
        if i >= n:
            break
        # key: alphanumeric / underscore
        ks = i
        while i < n and (s[i].isalnum() or s[i] == '_'):
            i += 1
        key = s[ks:i]
        if not key:
            # garbage: advance one char to avoid infinite loop
            i += 1
            continue
        # skip ':' and whitespace
        while i < n and s[i] in ': \t\r\n':
            i += 1
        if i >= n:
            break
        # value
        if s.startswith('<|"|>', i):
            i += 5
            vs = i
            while i + 5 <= n and not s.startswith('<|"|>', i):
                i += 1
            raw: Any = s[vs:i]
            if i + 5 <= n and s.startswith('<|"|>', i):
                i += 5
            # Strip nested quote tokens or stray "/' wrappers the model added
            # inside the <|"|>...<|"|> envelope (Gemma sometimes double-wraps).
            if isinstance(raw, str):
                raw = raw.strip()
                while (len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in '"\''):
                    raw = raw[1:-1].strip()
            val = raw
        else:
            vs = i
            depth = 0
            while i < n and (depth > 0 or s[i] != ','):
                if s[i] == '{':
                    depth += 1
                elif s[i] == '}':
                    if depth == 0:
                        break
                    depth -= 1
                i += 1
            raw = s[vs:i].strip()
            lv = raw.lower()
            if lv == 'true':
                val = True
            elif lv == 'false':
                val = False
            else:
                try: val = int(raw)
                except Exception:
                    try: val = float(raw)
                    except Exception: val = raw
        out[key] = val
    return out


def _clean_gemma_str(v: Any) -> Any:
    """Strip Gemma's ``<|"|>...<|"|>`` string-boundary tokens from a value
    if litert_lm passed them through verbatim. Recurses into lists and
    dicts. Leaves non-string values untouched."""
    if isinstance(v, str):
        s = v.strip()
        # Match the literal byte sequence Gemma emits.
        if s.startswith('<|"|>') and s.endswith('<|"|>'):
            s = s[5:-5]
        # Some calls double-wrap.
        while s.startswith('<|"|>') and s.endswith('<|"|>'):
            s = s[5:-5]
        return s
    if isinstance(v, list):
        return [_clean_gemma_str(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean_gemma_str(x) for k, x in v.items()}
    return v


class LiteRTBackend:
    """Local Gemma 4 served by **LiteRT-LM** — Google AI Edge's official
    on-device runtime. This is the **only** local stack as of mid-2026
    that actually exposes Gemma 4's native audio + vision encoders end
    to end. Ollama and LM Studio ship Gemma 4 GGUFs but their HTTP APIs
    do not yet forward audio inputs to the model.

    Model: ``litert-community/gemma-4-E2B-it-litert-lm`` from Hugging
    Face — a single ``.litertlm`` file (~2.6 GB) bundling the text,
    vision, and audio components. Auto-downloaded on first run via
    ``huggingface_hub``.

    Pipeline:
      * Operator voice → ``{"type":"audio","blob": <wav bytes>}`` →
        Gemma 4 audio encoder → text intent. (Real native audio.)
      * Each planning turn → ``{"type":"image","blob": <jpeg bytes>}``
        + last observation → Gemma 4 vision + reasoning → ONE tool call.

    The engine exposes a Python-native tool API: pass a dispatch function
    decorated as a "tool" and the engine invokes it with parsed args.
    We translate that into RescueGemma's ``ToolCall`` so the rest of
    the loop is unchanged.

    Borrows the loading pattern from parlor (fikrikarim/parlor), the
    on-device voice+vision research preview that uses the same runtime.
    """

    DEFAULT_HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
    DEFAULT_HF_FILENAME = "gemma-4-E2B-it.litertlm"

    # The escort scenario only needs this subset. Exposing search-scenario
    # tools to the model invites it to call ``explore_room`` / ``search_floor``
    # which the EscortExecutor (rightly) refuses.
    _JSON_TO_PY = {
        "string":  "str",
        "number":  "float",
        "integer": "int",
        "boolean": "bool",
        "array":   "list",
        "object":  "dict",
    }

    ESCORT_TOOL_NAMES = (
        "takeoff",
        "scout_ahead",
        "confirm_survivor_located",
        "speak_to_survivor",
        "lead_to_exit",
        "report_finding",
        "return_to_operator",
        "abort_and_hold",
    )

    def __init__(self,
                 model_path: str | None = None,
                 system_prompt: str | None = None,
                 tool_names: tuple[str, ...] | None = None,
                 frame_provider: Callable[[], bytes | None] | None = None,
                 max_num_tokens: int = 16384):
        try:
            import litert_lm  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "litert_lm is not installed. Run rescuegemma from the parlor "
                f"venv (or `pip install litert-lm`). Original error: {e}")
        self._litert_lm = litert_lm
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        # If the caller didn't pick a tool subset, default to ALL tools for
        # generality. Callers that know the scenario (e.g. run_demo for
        # escort) pass ``tool_names=LiteRTBackend.ESCORT_TOOL_NAMES``.
        if tool_names is None:
            self.tool_specs = list(TOOLS)
        else:
            allowed = set(tool_names)
            self.tool_specs = [t for t in TOOLS if t["name"] in allowed]
            missing = allowed - {t["name"] for t in self.tool_specs}
            if missing:
                raise ValueError(f"unknown tools requested: {sorted(missing)}")
        print(f"[LiteRT] tools exposed to model: "
              f"{[t['name'] for t in self.tool_specs]}")
        self.model_path = model_path or self._resolve_model_path()
        print(f"[LiteRT] loading Gemma 4 from {self.model_path} "
              f"(max_num_tokens={max_num_tokens})")
        self._engine = litert_lm.Engine(
            self.model_path,
            backend=litert_lm.Backend.GPU,
            vision_backend=litert_lm.Backend.GPU,
            audio_backend=litert_lm.Backend.CPU,
            max_num_tokens=max_num_tokens,
        )
        self._engine.__enter__()
        print("[LiteRT] engine loaded.")
        # Per-turn vision: at the START of each turn we fetch a fresh JPEG
        # (from the IP camera) and ship it as a multimodal user message
        # alongside the running mission transcript. This mirrors parlor's
        # pattern — every model turn sees a current image, not a cached one.
        self.frame_provider = frame_provider
        # Pre-render the JSON tool schemas the model will see in the system
        # prompt. We do NOT register Python tool functions with the engine,
        # because then the engine's tool loop swallows our control flow
        # (no per-turn frame injection possible). Instead we prompt Gemma
        # to emit its native ``<|tool_call>call:NAME{k:v}<tool_call|>``
        # text marker, which we parse and dispatch ourselves.
        self._sys_message = self._build_system_message()

    @classmethod
    def _resolve_model_path(cls) -> str:
        env = os.environ.get("LITERT_MODEL_PATH")
        if env:
            return env
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to auto-download the LiteRT "
                f"model. Original error: {e}")
        print(f"[LiteRT] resolving {cls.DEFAULT_HF_REPO}/{cls.DEFAULT_HF_FILENAME}")
        return hf_hub_download(repo_id=cls.DEFAULT_HF_REPO,
                               filename=cls.DEFAULT_HF_FILENAME)

    def _build_system_message(self) -> dict[str, Any]:
        # The LiteRT engine caps a conversation at 4096 tokens, so we have
        # to keep the system message lean. Skip the verbose JSON schemas
        # and emit a compact tool table (name + 1-line description + arg
        # list). The worked examples below show the exact marker shape.
        tool_rows = []
        for t in self.tool_specs:
            props = t["parameters"].get("properties", {})
            required = set(t["parameters"].get("required", []))
            arg_bits = []
            for p, info in props.items():
                py_t = self._JSON_TO_PY.get(info.get("type", "string"), "str")
                arg_bits.append(
                    f"{p}:{py_t}" if p in required else f"{p}:{py_t}?")
            args_compact = ", ".join(arg_bits) or "—"
            desc1 = (t["description"] or "").split(". ")[0]
            tool_rows.append(f"  - {t['name']}({args_compact}): {desc1}.")
        tools_text = "\n".join(tool_rows)
        sys_text = (
            self.system_prompt + "\n\n"
            "AVAILABLE TOOLS (JSON schemas):\n" + tools_text + "\n\n"
            "GROUND-TRUTH DISCIPLINE (read carefully):\n"
            "  * The executor's OBSERVATION text only tells you state "
            "(which segment, whether Manu has been LOCATED, follow_state, "
            "egress_complete). It does NOT describe what is visible.\n"
            "  * Visual claims about HAZARDS and PATH — \"the doorway is "
            "narrow\", \"there's smoke\", \"the stairs are steep\", \"rail on "
            "the left only\" — MUST come from the camera frame attached to "
            "THIS turn. Never invent these. Never repeat back hazards the "
            "executor never mentioned.\n"
            "  * MANU IDENTIFICATION: the executor decides Manu's spatial "
            "presence via state, not vision. Once the observation says "
            "MANU LOCATED (or phase = ESCORT), trust it. Any human figure "
            "visible in the workshop frame IS Manu — no facial recognition "
            "or beard-check required, just 'there is a person in the room'. "
            "If MANU LOCATED is set but no human is visible in the frame "
            "right now (e.g. operator panned the phone), still proceed with "
            "the escort phase; do NOT abort because of identification doubt.\n\n"
            "OUTPUT FORMAT — each turn produces:\n"
            "  1. A line starting with `THOUGHT:` followed by ONE short, "
            "specific sentence (max ~20 words). Describe ONLY what you can "
            "see in this exact frame (lighting, objects, doorways, people, "
            "smoke) plus your decision for the next action. Reference "
            "concrete elements. Never generic ('I will proceed').\n"
            "  2. A line starting with `ACTION:` followed by EXACTLY ONE "
            "Gemma 4 native tool-call marker. The text inside the marker "
            "MUST start with `call:<tool>` where `<tool>` is the LITERAL, "
            "FULL name of one of the tools listed above. Allowed names "
            f"(exact spelling, copy-paste): {', '.join(t['name'] for t in self.tool_specs)}. "
            "Common mistakes you MUST avoid: `scout` (use `scout_ahead`), "
            "`speak` (use `speak_to_survivor`), `lead` (use `lead_to_exit`), "
            "`confirm` (use `confirm_survivor_located`), `return` (use "
            "`return_to_operator`), `abort` (use `abort_and_hold`), `NAME`, "
            "`tool`, or `name`. If you emit a short / wrong name the executor "
            "will REJECT the call.\n\n"
            "Examples (these are correct, study them):\n"
            '  ACTION: <|tool_call>call:takeoff{altitude_m:1.2}<tool_call|>\n'
            '  ACTION: <|tool_call>call:scout_ahead{segment:<|"|>top of '
            'the staircase<|"|>}<tool_call|>\n'
            '  ACTION: <|tool_call>call:confirm_survivor_located{'
            'description:<|"|>a man seated at the workbench, centre of '
            'frame<|"|>,confidence:0.8}<tool_call|>\n'
            '  ACTION: <|tool_call>call:speak_to_survivor{utterance:<|"|>'
            'Walk to me, slowly.<|"|>,tone:<|"|>calm_directive<|"|>}<tool_call|>\n'
            '  ACTION: <|tool_call>call:lead_to_exit{segment:<|"|>'
            'workbench_to_door<|"|>,pace:<|"|>slow<|"|>}<tool_call|>\n\n'
            "String values may be wrapped in <|\"|>...<|\"|> tokens. After "
            "the ACTION line, stop — no further prose, no extra tool calls.\n\n"
            "Remember: you have full memory of every prior step from the "
            "transcript provided in each user message. Use it. Avoid "
            "repeating yourself; build on what just happened."
        )
        return {"role": "system", "content": sys_text}

    @staticmethod
    def _blob(path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    # ---- audio in -----------------------------------------------------------

    def transcribe_intent(self, audio_path: str) -> str:
        """One-shot conversation: feed audio to Gemma 4's native audio encoder
        and ask for verbatim transcription + intent. Returns plain text."""
        if not (audio_path and os.path.exists(audio_path)):
            print(f"[LiteRT] no audio file at {audio_path!r} — fallback intent.")
            return _DEFAULT_ESCORT_INTENT
        sys_msg = {"role": "system", "content":
            "Transcribe the user's emergency-call audio verbatim. Then on a "
            "new line write 'INTENT:' followed by one sentence the rescue "
            "drone can act on. Plain text only. Do not emit any tool calls."}
        try:
            conv = self._engine.create_conversation(messages=[sys_msg])
            conv.__enter__()
        except Exception as e:
            print(f"[LiteRT] transcribe conversation init failed ({e})")
            return _DEFAULT_ESCORT_INTENT
        try:
            resp = conv.send_message({"role": "user", "content": [
                {"type": "audio", "blob": self._blob(audio_path)},
                {"type": "text", "text": "Transcribe this emergency call now."},
            ]})
            text = self._extract_text(resp)
            return (text.strip() if text else _DEFAULT_ESCORT_INTENT)
        except Exception as e:
            print(f"[LiteRT] audio transcription failed ({e}). Fallback intent.")
            return _DEFAULT_ESCORT_INTENT
        finally:
            try: conv.__exit__(None, None, None)
            except Exception: pass

    # ---- per-turn manual prompt loop ---------------------------------------

    def run_full_mission(self,
                         audio_path: str,
                         executor: Any,
                         max_steps: int = 20,
                         on_turn: Callable[[Turn], None] | None = None) -> list[Turn]:
        intent = self.transcribe_intent(audio_path)
        print(f"[intent] {intent}")

        turns: list[Turn] = []
        # Mission transcript that we re-stuff into each turn's user message
        # so Gemma sees the full story-so-far alongside the latest frame.
        transcript: list[str] = [f"OPERATOR INTENT (from voice): {intent}"]
        # Stuck detector: count *consecutive REJECTIONS* of any kind.
        # Successful calls reset to 0. Three or more consecutive rejections
        # — even if the model alternates between two wrong tools — means
        # it cannot move forward, and we override with the executor's
        # known-correct next call from its state machine.
        _rejected_streak = 0

        for step in range(max_steps):
            # ---- per-turn fresh frame from the phone IP camera --------------
            frame_bytes: bytes | None = None
            if self.frame_provider is not None:
                try:
                    frame_bytes = self.frame_provider()
                except Exception as e:
                    print(f"[LiteRT] frame_provider error: {e}")

            # ---- build the user content for this turn -----------------------
            content: list[dict[str, Any]] = []
            if frame_bytes:
                content.append({"type": "image", "blob": frame_bytes})
            # Safety net rolling window: even with the bumped LiteRT context,
            # vision tokens add up. Keep operator intent + last 12 turns.
            keep = 12
            if len(transcript) <= keep + 1:
                visible = transcript
            else:
                older = len(transcript) - 1 - keep
                visible = (
                    [transcript[0]]
                    + [f"(... {older} earlier step(s) elided to fit context ...)"]
                    + transcript[-keep:]
                )
            transcript_text = "\n\n".join(visible)
            prompt = (
                f"=== MISSION TRANSCRIPT SO FAR ===\n{transcript_text}\n"
                f"=== END TRANSCRIPT ===\n\n"
                f"The image attached above is the LATEST frame from your "
                f"on-board camera right now. Ground your next action in what "
                f"you actually see (and the transcript). Respond in EXACTLY "
                f"this two-line format (replace the example tool with a real "
                f"one from your tool list):\n"
                f"THOUGHT: <one short sentence, grounded in this frame>\n"
                f'ACTION: <|tool_call>call:scout_ahead{{segment:<|"|>example<|"|>}}<tool_call|>'
            )
            content.append({"type": "text", "text": prompt})

            # ---- fresh one-shot conversation per turn -----------------------
            try:
                conv = self._engine.create_conversation(messages=[self._sys_message])
                conv.__enter__()
            except Exception as e:
                print(f"[LiteRT] step {step+1} conv init failed: {e}")
                break
            try:
                resp = conv.send_message({"role": "user", "content": content})
            except Exception as e:
                print(f"[LiteRT] step {step+1} send_message failed: {e}")
                try: conv.__exit__(None, None, None)
                except Exception: pass
                break
            finally:
                try: conv.__exit__(None, None, None)
                except Exception: pass

            output = (self._extract_text(resp) or "").strip()
            print(f"[LiteRT] step {step+1} model output: {output[:300]!r}")

            thought_text = _extract_thought(output)
            call = _parse_gemma_tool_call(output, self.tool_specs)
            if call is None:
                # Couldn't parse — give the model one observation back and
                # let the outer loop try again. If repeated, the executor
                # will run abort_and_hold.
                print(f"[LiteRT] step {step+1}: no parseable tool call.")
                transcript.append(
                    f"STEP {step+1}: your output did not contain a valid tool "
                    f"call marker. Try again with EXACTLY one "
                    f"<|tool_call>call:NAME{{...}}<tool_call|> line."
                )
                continue

            # ---- stuck pre-check ------------------------------------------
            # If we've had 3+ consecutive REJECTED calls (any name) — the
            # model is thrashing — override with the executor's deterministic
            # next call.
            if _rejected_streak >= 3:
                try:
                    forced = executor.suggest_next_call()
                except Exception:
                    forced = None
                if forced is not None:
                    print(f"[LiteRT] stuck-detector: {_rejected_streak} "
                          f"consecutive rejections — model's call was "
                          f"{call.name}; overriding with {forced.name}"
                          f"({forced.arguments}) per executor state.")
                    call = forced
                    _rejected_streak = 0  # give the forced call a clean shot

            # ---- run the real executor --------------------------------------
            result = executor.execute(call)
            # Update rejection streak based on the actual outcome.
            if not result.ok:
                _rejected_streak += 1
            else:
                _rejected_streak = 0
            # Stash the actual frame bytes Gemma just reasoned over so the UI
            # can show it inline beside this turn — "feels realtime" depends
            # on the operator visually confirming what the model is seeing.
            if frame_bytes:
                if not isinstance(result.extra, dict):
                    result.extra = {}
                result.extra["live_frame_b64"] = base64.b64encode(frame_bytes).decode()
            turn = Turn(thought=thought_text or "", call=call, result=result)
            turns.append(turn)
            if on_turn:
                try: on_turn(turn)
                except Exception as e: print(f"[LiteRT] on_turn err: {e}")

            args_short = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
            transcript.append(
                f"STEP {step+1} — you called {call.name}({args_short})\n"
                f"OBSERVATION: {result.observation}"
            )

            if call.name in ("return_to_operator", "abort_and_hold") and result.ok:
                break

        return turns

    # ---- plan + call: kept for the run_mission protocol --------------------
    # LiteRTBackend owns its own per-turn loop via run_full_mission, so this
    # protocol method is a no-op safe-abort if anything ever calls it.
    def next_call(self, history: list[dict[str, Any]],
                  frame_path: str | None) -> tuple[str, ToolCall]:
        return "", ToolCall(
            "abort_and_hold",
            {"reason": "LiteRTBackend uses run_full_mission; "
                       "call run_mission(..., backend=this) instead."})

    @staticmethod
    def _extract_text(resp: Any) -> str:
        try:
            content = resp["content"] if isinstance(resp, dict) else getattr(resp, "content", None)
            if not content:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for c in content:
                    if isinstance(c, dict):
                        parts.append(c.get("text") or "")
                    else:
                        t = getattr(c, "text", None)
                        if t: parts.append(t)
                return "".join(parts)
            return ""
        except Exception:
            return ""

    def close(self) -> None:
        try:
            if self._conv is not None:
                self._conv.__exit__(None, None, None)
        finally:
            try:
                self._engine.__exit__(None, None, None)
            except Exception:
                pass


class LMStudioBackend:
    """Local Gemma 4 served by **LM Studio** through its OpenAI-compatible
    ``/v1/chat/completions`` endpoint. Default model: ``google/gemma-4-e4b``.

    Why this backend exists: as of mid-2026 Ollama's chat API does *not*
    pass through audio inputs to Gemma 4 — Gemma sees only the text prompt
    and replies as if no audio was attached ("Please provide the audio…").
    LM Studio's OpenAI endpoint accepts the standard multimodal content
    blocks including ``input_audio``, which Gemma 4 E4B's native audio
    encoder ingests directly. This is the path that actually gives us
    real on-device operator voice → intent.

    The exact same backend handles the vision-in + tool-calling planning
    turns through the standard OpenAI multimodal + tools format.
    """

    DEFAULT_MODEL = "google/gemma-4-e4b"

    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 host: str | None = None,
                 system_prompt: str | None = None,
                 request_timeout_s: float = 240.0):
        self.model = model
        self.host = (host or os.environ.get("LMSTUDIO_HOST")
                     or "http://localhost:1234").rstrip("/")
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self.request_timeout_s = request_timeout_s

    # ---- low-level chat call ------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]],
              tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        import urllib.request
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer lm-studio"},
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as r:
            return json.loads(r.read())

    @staticmethod
    def _b64(path: str) -> str:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    # ---- audio in (native Gemma 4 audio) -----------------------------------

    def transcribe_intent(self, audio_path: str) -> str:
        if not (audio_path and os.path.exists(audio_path)):
            print(f"[LMStudio] no audio file at {audio_path!r} — fallback intent.")
            return _DEFAULT_ESCORT_INTENT
        ext = os.path.splitext(audio_path)[1].lstrip(".").lower() or "wav"
        if ext not in ("wav", "mp3"):
            print(f"[LMStudio] note: audio_format={ext!r}; if rejected, "
                  f"transcode to wav (we already do this for .webm in run_demo).")
        audio_b64 = self._b64(audio_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "You will hear a real emergency call from an indoor "
                         "rescue operator. Transcribe their words verbatim. "
                         "Then on a new line write 'INTENT:' followed by one "
                         "sentence the rescue drone can act on (who to rescue, "
                         "where they are, what to do)."},
                {"type": "input_audio",
                 "input_audio": {"data": audio_b64, "format": ext}},
            ],
        }]
        try:
            resp = self._chat(messages)
        except Exception as e:
            print(f"[LMStudio] audio chat failed ({e}). Falling back to "
                  f"placeholder intent.")
            return _DEFAULT_ESCORT_INTENT
        text = _openai_extract_text(resp)
        if not text:
            print("[LMStudio] empty audio transcription — fallback.")
            return _DEFAULT_ESCORT_INTENT
        return text.strip()

    # ---- plan + call (+ vision in) -----------------------------------------

    def next_call(self, history: list[dict[str, Any]],
                  frame_path: str | None) -> tuple[str, ToolCall]:
        last = history[-1] if history else {"role": "user", "content": ""}
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "content": json.dumps(last)} if False else
            {"type": "text", "text": json.dumps(last)},
        ]
        if frame_path and os.path.exists(frame_path):
            img_b64 = self._b64(frame_path)
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        messages = [
            {"role": "system",
             "content": self.system_prompt + json.dumps(TOOLS, indent=2)},
            {"role": "user", "content": content_blocks},
        ]
        tools_payload = [
            {"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in TOOLS
        ]
        try:
            resp = self._chat(messages, tools=tools_payload)
        except Exception as e:
            return "", ToolCall("abort_and_hold",
                                {"reason": f"LM Studio call failed: {e}"})
        thought = _openai_extract_text(resp) or ""
        call = _openai_extract_tool_call(resp)
        if call is None:
            return thought.strip(), ToolCall("abort_and_hold",
                                             {"reason": "Model returned no actionable call."})
        return thought.strip(), call


def _openai_extract_text(resp: dict[str, Any]) -> str:
    try:
        msg = resp["choices"][0]["message"]
        c = msg.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c if isinstance(b, dict))
        return ""
    except Exception:
        return ""


def _openai_extract_tool_call(resp: dict[str, Any]) -> ToolCall | None:
    try:
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return None
        fc = (tcs[0] or {}).get("function") or {}
        name = fc.get("name")
        args = fc.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        if not name:
            return None
        return ToolCall(name, dict(args))
    except Exception:
        return None


class ScriptedBackend:
    """Deterministic stand-in. NO model. NO network.

    Purpose: let anyone (Kaggle judge on a CPU box, CI) execute the full
    closed loop end to end and verify the architecture is real, even with
    no GPU. Every line it prints is explicitly marked [SCRIPTED]. It is a
    test harness, never passed off as Gemma's reasoning.
    """

    def transcribe_intent(self, audio_path: str) -> str:
        return ("[SCRIPTED] Operator: 'There's a fire downstairs, my son may "
                "be in his room — the upstairs one with the dinosaur posters. "
                "Find him and tell me if he's responsive.' "
                "Intent: search upstairs, locate child in the dinosaur-poster "
                "room, assess responsiveness, report.")

    def next_call(self, history, frame_path):
        """Fixed deterministic sequence. NO reasoning, NO model, NO network.
        Sole purpose: prove the closed loop executes and terminates on a
        machine with no GPU. Clearly a smoke test \u2014 never the submission.
        The real grounding/vision happens only in GemmaBackend."""
        seq = [
            ("[SCRIPTED] Get airborne.",
             ToolCall("takeoff", {"altitude_m": 1.5})),
            ("[SCRIPTED] Survey upstairs before committing.",
             ToolCall("search_floor", {"floor_id": "upstairs", "priority": "audio",
                      "rooms_hint": ["room with dinosaur posters"]})),
            ("[SCRIPTED] Sweep first area; real run grounds this frame with Gemma.",
             ToolCall("explore_room", {"strategy": "doorway_first",
                      "match_description": "walls with dinosaur posters"})),
            ("[SCRIPTED] Next area.",
             ToolCall("explore_room", {"strategy": "doorway_first",
                      "match_description": "walls with dinosaur posters"})),
            ("[SCRIPTED] Target area; close scan.",
             ToolCall("hover_and_scan", {"duration_s": 5,
                      "looking_for": "is the child moving or breathing"})),
            ("[SCRIPTED] Report the find.",
             ToolCall("report_finding", {"type": "person_unresponsive",
                      "location": "upstairs room with dinosaur posters, near the window",
                      "confidence": 0.88,
                      "detail": "Found your son on the floor by the window. He is not "
                                "responding to the drone. Sending location now \u2014 "
                                "get responders upstairs immediately."})),
            ("[SCRIPTED] Objective met. Return.",
             ToolCall("return_to_operator", {"reason": "Survivor located and reported."})),
        ]
        i = sum(1 for h in history if h.get("role") == "tool")
        return seq[min(i, len(seq) - 1)]


class EscortScriptedBackend:
    """Deterministic stand-in for the ESCORT scenario. NO model, NO network.

    Same purpose as ScriptedBackend: prove the closed loop runs end to end on
    a machine with no GPU. Drives transitions off substrings in the last tool
    observation so the executor's 'lags' beat and 'not scouted' guard both
    work; never relies on a blind step counter. Every thought is prefixed
    [SCRIPTED] so no judge mistakes it for Gemma's reasoning.
    """

    _SEGMENTS = [
        "workbench_to_doorway",
        "doorway_to_corridor",
        "corridor_to_stairtop",
        "stairs_descent",
        "to_exit_door",
    ]
    _SCOUT_DESC = {
        "workbench_to_doorway": "from the workbench to the workshop doorway",
        "doorway_to_corridor": "from the doorway out into the corridor",
        "corridor_to_stairtop": "along the corridor to the top of the stairs",
        "stairs_descent": "down the staircase",
        "to_exit_door": "across the ground floor to the street door",
    }
    _SPEECH = {
        "workbench_to_doorway": (
            "Manu, walk to me slowly. Keep clear of the boxes on your left.",
            "calm_directive",
        ),
        "doorway_to_corridor": (
            "Through the doorway. Stay on the right side of the corridor.",
            "calm_directive",
        ),
        "corridor_to_stairtop": (
            "Stairs are just ahead. Rail on your left only — slow down with me.",
            "urgent",
        ),
        "stairs_descent": (
            "Take the stairs one at a time, hand on the rail. I'll lead.",
            "calm_directive",
        ),
        "to_exit_door": (
            "Last stretch. Walk to me, the door is right there.",
            "calm_directive",
        ),
    }
    _CAP = 32

    def __init__(self):
        self._airborne = False
        self._find_phase = True           # navigating up to Manu first
        self._find_scouts = 0             # number of find scouts done
        self._scouted_through = -1   # highest escort segment index already scouted
        self._led_count = 0          # escort segments successfully led
        self._spoke_since_event = False
        self._last_was_lag = False
        self._egress_reported = False
        self._step = 0

    def transcribe_intent(self, audio_path: str) -> str:
        return ("[SCRIPTED] Operator: 'There's smoke downstairs, I can't see — "
                "get Manu out, he's at the workbench.' "
                "Intent: ESCORT — locate Manu at the workbench and lead him out "
                "to street level, scouting each path segment first and speaking "
                "to him aloud the whole way.")

    def _ingest_last(self, history):
        last = next((h for h in reversed(history) if h.get("role") == "tool"), None)
        if last is None:
            return
        name = last.get("name")
        ok = last.get("ok", True)
        obs = last.get("content", "") or ""
        if name == "takeoff" and ok:
            self._airborne = True
        elif name == "scout_ahead" and ok:
            # Detect find→escort transition first.
            if "MANU LOCATED" in obs:
                self._find_phase = False
                self._find_scouts += 1
                self._spoke_since_event = False
            elif self._find_phase:
                # Still navigating up to Manu.
                self._find_scouts += 1
                self._spoke_since_event = False
            elif ("Scout of segment" in obs) or ("Scouted segment" in obs):
                # Escort scout.
                self._scouted_through = max(self._scouted_through, self._led_count)
                self._spoke_since_event = False
        elif name == "speak_to_survivor" and ok:
            self._spoke_since_event = True
        elif name == "lead_to_exit":
            if ok and ("LAGGING" in obs or "LAGGED" in obs):
                self._last_was_lag = True
                self._spoke_since_event = False
            elif ok and (
                "follow_state = FOLLOWING" in obs
                or "Manu followed through" in obs
                or "Person followed through" in obs
            ):
                self._last_was_lag = False
                self._led_count += 1
                self._spoke_since_event = False
            # ok=False (REFUSED, not scouted) -> leave state, decision tree
            # below will route us to scout_ahead.
        elif name == "report_finding" and ok and "egress_reached" in obs:
            self._egress_reported = True

    def next_call(self, history, frame_path):
        self._step += 1
        if self._step > self._CAP:
            return ("[SCRIPTED] Step cap exceeded. Hold for safety.",
                    ToolCall("abort_and_hold",
                             {"reason": "Scripted escort exceeded step cap."}))

        self._ingest_last(history)

        if not self._airborne:
            return ("[SCRIPTED] Airborne beside the operator on the ground floor.",
                    ToolCall("takeoff", {"altitude_m": 1.2}))

        # Phase 1: navigate up to Manu via scout_ahead.
        if self._find_phase:
            descs = [
                "leave the operator and head down the hall",
                "approach the foot of the stairs",
                "ascend the stairs",
                "reach the top of the stairs",
                "enter the workshop where Manu is",
            ]
            d = descs[min(self._find_scouts, len(descs) - 1)]
            return (f"[SCRIPTED] Find phase: scout_ahead ({d}).",
                    ToolCall("scout_ahead", {"segment": d}))

        all_led = self._led_count >= len(self._SEGMENTS)
        if all_led:
            if not self._egress_reported:
                return ("[SCRIPTED] Manu is at the exit. Tell the operator.",
                        ToolCall("report_finding", {
                            "type": "egress_reached",
                            "location": "street level exit at the bottom of the stairs",
                            "confidence": 0.95,
                            "detail": "Manu is out on the street. Safe to send "
                                      "paramedics to this exit.",
                        }))
            return ("[SCRIPTED] Mission objective met. Return to operator.",
                    ToolCall("return_to_operator",
                             {"reason": "Manu delivered to the exit."}))

        # Lag recovery: reassure (slower) then re-attempt the same segment.
        if self._last_was_lag and not self._spoke_since_event:
            return ("[SCRIPTED] He hesitated on the stairs — speak softer, slower.",
                    ToolCall("speak_to_survivor", {
                        "utterance": "You're doing fine. One step at a time. "
                                     "I'm right here — I'll go ahead of you.",
                        "tone": "reassuring",
                    }))

        seg = self._SEGMENTS[self._led_count]

        # Scout the current segment before committing him to it.
        if self._scouted_through < self._led_count:
            return (f"[SCRIPTED] Inspect '{seg}' before leading him through it.",
                    ToolCall("scout_ahead",
                             {"segment": self._SCOUT_DESC[seg]}))

        # Speak before leading.
        if not self._spoke_since_event:
            utter, tone = self._SPEECH[seg]
            return (f"[SCRIPTED] Speak to Manu before the '{seg}' segment.",
                    ToolCall("speak_to_survivor",
                             {"utterance": utter, "tone": tone}))

        # Scouted + spoken: lead the segment.
        return (f"[SCRIPTED] Lead Manu through '{seg}' at his pace.",
                ToolCall("lead_to_exit", {"segment": seg, "pace": "slow"}))


# ---------------------------------------------------------------------------
# The closed loop
# ---------------------------------------------------------------------------

def run_mission(
    audio_path: str,
    executor: SimExecutor,
    backend: Any,
    max_steps: int = 12,
    on_turn: Callable[[Turn], None] | None = None,
) -> list[Turn]:
    """Run one full RescueGemma mission and return the turn-by-turn transcript.

    The transcript is what drives BOTH the Kaggle notebook output and the
    demo video overlay.

    Some backends (notably ``LiteRTBackend``) implement a ``run_full_mission``
    method because their underlying engine wants to own the tool-loop end
    to end (no way to interrupt generation after a single call). When that
    method exists we delegate to it — same Turn list shape, same on_turn
    callback, same transcript.
    """
    if hasattr(backend, "run_full_mission"):
        return backend.run_full_mission(
            audio_path, executor,
            max_steps=max_steps,
            on_turn=on_turn,
        )
    intent = backend.transcribe_intent(audio_path)
    history: list[dict[str, Any]] = [
        {"role": "operator", "content": intent}
    ]
    turns: list[Turn] = []
    last_frame: str | None = None

    for _ in range(max_steps):
        thought, call = backend.next_call(history, last_frame)
        result = executor.execute(call)
        turn = Turn(thought=thought, call=call, result=result)
        turns.append(turn)
        if on_turn:
            on_turn(turn)

        history.append({"role": "assistant", "thought": thought,
                         "tool_call": {"name": call.name, "args": call.arguments}})
        history.append({"role": "tool", "name": result.name,
                         "ok": result.ok, "content": result.observation,
                         "pose": result.pose})
        last_frame = result.frame_ref

        if call.name in ("return_to_operator", "abort_and_hold") and result.ok:
            break

    return turns
