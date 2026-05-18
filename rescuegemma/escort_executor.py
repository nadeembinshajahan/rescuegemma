"""
RescueGemma — Escort executor (two-phase: find → escort).

Body for the FIND + ESCORT scenario.

  Phase 1 — FIND (drone alone). Drone takes off next to the operator
            downstairs. Manu is somewhere upstairs at a workbench. The
            agent issues ``scout_ahead`` repeatedly to navigate up the
            stairs and into the workshop. ``speak_to_survivor`` and
            ``lead_to_exit`` are REFUSED in this phase — there is nobody
            to speak to or lead yet.

  Phase 2 — ESCORT (Manu in tow). Once the last find segment is scouted
            the executor flips to escort mode and the observation says
            "Manu LOCATED". The existing escort behaviour runs:
            scout_ahead → speak_to_survivor → lead_to_exit per segment,
            with a hard-coded 'lags' beat on the stairs.

Same ToolResult contract as the real flight bridge. Only flight dynamics
and the person's follow signal are simulated; frames are real footage.
"""

from __future__ import annotations

import json
import os

from .schemas import ToolCall, ToolResult, validate_call


class EscortExecutor:
    def __init__(self, egress_path: str, frames_dir: str):
        with open(egress_path) as fh:
            self.eg = json.load(fh)
        self.frames_dir = frames_dir
        self.find_segments = self.eg.get("find_segments", [])
        self.escort_segments = self.eg.get("segments", [])
        self._find_i = 0
        self._escort_i = 0
        self._scouted_escort: set[str] = set()
        self._spoke_for_seg: str | None = None  # last segment we spoke for
        self._airborne = False
        self._cur: dict[str, int] = {}
        self._phase = "find" if self.find_segments else "escort"
        self._manu_located = (self._phase == "escort")

    # ---- internals ----------------------------------------------------

    def _frame(self, seg, kind):
        fs = seg.get(f"{kind}_frames", [])
        if not fs:
            return None
        k = f"{seg['id']}:{kind}"
        idx = self._cur.get(k, 0)
        f = fs[min(idx, len(fs) - 1)]
        self._cur[k] = idx + 1
        p = os.path.join(self.frames_dir, f)
        return p if os.path.exists(p) else f

    def _escort_seg(self):
        if self._escort_i < len(self.escort_segments):
            return self.escort_segments[self._escort_i]
        return None

    def _find_seg(self):
        if self._find_i < len(self.find_segments):
            return self.find_segments[self._find_i]
        return None

    # ---- dispatch -----------------------------------------------------

    def execute(self, call: ToolCall) -> ToolResult:
        ok, msg = validate_call(call)
        if not ok:
            return ToolResult(call.name, False, f"REJECTED: {msg}")
        h = getattr(self, f"_do_{call.name}", None)
        if h is None:
            return ToolResult(call.name, False, f"No escort impl for '{call.name}'")
        return h(call.arguments)

    # ---- per-tool implementations -------------------------------------

    # ---- introspection: what call SHOULD happen next? -----------------
    # Used by the backend's stuck-detector to force progression when the
    # agent keeps emitting the wrong tool.
    def suggest_next_call(self) -> ToolCall:
        if not self._airborne:
            return ToolCall("takeoff", {"altitude_m": 1.2})
        if self._phase == "find":
            return ToolCall("scout_ahead",
                            {"segment": "advance toward the survivor"})
        # escort phase
        cur = self._escort_seg()
        if cur is None:
            # No segments left — wrap up.
            return ToolCall("report_finding", {
                "type": "egress_reached",
                "location": "exit",
                "confidence": 0.95,
                "detail": "Survivor delivered to the exit.",
            })
        seg_id = cur["id"]
        if seg_id not in self._scouted_escort:
            return ToolCall("scout_ahead", {"segment": seg_id})
        if self._spoke_for_seg != seg_id:
            return ToolCall("speak_to_survivor", {
                "utterance": f"Stay with me. Move through the {seg_id.replace('_',' ')}.",
                "tone": "calm_directive",
            })
        return ToolCall("lead_to_exit", {"segment": seg_id, "pace": "slow"})

    def _do_takeoff(self, a):
        self._airborne = True
        alt = float(a["altitude_m"])
        if self._phase == "find":
            return ToolResult(
                "takeoff", True,
                f"Airborne at {alt:.1f} m. Phase = FIND. "
                f"{len(self.find_segments)} scout segments remaining before "
                f"Manu's last known location.",
                extra={"phase": "find"},
            )
        return ToolResult(
            "takeoff", True,
            f"Airborne at {alt:.1f} m. Phase = ESCORT. "
            f"{len(self.escort_segments)} segments to the exit.",
            extra={"phase": "escort"},
        )

    def _do_scout_ahead(self, a):
        # The executor tracks *state only*. It does NOT describe the scene —
        # that's the model's job from the current camera frame. The model
        # must NEVER claim to see things the executor told it; only things
        # it can ground in the attached image.

        # ----- find phase: navigate up to Manu -----
        if self._phase == "find":
            seg = self._find_seg()
            if seg is None:
                return ToolResult("scout_ahead", True,
                    "Find phase already complete.")
            self._find_i += 1
            last = self._find_i >= len(self.find_segments)
            if last:
                self._phase = "escort"
                self._manu_located = True
                obs = (
                    f"Drone advanced (segment {self._find_i}/{len(self.find_segments)}). "
                    f"State change: MANU LOCATED → PHASE = ESCORT. "
                    f"{len(self.escort_segments)} escort segments to the exit. "
                    f"Describe Manu and the workbench area from the attached "
                    f"frame yourself."
                )
            else:
                remaining = len(self.find_segments) - self._find_i
                obs = (
                    f"Drone advanced (segment {self._find_i}/{len(self.find_segments)}). "
                    f"Manu not yet reached. {remaining} scout segment(s) remain. "
                    f"Describe what is in the current frame yourself."
                )
            return ToolResult(
                "scout_ahead", True, obs,
                frame_ref=self._frame(seg, "scout"),
                extra={"segment": seg["id"], "phase": "find",
                       "manu_located": last},
            )

        # ----- escort phase: scout the next path segment for Manu -----
        seg = self._escort_seg()
        if seg is None:
            return ToolResult("scout_ahead", True,
                "No further escort segments — already at the exit.")
        # Block the rescout loop: if the model has already scouted the
        # current segment, it must move on to speak_to_survivor (then
        # lead_to_exit) before scouting anything else.
        if seg["id"] in self._scouted_escort:
            return ToolResult(
                "scout_ahead", False,
                f"REJECTED: segment '{seg['id']}' is already scouted. "
                f"Your NEXT call MUST be speak_to_survivor with a directive "
                f"to Manu grounded in the current frame (what you see), then "
                f"after that lead_to_exit on the SAME segment '{seg['id']}'. "
                f"Do NOT call scout_ahead again until the current segment "
                f"has been led successfully.",
            )
        self._scouted_escort.add(seg["id"])
        remaining = len(self.escort_segments) - self._escort_i
        return ToolResult(
            "scout_ahead", True,
            f"Scout of segment '{seg['id']}' complete (escort "
            f"{self._escort_i + 1}/{len(self.escort_segments)}). "
            f"Drone returned to Manu. {remaining} escort segment(s) remain. "
            f"YOUR NEXT TWO CALLS MUST BE, IN ORDER: (1) speak_to_survivor "
            f"with a directive line grounded in the frame you just saw, "
            f"(2) lead_to_exit on segment '{seg['id']}'. "
            f"Do NOT scout again.",
            frame_ref=self._frame(seg, "scout"),
            extra={"segment": seg["id"], "phase": "escort"},
        )

    def _do_speak_to_survivor(self, a):
        if not self._manu_located:
            return ToolResult("speak_to_survivor", False,
                "REJECTED: Manu has not been located yet (still in find "
                "phase). Continue scout_ahead until you reach him.")
        cur = self._escort_seg()
        if cur is not None and cur["id"] not in self._scouted_escort:
            return ToolResult("speak_to_survivor", False,
                f"REJECTED: speak_to_survivor requires you to have just "
                f"scouted the upcoming segment so your words describe a "
                f"real path. Call scout_ahead for '{cur['id']}' first, "
                f"then speak grounded in what that scout showed.")
        # Block speak loops: you've already spoken for this segment, the
        # next call MUST be lead_to_exit.
        if cur is not None and self._spoke_for_seg == cur["id"]:
            return ToolResult("speak_to_survivor", False,
                f"REJECTED: you already spoke for segment '{cur['id']}'. "
                f"Do NOT speak again. Your NEXT call MUST be "
                f"lead_to_exit(segment=<|\"|>{cur['id']}<|\"|>, "
                f"pace=<|\"|>slow<|\"|>).")
        self._spoke_for_seg = cur["id"] if cur else None
        next_id = cur["id"] if cur else "(no segments left)"
        return ToolResult(
            "speak_to_survivor", True,
            f"Spoke aloud (tone={a.get('tone')}): \"{a['utterance']}\". "
            f"Manu heard it. NEXT CALL EXPECTED: "
            f"lead_to_exit(segment='{next_id}', pace='slow'). Do NOT speak again.",
            extra={"spoken": a["utterance"], "tone": a.get("tone")},
        )

    def _do_lead_to_exit(self, a):
        if self._phase == "find":
            return ToolResult("lead_to_exit", False,
                "REJECTED: still in find phase — Manu has not been located. "
                "Use scout_ahead until you reach the workbench, then "
                "speak_to_survivor before leading him.")
        seg = self._escort_seg()
        if seg is None:
            return ToolResult("lead_to_exit", True, "Already at the exit.")
        if seg["id"] not in self._scouted_escort:
            return ToolResult("lead_to_exit", False,
                f"REFUSED: '{seg['id']}' not scouted. scout_ahead first — "
                f"never lead Manu into uninspected space.")
        beh = seg.get("follow_behaviour", "follows")
        frame = self._frame(seg, "lead")
        if beh == "lags":
            seg["follow_behaviour"] = "follows"  # next attempt succeeds
            # Allow a reassuring re-speak after a lag.
            self._spoke_for_seg = None
            return ToolResult(
                "lead_to_exit", True,
                f"follow_state = LAGGING on '{seg['id']}'. Manu hesitated. "
                f"NEXT CALL EXPECTED: speak_to_survivor with a REASSURING "
                f"tone, then lead_to_exit on '{seg['id']}' again.",
                frame_ref=frame,
                extra={"segment": seg["id"], "follow_state": "lagging"},
            )
        self._escort_i += 1
        done = self._escort_i >= len(self.escort_segments)
        # New segment in play (or mission ending) — clear per-segment speak flag.
        self._spoke_for_seg = None
        if done:
            next_hint = ("NEXT CALL EXPECTED: report_finding("
                         "type='egress_reached', ...) then return_to_operator(...).")
        else:
            next_seg = self.escort_segments[self._escort_i]["id"]
            next_hint = (f"NEXT CALL EXPECTED: scout_ahead("
                         f"segment='{next_seg}').")
        return ToolResult(
            "lead_to_exit", True,
            (f"follow_state = FOLLOWING on '{seg['id']}'. " +
             ("Egress complete. " if done
              else f"{len(self.escort_segments) - self._escort_i} segment(s) remain. ") +
             next_hint),
            frame_ref=frame,
            extra={"segment": seg["id"], "follow_state": "following",
                   "egress_complete": done},
        )

    def _do_confirm_survivor_located(self, a):
        """Vision-asserted phase transition — the model has seen Manu in
        the current frame and is telling us to skip the rest of the find
        counter."""
        if self._phase == "escort":
            seg = self._escort_seg()
            next_id = seg["id"] if seg else "the next segment"
            need_scout = seg is not None and seg["id"] not in self._scouted_escort
            return ToolResult("confirm_survivor_located", False,
                f"REJECTED: Manu was already confirmed and the mission is "
                f"in PHASE 2 (escort). Do NOT call confirm_survivor_located "
                f"again. Your NEXT call MUST be "
                + (f"scout_ahead on segment '{next_id}'."
                   if need_scout else
                   f"speak_to_survivor (segment '{next_id}' already scouted), "
                   f"then lead_to_exit."))
        # Accept the assertion. The model is responsible for actually having
        # seen a human in the frame.
        self._phase = "escort"
        self._manu_located = True
        desc = (a.get("description") or "").strip()
        conf = float(a.get("confidence") or 0.0)
        return ToolResult(
            "confirm_survivor_located", True,
            f"PHASE TRANSITION: MANU LOCATED (vision-asserted, "
            f"confidence={conf:.2f}). {desc} "
            f"Switching to PHASE 2 (escort). "
            f"{len(self.escort_segments)} escort segments to the exit. "
            f"Next: scout_ahead the first segment, then speak_to_survivor, "
            f"then lead_to_exit.",
            extra={"phase_transition": True,
                   "manu_description": desc,
                   "confidence": conf},
        )

    def _do_report_finding(self, a):
        return ToolResult(
            "report_finding", True,
            f"Relayed to operator: [{a['type']}] @ {a['location']} "
            f"(conf {a['confidence']:.2f}) — {a['detail']}",
            extra={"finding": a},
        )

    def _do_return_to_operator(self, a):
        return ToolResult(
            "return_to_operator", True,
            f"Manu delivered to the exit. Returning. Reason: "
            f"{a.get('reason','')}. Mission complete.",
        )

    def _do_abort_and_hold(self, a):
        return ToolResult(
            "abort_and_hold", True,
            f"Holding. Reason: {a.get('reason','')}. Awaiting operator.",
        )
