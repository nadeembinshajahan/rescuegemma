"""
RescueGemma — Simulated flight executor.

This is the *body* of the drone for the demo. It satisfies exactly the same
contract as the real PX4/MAVLink flight bridge (see schemas.ToolResult), so
the planner driving it is identical to the one that will run on hardware.

What it does:
  * Maintains a simple kinematic pose in local NED.
  * Maps the recorded phone-walkthrough frames onto a virtual floorplan so
    that when the planner "flies into" a room, the executor returns the real
    frame captured there during the walkthrough.
  * Returns honest natural-language observations the planner reasons over.

It does NOT fake perception. The frames are real footage; Gemma genuinely
analyses them. The only thing simulated is the flight dynamics.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from .schemas import ToolCall, ToolResult, validate_call


@dataclass
class _Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0      # NED down; negative = airborne
    yaw: float = 0.0
    armed: bool = False


class SimExecutor:
    """Consumes ToolCalls, returns ToolResults with real frames.

    floorplan.json describes the building the operator walked:

        {
          "rooms": [
            {
              "id": "hallway",
              "center": [0, 0],
              "frames": ["hallway_01.jpg", "hallway_02.jpg"],
              "tags": ["corridor", "doors on both sides"],
              "survivor": false
            },
            {
              "id": "kids_room",
              "center": [4.0, 2.0],
              "frames": ["kids_room_01.jpg", "kids_room_02.jpg"],
              "tags": ["dinosaur posters", "single bed", "small backpack"],
              "survivor": true,
              "survivor_state": "prone, near window, not responding to voice"
            }
          ]
        }

    The executor never tells the planner which room has the survivor. The
    planner must ground the operator's description against the returned
    frames and decide. That is the part being judged.
    """

    def __init__(self, floorplan_path: str, frames_dir: str):
        with open(floorplan_path) as fh:
            self.plan = json.load(fh)
        self.frames_dir = frames_dir
        self.pose = _Pose()
        self.rooms = {r["id"]: r for r in self.plan["rooms"]}
        self._room_order = [r["id"] for r in self.plan["rooms"]]
        self._current_room = self._room_order[0]
        self._visited: set[str] = set()
        self._frame_cursor: dict[str, int] = {}

    # ----- helpers --------------------------------------------------------

    def _next_frame(self, room_id: str) -> str | None:
        room = self.rooms[room_id]
        frames = room.get("frames", [])
        if not frames:
            return None
        i = self._frame_cursor.get(room_id, 0)
        frame = frames[min(i, len(frames) - 1)]
        self._frame_cursor[room_id] = i + 1
        path = os.path.join(self.frames_dir, frame)
        return path if os.path.exists(path) else frame  # tolerate missing in dry runs

    def _pose_dict(self) -> dict[str, float]:
        return {"x": self.pose.x, "y": self.pose.y, "z": self.pose.z, "yaw": self.pose.yaw}

    def _move_to_room(self, room_id: str) -> None:
        cx, cy = self.rooms[room_id]["center"]
        self.pose.x, self.pose.y = float(cx), float(cy)
        self._current_room = room_id
        self._visited.add(room_id)

    # ----- the dispatch ---------------------------------------------------

    def execute(self, call: ToolCall) -> ToolResult:
        ok, msg = validate_call(call)
        if not ok:
            return ToolResult(call.name, False, f"REJECTED: {msg}")

        handler = getattr(self, f"_do_{call.name}", None)
        if handler is None:
            return ToolResult(call.name, False, f"No executor implementation for '{call.name}'")
        return handler(call.arguments)

    # ----- per-tool implementations --------------------------------------

    def _do_takeoff(self, args):
        alt = float(args["altitude_m"])
        self.pose.armed = True
        self.pose.z = -alt
        return ToolResult(
            "takeoff", True,
            f"Airborne and stable at {alt:.1f} m. Position: hallway entry. Ready for tasking.",
            pose=self._pose_dict(),
        )

    def _do_goto_waypoint(self, args):
        if not self.pose.armed:
            return ToolResult("goto_waypoint", False, "Cannot navigate: not airborne. Call takeoff first.")
        self.pose.x = float(args["x"])
        self.pose.y = float(args["y"])
        self.pose.z = float(args["z"])
        # snap to nearest room for frame lookup
        nearest = min(
            self.rooms.values(),
            key=lambda r: math.hypot(r["center"][0] - self.pose.x, r["center"][1] - self.pose.y),
        )
        self._current_room = nearest["id"]
        reason = args.get("reason", "")
        return ToolResult(
            "goto_waypoint", True,
            f"At local ({self.pose.x:.1f}, {self.pose.y:.1f}). Nearest area: '{nearest['id']}'. {reason}".strip(),
            pose=self._pose_dict(),
            frame_ref=self._next_frame(nearest["id"]),
        )

    def _do_search_floor(self, args):
        # Returns the list of rooms with their visible tags so the planner can
        # decide ordering. Does NOT reveal survivor flags.
        floor = args["floor_id"]
        manifest = [
            {"room": r["id"], "visible_features": r.get("tags", [])}
            for r in self.plan["rooms"]
        ]
        hint = args.get("rooms_hint", [])
        return ToolResult(
            "search_floor", True,
            (f"Floor '{floor}' has {len(manifest)} navigable areas. "
             f"Visible features per area: {json.dumps(manifest)}. "
             f"Operator priority hints: {hint}. "
             f"Decide an order and explore_room each in turn."),
            pose=self._pose_dict(),
        )

    def _do_explore_room(self, args):
        # Advance to the next unvisited room in spatial order.
        unvisited = [r for r in self._room_order if r not in self._visited]
        target = unvisited[0] if unvisited else self._current_room
        self._move_to_room(target)
        room = self.rooms[target]
        match_desc = args.get("match_description", "")
        frame = self._next_frame(target)
        obs = (
            f"Entered area '{target}'. Visible features: {room.get('tags', [])}. "
            f"Camera frame attached for your analysis."
        )
        if match_desc:
            obs += (
                f" You asked to match: '{match_desc}'. "
                f"Ground this against the attached frame yourself — do not assume."
            )
        return ToolResult(
            "explore_room", True, obs,
            pose=self._pose_dict(),
            frame_ref=frame,
            extra={"room_id": target},
        )

    def _do_hover_and_scan(self, args):
        room = self.rooms[self._current_room]
        looking = args.get("looking_for", "")
        # Serve a second/closer frame of the same room if available.
        frame = self._next_frame(self._current_room)
        obs = (
            f"Held position in '{self._current_room}' for {args['duration_s']}s. "
            f"Close-range frame attached."
        )
        if looking:
            obs += f" Operator question to resolve from the frame: '{looking}'."
        # Provide the staged survivor state ONLY as ground truth in extra, so
        # an eval harness can score the planner's vision call — never in the
        # natural-language observation the planner reads.
        extra = {}
        if room.get("survivor"):
            extra["_ground_truth"] = room.get("survivor_state", "person present")
        return ToolResult(
            "hover_and_scan", True, obs,
            pose=self._pose_dict(),
            frame_ref=frame,
            extra=extra,
        )

    def _do_report_finding(self, args):
        # The executor just acknowledges; the operator UI would surface this.
        return ToolResult(
            "report_finding", True,
            f"Finding relayed to operator: [{args['type']}] @ {args['location']} "
            f"(conf {args['confidence']:.2f}) — {args['detail']}",
            pose=self._pose_dict(),
            extra={"finding": args},
        )

    def _do_return_to_operator(self, args):
        self._move_to_room(self._room_order[0])
        return ToolResult(
            "return_to_operator", True,
            f"Returning to operator. Reason: {args.get('reason','')}. Mission loop complete.",
            pose=self._pose_dict(),
        )

    def _do_abort_and_hold(self, args):
        return ToolResult(
            "abort_and_hold", True,
            f"Holding position. Reason: {args.get('reason','')}. Awaiting operator.",
            pose=self._pose_dict(),
        )
