"""
RescueGemma — Function-calling schemas (the drone primitive contract).

This module is the single source of truth for the interface between the
Gemma 4 planner and the flight executor. The SAME schema is used by:

  * the simulated executor (sim_executor.py)  — used for the demo / Kaggle notebook
  * the real flight bridge (flight_bridge.py) — Thomas's MAVLink/PX4 wrapper

Because both implementations satisfy this identical contract, the demo is
not a mock: the planner code path is byte-for-byte the same one that will
run on the Jetson against PX4. Only the body (executor) is swapped.

Gemma 4 has native function calling. We expose these as the tool list and
let the model emit structured calls. Each tool is a thin, typed wrapper
around a PX4 offboard / MAVLink primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enumerations shared across tools
# ---------------------------------------------------------------------------

class SearchStrategy(str, Enum):
    PERIMETER = "perimeter"   # hug walls, good for survivors against edges
    RASTER = "raster"         # lawnmower sweep, good for open floor area
    DOORWAY_FIRST = "doorway_first"  # peek then enter, lowest-risk entry


class SearchPriority(str, Enum):
    AUDIO = "audio"           # bias toward rooms with detected human voice
    MOTION = "motion"         # bias toward visual motion
    THERMAL = "thermal"       # bias toward heat signatures (if payload present)
    SYSTEMATIC = "systematic"  # plain spatial coverage, no sensor bias


class FindingType(str, Enum):
    PERSON_RESPONSIVE = "person_responsive"
    PERSON_UNRESPONSIVE = "person_unresponsive"
    PERSON_UNKNOWN = "person_unknown"
    HAZARD_FIRE = "hazard_fire"
    HAZARD_SMOKE = "hazard_smoke"
    HAZARD_STRUCTURAL = "hazard_structural"
    HAZARD_STAIRS = "hazard_stairs"          # steep descent / missing railing
    HAZARD_OBSTRUCTION = "hazard_obstruction"  # clutter / debris on the egress path
    HAZARD_NARROW = "hazard_narrow"          # constriction the person must slow for
    BLOCKED_PATH = "blocked_path"
    AREA_CLEAR = "area_clear"
    EGRESS_REACHED = "egress_reached"        # person delivered to the exit


class GuidanceTone(str, Enum):
    """How the spoken line should land. A trapped person under smoke needs
    calm and short, not verbose. The model picks tone per situation."""
    CALM_DIRECTIVE = "calm_directive"   # "Follow me. Slowly."
    URGENT = "urgent"                   # "Stop. Don't move forward."
    REASSURING = "reassuring"           # "You're doing fine. Almost there."


# ---------------------------------------------------------------------------
# The tool definitions, in the JSON-schema shape Gemma 4 expects for native
# function calling. Keep descriptions short and operational — the model reads
# them, so they double as the planner's spec.
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "takeoff",
        "description": "Arm and ascend to a target altitude above the takeoff point. Must be called before any navigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "altitude_m": {
                    "type": "number",
                    "description": "Target altitude in metres above launch. Indoor: 1.2-2.0 typical.",
                    "minimum": 0.3,
                    "maximum": 4.0,
                }
            },
            "required": ["altitude_m"],
        },
    },
    {
        "name": "goto_waypoint",
        "description": "Fly to a local coordinate. Frame is local NED relative to takeoff. Use for moving between known points (e.g. into a hallway, toward a doorway).",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "North metres from launch."},
                "y": {"type": "number", "description": "East metres from launch."},
                "z": {"type": "number", "description": "Down metres from launch (negative = up). Indoor: ~ -1.5."},
                "reason": {"type": "string", "description": "One short phrase: why go here. Logged for the operator."},
            },
            "required": ["x", "y", "z"],
        },
    },
    {
        "name": "search_floor",
        "description": "Execute a coordinated search of a named floor/area, room by room. The executor handles intra-floor pathing; the planner only sets intent and priority.",
        "parameters": {
            "type": "object",
            "properties": {
                "floor_id": {"type": "string", "description": "Operator's label for the area, e.g. 'second floor', 'east wing'."},
                "priority": {
                    "type": "string",
                    "enum": [p.value for p in SearchPriority],
                    "description": "What to bias the search toward.",
                },
                "rooms_hint": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional operator descriptions of rooms to prioritise, e.g. ['room with dinosaur posters'].",
                },
            },
            "required": ["floor_id", "priority"],
        },
    },
    {
        "name": "explore_room",
        "description": "Enter and clear the current/nearest room using a strategy. Returns observations the planner can reason over.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": [s.value for s in SearchStrategy],
                },
                "match_description": {
                    "type": "string",
                    "description": "Optional: a visual description to look for, grounded against the camera, e.g. 'walls with dinosaur posters'. The executor returns whether the current room matches.",
                },
            },
            "required": ["strategy"],
        },
    },
    {
        "name": "hover_and_scan",
        "description": "Hold position and capture a careful multi-angle visual scan of the current location. Use when something needs a closer look before deciding.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration_s": {"type": "number", "minimum": 1, "maximum": 30},
                "looking_for": {"type": "string", "description": "What the planner wants confirmed, e.g. 'is the person breathing / moving'."},
            },
            "required": ["duration_s"],
        },
    },
    {
        "name": "report_finding",
        "description": "Send a structured finding back to the operator. This is the primary output channel — call it whenever something the operator needs to know is observed.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": [f.value for f in FindingType]},
                "location": {"type": "string", "description": "Human-readable location, e.g. 'upstairs room with dinosaur posters, near the window'."},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "detail": {"type": "string", "description": "One or two sentences in plain language for a non-engineer operator under stress."},
            },
            "required": ["type", "location", "confidence", "detail"],
        },
    },
    {
        "name": "speak_to_survivor",
        "description": "Speak a short spoken line aloud to the person being escorted, via the onboard speaker (Gemma audio-out). Use this to give directions, warn of a hazard you can see in the frame, or reassure. Keep it to one or two short sentences a frightened person under smoke can follow. Generate the words from what you actually see — do not use canned phrases.",
        "parameters": {
            "type": "object",
            "properties": {
                "utterance": {
                    "type": "string",
                    "description": "The exact words to say aloud. Short. Plain. Grounded in the current frame, e.g. 'Stairs right in front of you — no rail on the left. Take them one at a time, I'll wait.'",
                },
                "tone": {
                    "type": "string",
                    "enum": [t.value for t in GuidanceTone],
                },
            },
            "required": ["utterance", "tone"],
        },
    },
    {
        "name": "confirm_survivor_located",
        "description": "Assert that you can SEE the trapped survivor (Manu) in the current camera frame. Call this the instant a human figure appears in the workshop / target area during PHASE 1 (find) — it immediately ends the find phase and flips the mission to PHASE 2 (escort). Do not wait for the find counter to expire if Manu is already visible.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "One short sentence describing where in the frame the person is (e.g. 'a man in a blue shirt seated at the wooden workbench, centre of frame').",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Your confidence that this is a human (not a coat / mannequin / shadow). 0.5+ recommended.",
                },
            },
            "required": ["description", "confidence"],
        },
    },
    {
        "name": "scout_ahead",
        "description": "Fly ahead of the person along the intended egress path, inspect what's coming (next room, the stairwell, a doorway), then the executor returns you to them with a frame of what you saw. Use this BEFORE leading the person into any space you haven't visually cleared. This is the core safety behaviour: check, then commit.",
        "parameters": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "string",
                    "description": "Which path segment to scout, e.g. 'the doorway and corridor ahead', 'top of the staircase'.",
                },
            },
            "required": ["segment"],
        },
    },
    {
        "name": "lead_to_exit",
        "description": "Move forward one egress segment at the person's pace, expecting them to follow. Only call after that segment has been scouted and is clear, and after you've told the person what to do. The executor returns the person's follow state (following / lagging / stopped) and a frame so you can reassess.",
        "parameters": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "string",
                    "description": "The segment being traversed, e.g. 'workbench to the door', 'door to the top of the stairs'.",
                },
                "pace": {
                    "type": "string",
                    "enum": ["slow", "normal"],
                    "description": "Slow when the person is injured, smoke is heavy, or terrain is hazardous.",
                },
            },
            "required": ["segment"],
        },
    },
    {
        "name": "return_to_operator",
        "description": "Abort remaining search and fly back to the launch/operator point. Call when the mission objective is met or on explicit recall.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "required": ["reason"],
        },
    },
    {
        "name": "abort_and_hold",
        "description": "Immediately stop and hold position safely. Use on unrecoverable hazard or ambiguous unsafe state. Conservative default.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "required": ["reason"],
        },
    },
]


# Quick name->schema index for validation in the executor.
TOOLS_BY_NAME: dict[str, dict[str, Any]] = {t["name"]: t for t in TOOLS}


@dataclass
class ToolCall:
    """A single call emitted by the planner."""
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """What the executor hands back to the planner after running a call."""
    name: str
    ok: bool
    observation: str                      # natural-language state for Gemma to read
    pose: dict[str, float] | None = None  # x,y,z,yaw in local NED, if applicable
    frame_ref: str | None = None          # filename of the camera frame to attach
    extra: dict[str, Any] = field(default_factory=dict)


def validate_call(call: ToolCall) -> tuple[bool, str]:
    """Cheap structural validation before the executor acts on a model call.

    Real safety lives in the executor (geofence, battery, link), but catching
    a malformed call here keeps the loop honest and gives Gemma a correctable
    error string instead of a crash.
    """
    schema = TOOLS_BY_NAME.get(call.name)
    if schema is None:
        return False, f"Unknown tool '{call.name}'. Valid tools: {list(TOOLS_BY_NAME)}"
    required = schema["parameters"].get("required", [])
    missing = [r for r in required if r not in call.arguments]
    if missing:
        return False, f"Tool '{call.name}' missing required args: {missing}"
    return True, "ok"
