"""RescueGemma — voice-commanded autonomous indoor search-and-rescue agent.

Built on Gemma 4 for the Kaggle Gemma 4 Good Hackathon.
StratoFirma Autonomy Labs.
"""

from .schemas import TOOLS, ToolCall, ToolResult
from .sim_executor import SimExecutor
from .escort_executor import EscortExecutor
from .planner import (
    run_mission,
    GemmaBackend,
    OllamaBackend,
    LMStudioBackend,
    LiteRTBackend,
    RoutedBackend,
    ScriptedBackend,
    EscortScriptedBackend,
    Turn,
    SYSTEM_PROMPT,
    ESCORT_SYSTEM_PROMPT,
)

__all__ = [
    "TOOLS", "ToolCall", "ToolResult",
    "SimExecutor", "EscortExecutor", "run_mission",
    "GemmaBackend", "OllamaBackend", "LMStudioBackend", "LiteRTBackend",
    "RoutedBackend",
    "ScriptedBackend", "EscortScriptedBackend", "Turn",
    "SYSTEM_PROMPT", "ESCORT_SYSTEM_PROMPT",
]
