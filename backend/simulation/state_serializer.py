"""
Converts the raw numpy state living inside an `AircraftCombatEnv` instance
into plain-Python / JSON-serializable dictionaries for the WebSocket and
REST layers.

No values are invented here: every field is read directly off the live
environment object (`env.agent_pos`, `env.missile_fired`, etc.) or off the
action array that was actually passed to `env.step()`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from backend.simulation.combat_env import AircraftCombatEnv


def _vec3(v: Optional[np.ndarray]) -> list:
    if v is None:
        return [0.0, 0.0, 0.0]
    return [float(v[0]), float(v[1]), float(v[2])]


def serialize_entity(position: np.ndarray, velocity: np.ndarray, direction: np.ndarray) -> Dict[str, Any]:
    return {
        "position": _vec3(position),
        "velocity": _vec3(velocity),
        "direction": _vec3(direction),
    }


def serialize_action(raw_action: np.ndarray) -> Dict[str, Any]:
    """Decode the raw [-1, 1]^4 PPO action the same way `env.step()` does."""
    throttle = float((raw_action[0] + 1) / 2)
    pitch = float(raw_action[1])
    yaw = float(raw_action[2])
    fire = (raw_action[3] + 1) / 2 > 0.5
    return {"throttle": throttle, "pitch": pitch, "yaw": yaw, "fire": bool(fire)}


def serialize_state(
    env: AircraftCombatEnv,
    *,
    raw_action: np.ndarray,
    reward: float,
    episode_reward: float,
    status: str,
    terminated: bool,
    truncated: bool,
    max_steps: int = 250,
) -> Dict[str, Any]:
    """Build the exact JSON shape documented in the project spec's Phase 5."""

    distance = float(np.linalg.norm(env.agent_pos - env.enemy_pos))

    missile_state = {
        "active": bool(env.missile_fired),
        "position": _vec3(env.missile_pos) if env.missile_fired else None,
    }
    enemy_missile_state = {
        "active": bool(env.enemy_missile_fired),
        "position": _vec3(env.enemy_missile_pos) if env.enemy_missile_fired else None,
    }

    return {
        "type": "state",
        "step": int(env.steps),
        "max_steps": max_steps,
        "status": status,
        "reward": round(float(reward), 4),
        "episode_reward": round(float(episode_reward), 4),
        "distance": round(distance, 4),
        "agent": serialize_entity(env.agent_pos, env.agent_vel, env.agent_dir),
        "enemy": serialize_entity(env.enemy_pos, env.enemy_vel, env.enemy_dir),
        "missile": missile_state,
        "enemy_missile": enemy_missile_state,
        "action": serialize_action(raw_action),
        "agent_hit": bool(env.agent_hit),
        "enemy_hit": bool(env.enemy_hit),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
