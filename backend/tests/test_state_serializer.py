import numpy as np

from backend.simulation.combat_env import AircraftCombatEnv
from backend.simulation.state_serializer import serialize_action, serialize_state


def test_serialize_action_matches_env_decoding():
    raw = np.array([0.4, -0.5, 0.2, 0.9], dtype=np.float32)
    decoded = serialize_action(raw)
    assert abs(decoded["throttle"] - (0.4 + 1) / 2) < 1e-6
    assert decoded["pitch"] == -0.5
    assert decoded["yaw"] == 0.2
    assert decoded["fire"] is True  # (0.9+1)/2 = 0.95 > 0.5


def test_serialize_state_shape():
    env = AircraftCombatEnv()
    env.reset()
    raw_action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(raw_action)
    state = serialize_state(
        env,
        raw_action=raw_action,
        reward=reward,
        episode_reward=reward,
        status="ACTIVE",
        terminated=terminated,
        truncated=truncated,
    )
    assert state["type"] == "state"
    assert len(state["agent"]["position"]) == 3
    assert len(state["enemy"]["velocity"]) == 3
    assert "missile" in state and "enemy_missile" in state
    assert state["max_steps"] == 250


def test_missile_position_none_when_inactive():
    env = AircraftCombatEnv()
    env.reset()
    raw_action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, _ = env.step(raw_action)
    state = serialize_state(
        env, raw_action=raw_action, reward=reward, episode_reward=reward,
        status="ACTIVE", terminated=terminated, truncated=truncated,
    )
    assert state["missile"]["active"] is False
    assert state["missile"]["position"] is None
