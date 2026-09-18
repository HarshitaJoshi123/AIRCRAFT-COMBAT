import numpy as np
import pytest

from backend.simulation.combat_env import AircraftCombatEnv


def test_reset_returns_valid_observation():
    env = AircraftCombatEnv()
    obs, info = env.reset()
    assert obs.shape == (13,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_reset_enforces_minimum_distance():
    env = AircraftCombatEnv()
    for _ in range(20):
        env.reset()
        dist = np.linalg.norm(env.agent_pos - env.enemy_pos)
        assert dist >= 120.0 - 1e-6


def test_action_space_shape_and_bounds():
    env = AircraftCombatEnv()
    assert env.action_space.shape == (4,)
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)


def test_observation_space_shape():
    env = AircraftCombatEnv()
    assert env.observation_space.shape == (13,)


def test_step_returns_correct_tuple_shape():
    env = AircraftCombatEnv()
    env.reset()
    action = np.array([0.5, 0.0, 0.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (13,)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_truncates_at_250_steps():
    env = AircraftCombatEnv()
    env.reset()
    # No-fire action to avoid an early terminal hit, though a random hit is
    # still theoretically possible; loop breaks on either signal.
    action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    truncated = terminated = False
    steps = 0
    while not (terminated or truncated) and steps < 400:
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
    assert steps <= 250
    assert terminated or truncated


def test_agent_and_enemy_positions_stay_within_bounds():
    env = AircraftCombatEnv()
    env.reset()
    action = np.array([1.0, 0.3, -0.3, -1.0], dtype=np.float32)
    for _ in range(50):
        env.step(action)
        assert np.all(env.agent_pos >= 0) and np.all(env.agent_pos <= env.space_limit)
        assert np.all(env.enemy_pos >= 0) and np.all(env.enemy_pos <= env.space_limit)


def test_missile_fires_when_fire_action_high():
    env = AircraftCombatEnv()
    env.reset()
    action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # fire = (1+1)/2 = 1.0 > 0.5
    env.step(action)
    assert env.missile_fired is True
    assert env.missile_pos is not None


def test_missile_does_not_fire_when_fire_action_low():
    env = AircraftCombatEnv()
    env.reset()
    action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)  # fire = 0.0 <= 0.5
    env.step(action)
    assert env.missile_fired is False
