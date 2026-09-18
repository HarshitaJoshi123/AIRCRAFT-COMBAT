"""
AircraftCombatEnv - production module.

This is a DIRECT, BEHAVIOR-PRESERVING port of the `AircraftCombatEnv` class
from the original Google Colab notebook (`AIRCRAFT_COMBAT_NEW.ipynb`).

RULES FOLLOWED:
- Observation space: unchanged, Box(-inf, inf, shape=(13,), float32)
- Action space: unchanged, Box(-1, 1, shape=(4,), float32)
- Reward shaping: unchanged, term-for-term identical to the notebook.
- Movement / missile / termination logic: unchanged.
- The ONLY things removed from the original class are pure-visualization
  concerns (matplotlib), which never lived inside the env class itself in
  the notebook, so nothing was actually removed here - this file matches
  the notebook's `AircraftCombatEnv` 1:1.

Do not modify the numeric constants or the ordering of operations in
`step()` / `reset()` / `_get_obs()` - the trained model
(`models/ppo_aircraft_model.zip`) was trained against this exact
transition function and observation layout, and any behavioral drift will
silently invalidate the trained policy.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AircraftCombatEnv(gym.Env):
    """13-dim observation / 4-dim continuous action dogfight environment.

    Observation (13,):
        [agent_pos(3), agent_vel(3), enemy_pos(3), enemy_vel(3), distance(1)]

    Action (4,) each in [-1, 1]:
        [throttle_raw, pitch, yaw, fire_raw]
        throttle = (throttle_raw + 1) / 2   -> [0, 1]
        fire     = (fire_raw + 1) / 2       -> [0, 1], fires when > 0.5
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.space_limit = 100.0
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.agent_dir = np.array([0.0, 0.0, 1.0])  # initially facing forward along Z
        # flag to detect when an enemy missile just expired
        self.missile_just_expired = False

    def random_unit_vector(self):
        vec = np.random.normal(size=3)  # sample from Gaussian
        return vec / np.linalg.norm(vec)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        min_distance = 120.0
        while True:
            self.agent_pos = np.random.uniform(5, 95, size=3)
            self.enemy_pos = np.random.uniform(5, 95, size=3)
            if np.linalg.norm(self.agent_pos - self.enemy_pos) >= min_distance:
                break

        self.agent_dir = self.random_unit_vector()
        self.enemy_dir = self.random_unit_vector()
        self.agent_vel = np.zeros(3)
        self.enemy_vel = np.zeros(3)

        self.steps = 0

        self.missile_fired = False
        self.missile_pos = None
        self.missile_dir = None

        self.enemy_missile_fired = False
        self.enemy_missile_pos = None
        self.enemy_missile_dir = None
        self.missile_just_expired = False

        self.agent_hit = False
        self.enemy_hit = False

        self.path_history = [self.agent_pos.copy()]
        self.enemy_path_history = [self.enemy_pos.copy()]

        return self._get_obs(), {}

    def _get_obs(self):
        dist = np.linalg.norm(self.agent_pos - self.enemy_pos)
        return np.concatenate([
            self.agent_pos.astype(np.float32),
            self.agent_vel.astype(np.float32),
            self.enemy_pos.astype(np.float32),
            self.enemy_vel.astype(np.float32),
            np.array([dist], dtype=np.float32)
        ])

    def step(self, action):
        self.agent_hit = False
        self.enemy_hit = False
        self.missile_just_expired = False

        # --- Parse Actions ---
        throttle = (action[0] + 1) / 2
        pitch = action[1]
        yaw = action[2]
        fire = (action[3] + 1) / 2

        # --- Update Agent Direction using Pitch and Yaw ---
        yaw_angle = yaw * 0.05
        pitch_angle = pitch * 0.05

        cos_p = np.cos(pitch_angle)
        sin_p = np.sin(pitch_angle)
        rotation_pitch = np.array([
            [cos_p, 0, sin_p],
            [0,     1, 0    ],
            [-sin_p, 0, cos_p]
        ])

        cos_y = np.cos(yaw_angle)
        sin_y = np.sin(yaw_angle)
        rotation_yaw = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])

        self.agent_dir = rotation_yaw @ rotation_pitch @ self.agent_dir
        self.agent_dir /= np.linalg.norm(self.agent_dir)

        # --- Update Positions ---
        self.agent_vel = throttle * self.agent_dir
        self.agent_pos += self.agent_vel
        self.enemy_pos += self.enemy_vel

        # === LEAD PURSUIT FOR ENEMY MOVEMENT (with safe distance) ===
        relative_pos = self.agent_pos - self.enemy_pos
        dist = np.linalg.norm(relative_pos)
        closing_speed = 1.5  # missile/closure speed for timing

        # Predict where agent will be
        time_to_intercept = dist / closing_speed
        intercept_point = self.agent_pos + self.agent_vel * time_to_intercept

        # Direction toward intercept point
        desired_direction = (intercept_point - self.enemy_pos)
        desired_direction /= np.linalg.norm(desired_direction)

        # Enforce a minimum safe distance so enemy doesn't collide
        safe_distance = 8.0
        if dist > safe_distance:
            enemy_speed = 0.8
            self.enemy_vel = desired_direction * enemy_speed
        else:
            self.enemy_vel = np.zeros(3)

        # Update enemy orientation
        if np.linalg.norm(self.enemy_vel) > 1e-6:
            self.enemy_dir = self.enemy_vel / np.linalg.norm(self.enemy_vel)

        # Clamp to bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.space_limit)
        self.enemy_pos = np.clip(self.enemy_pos, 0, self.space_limit)

        self.steps += 1
        terminated = False
        truncated = False
        reward = 0

        # --- Fire Agent Missile ---
        if fire > 0.5 and not self.missile_fired:
            self.missile_fired = True
            self.missile_pos = np.copy(self.agent_pos)
            direction = self.enemy_pos - self.agent_pos
            self.missile_dir = direction / (np.linalg.norm(direction) + 1e-8)

        if self.missile_fired:
            self.missile_pos += self.missile_dir * 2.0
            if np.linalg.norm(self.missile_pos - self.enemy_pos) < 10:
                reward += 100
                terminated = True
                self.enemy_hit = True
            elif self.steps > 200 or np.any(self.missile_pos < 0) or np.any(self.missile_pos > self.space_limit):
                self.missile_fired = False

        # --- Fire Enemy Missile ---
        if not self.enemy_missile_fired and dist < 60:
            aim_dot = np.dot(desired_direction, (self.agent_pos - self.enemy_pos) / (dist + 1e-6))
            # --- When firing, compute lead pursuit intercept for missile ---
            self.enemy_missile_fired = True
            self.enemy_missile_pos = self.enemy_pos.copy()

            # Missile lead pursuit
            missile_speed = 2.0
            relative_pos = self.agent_pos - self.enemy_pos
            distance = np.linalg.norm(relative_pos)
            time_to_hit = distance / missile_speed
            future_agent_pos = self.agent_pos + self.agent_vel * time_to_hit

            aim_vector = future_agent_pos - self.enemy_pos
            self.enemy_missile_dir = aim_vector / (np.linalg.norm(aim_vector) + 1e-8)

        if self.enemy_missile_fired:
            self.enemy_missile_pos += self.enemy_missile_dir * 2.0
            if np.linalg.norm(self.enemy_missile_pos - self.agent_pos) < 10:
                reward -= 100
                terminated = True
                self.agent_hit = True
                self.enemy_missile_fired = False
            elif self.steps > 250 or np.any(self.enemy_missile_pos < 0) or np.any(self.enemy_missile_pos > self.space_limit):
                self.enemy_missile_fired = False
                self.missile_just_expired = True

        # ================= IMPORTANT FIX =================
        # STOP if game ended (VERY IMPORTANT)
        if terminated:
            return self._get_obs(), reward, terminated, truncated, {}

        # --- Reward Shaping (final scheme) ---
        # Continuous proximity
        reward += (1 - dist / self.space_limit) * 5

        # Missile fire encouragement & strategic fire
        if fire > 0.5:
            reward += 0.5
            if 30 <= dist <= 60:
                reward += 5
            elif dist > 70:
                reward -= 2

        # Speed incentive
        spd = np.linalg.norm(self.agent_vel)
        reward += 2.0 * spd

        # Survival bonus
        reward += 0.2

        # Enemy missile dodge bonus
        if self.missile_just_expired:
            reward += 10

        # Danger-zone penalty
        if self.enemy_missile_fired:
            dm = np.linalg.norm(self.agent_pos - self.enemy_missile_pos)
            if dm < 15:
                reward -= 5

        # Idle penalty
        if spd < 0.05:
            reward -= 1

        if self.steps >= 250:
            truncated = True

        # inside your reward shaping block, after all other terms:
        margin = 10.0
        dist_to_edge = min(
            self.agent_pos[0], self.space_limit - self.agent_pos[0],
            self.agent_pos[1], self.space_limit - self.agent_pos[1],
            self.agent_pos[2], self.space_limit - self.agent_pos[2]
        )
        if dist_to_edge < margin:
            reward -= (margin - dist_to_edge) * 0.5  # stronger penalty closer in

        # --- Store Paths for Visualization ---
        self.path_history.append(self.agent_pos.copy())
        self.enemy_path_history.append(self.enemy_pos.copy())

        return self._get_obs(), reward, terminated, truncated, {}
