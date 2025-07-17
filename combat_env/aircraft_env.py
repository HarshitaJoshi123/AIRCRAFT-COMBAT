import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AircraftCombatEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.space_limit = 100.0
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.agent_dir = np.array([0.0, 0.0, 1.0])  # initially facing forward along Z

    def random_unit_vector(self):
        vec = np.random.normal(size=3)
        return vec / np.linalg.norm(vec)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        min_distance = 80.0
        while True:
            self.agent_pos = np.random.uniform(low=5, high=95, size=3)
            self.enemy_pos = np.random.uniform(low=5, high=95, size=3)
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

        throttle = (action[0] + 1) / 2
        pitch = action[1]
        yaw = action[2]
        fire = (action[3] + 1) / 2

        yaw_angle = yaw * 0.05
        pitch_angle = pitch * 0.05

        rotation_pitch = np.array([
            [np.cos(pitch_angle), 0, np.sin(pitch_angle)],
            [0, 1, 0],
            [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]
        ])

        rotation_yaw = np.array([
            [np.cos(yaw_angle), -np.sin(yaw_angle), 0],
            [np.sin(yaw_angle), np.cos(yaw_angle), 0],
            [0, 0, 1]
        ])

        self.agent_dir = rotation_yaw @ rotation_pitch @ self.agent_dir
        self.agent_dir /= np.linalg.norm(self.agent_dir)

        self.agent_vel = throttle * self.agent_dir
        self.agent_pos += self.agent_vel
        self.enemy_pos += self.enemy_vel

        relative_pos = self.agent_pos - self.enemy_pos
        distance = np.linalg.norm(relative_pos)
        agent_speed = np.linalg.norm(self.agent_vel) + 1e-6
        time_to_intercept = distance / 1.5
        intercept_point = self.agent_pos + self.agent_vel * time_to_intercept

        desired_direction = intercept_point - self.enemy_pos
        desired_direction /= np.linalg.norm(desired_direction)

        self.enemy_vel = desired_direction * 0.8
        if np.linalg.norm(self.enemy_vel) > 1e-6:
            self.enemy_dir = self.enemy_vel / np.linalg.norm(self.enemy_vel)

        self.agent_pos = np.clip(self.agent_pos, 0, self.space_limit)
        self.enemy_pos = np.clip(self.enemy_pos, 0, self.space_limit)

        self.steps += 1
        terminated = False
        truncated = False
        reward = 0
        dist_to_enemy = np.linalg.norm(self.agent_pos - self.enemy_pos)

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
        if not self.enemy_missile_fired and distance < 60:
            aim_dot = np.dot(desired_direction, (self.agent_pos - self.enemy_pos) / (distance + 1e-6))
            if aim_dot > 0.95:
                self.enemy_missile_fired = True
                self.enemy_missile_pos = self.enemy_pos.copy()
                self.enemy_missile_dir = desired_direction

        if self.enemy_missile_fired:
            self.enemy_missile_pos += self.enemy_missile_dir * 2.0
            if np.linalg.norm(self.enemy_missile_pos - self.agent_pos) < 10:
                reward -= 100
                terminated = True
                self.agent_hit = True
            elif self.steps > 200 or np.any(self.enemy_missile_pos < 0) or np.any(self.enemy_missile_pos > self.space_limit):
                self.enemy_missile_fired = False

        # --- Reward Shaping ---
        reward += (1 - dist_to_enemy / self.space_limit) * 5
        if dist_to_enemy < 10:
            reward += 70
        elif dist_to_enemy < 20:
            reward += 40
        elif dist_to_enemy < 50:
            reward += 20

        if fire > 0.5:
            reward += 0.5
        if self.steps >= 200:
            truncated = True
        reward += np.linalg.norm(self.agent_vel) * 2.0

        self.path_history.append(self.agent_pos.copy())
        self.enemy_path_history.append(self.enemy_pos.copy())

        return self._get_obs(), reward, terminated, truncated, {}
