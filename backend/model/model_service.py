"""
ModelService - loads the trained Stable-Baselines3 PPO model exactly once
and serves deterministic inference to the simulation manager.

The model file is the actual artifact trained in the original Colab
notebook (`ppo_aircraft_model.zip`, trained for ~500k timesteps with
`PPO("MlpPolicy", ...)` against `AircraftCombatEnv`). This service does
NOT retrain, fine-tune, or otherwise modify the model. It only loads it
and calls `.predict()`.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("aircraft_combat.model_service")


class ModelLoadError(RuntimeError):
    """Raised when the PPO model file is missing, corrupted, or incompatible."""


class ModelService:
    """Thread-safe singleton wrapper around a loaded SB3 PPO model."""

    _instance: Optional["ModelService"] = None
    _lock = threading.Lock()

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    default_path = os.environ.get(
                        "PPO_MODEL_PATH",
                        str(Path(__file__).resolve().parent.parent.parent / "models" / "ppo_aircraft_model.zip"),
                    )
                    cls._instance = cls(default_path)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the PPO model from disk. Safe to call multiple times."""
        with self._load_lock:
            if self._model is not None:
                return

            model_file = Path(self.model_path)
            if not model_file.exists():
                raise ModelLoadError(
                    f"PPO model file not found at '{self.model_path}'. "
                    f"Place the trained 'ppo_aircraft_model.zip' in the models/ "
                    f"directory or set the PPO_MODEL_PATH environment variable."
                )

            try:
                # Imported lazily so the rest of the API can start up (and
                # report a clean error) even if torch/sb3 aren't installed yet.
                from stable_baselines3 import PPO
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError(
                    "stable-baselines3 is not installed. Run "
                    "`pip install -r backend/requirements.txt`."
                ) from exc

            try:
                self._model = PPO.load(str(model_file), device="cpu")
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load PPO model from '{self.model_path}': {exc}"
                ) from exc

            # Sanity-check the model's spaces against the environment we run it in.
            from backend.simulation.combat_env import AircraftCombatEnv

            expected_env = AircraftCombatEnv()
            if tuple(self._model.observation_space.shape) != tuple(expected_env.observation_space.shape):
                raise ModelLoadError(
                    "Loaded PPO model's observation space "
                    f"{self._model.observation_space.shape} does not match "
                    f"AircraftCombatEnv's observation space "
                    f"{expected_env.observation_space.shape}. Refusing to run "
                    "inference with a mismatched model/environment pair."
                )
            if tuple(self._model.action_space.shape) != tuple(expected_env.action_space.shape):
                raise ModelLoadError(
                    "Loaded PPO model's action space "
                    f"{self._model.action_space.shape} does not match "
                    f"AircraftCombatEnv's action space {expected_env.action_space.shape}."
                )

            logger.info(
                "PPO model loaded from %s (obs_shape=%s, action_shape=%s)",
                self.model_path,
                self._model.observation_space.shape,
                self._model.action_space.shape,
            )

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Run real PPO inference. Raises if the model isn't loaded."""
        if self._model is None:
            raise ModelLoadError("PPO model is not loaded. Call load() first.")
        action, _state = self._model.predict(observation, deterministic=deterministic)
        return action

    def unload(self) -> None:
        """Used only by tests to reset the singleton state."""
        self._model = None
