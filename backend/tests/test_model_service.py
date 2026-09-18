import os

import numpy as np
import pytest

from backend.model.model_service import ModelLoadError, ModelService
from backend.simulation.combat_env import AircraftCombatEnv

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo_aircraft_model.zip")


@pytest.fixture
def fresh_service():
    return ModelService(model_path=os.path.abspath(MODEL_PATH))


def test_model_file_exists():
    assert os.path.exists(os.path.abspath(MODEL_PATH)), (
        "models/ppo_aircraft_model.zip is missing - place the trained model there."
    )


def test_model_loads_successfully(fresh_service):
    fresh_service.load()
    assert fresh_service.is_loaded


def test_missing_model_raises_clear_error():
    service = ModelService(model_path="/tmp/does_not_exist_ppo_model.zip")
    with pytest.raises(ModelLoadError):
        service.load()


def test_model_predict_matches_env_action_shape(fresh_service):
    fresh_service.load()
    env = AircraftCombatEnv()
    obs, _ = env.reset()
    action = fresh_service.predict(obs, deterministic=True)
    assert action.shape == env.action_space.shape
    assert np.all(action >= -1.0 - 1e-4) and np.all(action <= 1.0 + 1e-4)


def test_model_predict_is_deterministic(fresh_service):
    fresh_service.load()
    env = AircraftCombatEnv()
    obs, _ = env.reset()
    a1 = fresh_service.predict(obs, deterministic=True)
    a2 = fresh_service.predict(obs, deterministic=True)
    np.testing.assert_allclose(a1, a2)


def test_model_can_run_full_episode(fresh_service):
    """This is the key acceptance check: the actual trained model must be
    able to drive the refactored AircraftCombatEnv end-to-end."""
    fresh_service.load()
    env = AircraftCombatEnv()
    obs, _ = env.reset()
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 300:
        action = fresh_service.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
    assert terminated or truncated
