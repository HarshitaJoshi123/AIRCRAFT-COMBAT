import asyncio
import os

import pytest

from backend.model.model_service import ModelService
from backend.simulation.simulation_manager import (
    SimulationBusyError,
    SimulationManager,
    SimulationNotRunningError,
)

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "ppo_aircraft_model.zip"))

pytestmark = pytest.mark.asyncio


@pytest.fixture
def manager():
    service = ModelService(model_path=MODEL_PATH)
    service.load()
    return SimulationManager(service, max_steps=250, tick_hz=1000.0)  # fast tick for tests


async def test_reset_sets_idle_status(manager):
    state = await manager.reset()
    assert state["status"] == "IDLE"
    assert state["step"] == 0


async def test_start_transitions_to_active(manager):
    snap = await manager.start()
    assert snap["status"] == "ACTIVE"
    await manager.stop()


async def test_double_start_raises_busy_error(manager):
    await manager.start()
    with pytest.raises(SimulationBusyError):
        await manager.start()
    await manager.stop()


async def test_pause_without_active_raises(manager):
    await manager.reset()
    with pytest.raises(SimulationNotRunningError):
        await manager.pause()


async def test_pause_and_resume_cycle(manager):
    await manager.start()
    paused = await manager.pause()
    assert paused["status"] == "PAUSED"
    resumed = await manager.resume()
    assert resumed["status"] == "ACTIVE"
    await manager.stop()


async def test_simulation_steps_advance_episode(manager):
    await manager.start()
    await asyncio.sleep(0.2)  # let the background loop tick a few times
    snap = manager.snapshot()
    assert snap["step"] > 0
    await manager.stop()


async def test_reset_while_active_raises(manager):
    await manager.start()
    with pytest.raises(SimulationBusyError):
        await manager.reset()
    await manager.stop()
