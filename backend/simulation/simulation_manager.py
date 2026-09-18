"""
SimulationManager owns the single authoritative AircraftCombatEnv instance,
drives it with real PPO actions on a fixed tick, and fans the resulting
state out to any number of WebSocket subscribers.

Only one simulation runs at a time by design (the spec calls this out as
"preventing conflicting simulations") - this keeps the PPO model call graph
simple and matches a single-viewport demo application. Attempting to start
a second simulation while one is ACTIVE/PAUSED raises SimulationBusyError,
which the API layer turns into a 409 response.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from backend.model.model_service import ModelLoadError, ModelService
from backend.simulation.combat_env import AircraftCombatEnv
from backend.simulation.state_serializer import serialize_state

logger = logging.getLogger("aircraft_combat.simulation_manager")

MAX_STEPS = 250
TICK_HZ = 12.0  # backend simulation update rate; see Phase 9 (frontend interpolates between ticks)


class SimulationBusyError(RuntimeError):
    pass


class SimulationNotRunningError(RuntimeError):
    pass


@dataclass
class _RunState:
    status: str = "IDLE"  # IDLE | ACTIVE | PAUSED | VICTORY | DEFEAT | TIMEOUT | STOPPED
    episode_reward: float = 0.0
    last_reward: float = 0.0
    started_at: Optional[float] = None
    # previous-step flags, used only to detect transitions for the event log
    prev_missile_fired: bool = False
    prev_enemy_missile_fired: bool = False


class SimulationManager:
    def __init__(self, model_service: ModelService, max_steps: int = MAX_STEPS, tick_hz: float = TICK_HZ):
        self.model_service = model_service
        self.max_steps = max_steps
        self.tick_interval = 1.0 / tick_hz
        self.env = AircraftCombatEnv()
        self._obs: Optional[np.ndarray] = None
        self._state = _RunState()
        self._subscribers: List[asyncio.Queue] = []
        self._loop_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Pub/sub for WebSocket clients
    # ------------------------------------------------------------------ #
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _broadcast(self, message: dict) -> None:
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # Drop the oldest message rather than block the sim loop.
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except Exception:
                    dead.append(q)
        for q in dead:
            self.unsubscribe(q)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict:
        return {
            "status": self._state.status,
            "running": self._state.status == "ACTIVE",
            "step": int(self.env.steps) if self._obs is not None else 0,
            "max_steps": self.max_steps,
            "episode_reward": round(self._state.episode_reward, 4),
        }

    async def reset(self) -> Dict:
        async with self._lock:
            if self._state.status == "ACTIVE":
                raise SimulationBusyError("Cannot reset while simulation is ACTIVE. Pause or stop it first.")
            self._obs, _info = self.env.reset()
            self._state = _RunState(status="IDLE")
            state = self._build_state_message(raw_action=np.zeros(4, dtype=np.float32), reward=0.0)
            await self._broadcast({"type": "event", "step": 0, "message": "Simulation reset", "level": "info"})
            await self._broadcast(state)
            return state

    async def start(self) -> Dict:
        async with self._lock:
            if self._state.status == "ACTIVE":
                raise SimulationBusyError("A simulation is already ACTIVE.")

            if not self.model_service.is_loaded:
                self.model_service.load()

            if self._obs is None or self._state.status in ("VICTORY", "DEFEAT", "TIMEOUT", "STOPPED"):
                self._obs, _info = self.env.reset()
                self._state = _RunState()

            self._state.status = "ACTIVE"
            self._state.started_at = time.time()

        await self._broadcast({"type": "event", "step": int(self.env.steps), "message": "Simulation started", "level": "success"})

        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run_loop())

        return self.snapshot()

    async def pause(self) -> Dict:
        async with self._lock:
            if self._state.status != "ACTIVE":
                raise SimulationNotRunningError("Simulation is not ACTIVE, nothing to pause.")
            self._state.status = "PAUSED"
        await self._broadcast({"type": "event", "step": int(self.env.steps), "message": "Simulation paused", "level": "warning"})
        return self.snapshot()

    async def resume(self) -> Dict:
        async with self._lock:
            if self._state.status != "PAUSED":
                raise SimulationNotRunningError("Simulation is not PAUSED, nothing to resume.")
            self._state.status = "ACTIVE"
        await self._broadcast({"type": "event", "step": int(self.env.steps), "message": "Simulation resumed", "level": "info"})
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run_loop())
        return self.snapshot()

    async def stop(self) -> Dict:
        async with self._lock:
            self._state.status = "STOPPED"
        await self._broadcast({"type": "event", "step": int(self.env.steps) if self._obs is not None else 0, "message": "Simulation stopped", "level": "warning"})
        return self.snapshot()

    # ------------------------------------------------------------------ #
    # Stepping
    # ------------------------------------------------------------------ #
    def _build_state_message(self, *, raw_action: np.ndarray, reward: float, terminated: bool = False, truncated: bool = False) -> Dict:
        return serialize_state(
            self.env,
            raw_action=raw_action,
            reward=reward,
            episode_reward=self._state.episode_reward,
            status=self._state.status,
            terminated=terminated,
            truncated=truncated,
            max_steps=self.max_steps,
        )

    async def _emit_transition_events(self) -> None:
        env = self.env
        step = int(env.steps)

        if env.missile_fired and not self._state.prev_missile_fired:
            await self._broadcast({"type": "event", "step": step, "message": "Agent missile launched", "level": "info"})
        if env.enemy_missile_fired and not self._state.prev_enemy_missile_fired:
            await self._broadcast({"type": "event", "step": step, "message": "Enemy missile launched", "level": "info"})
        if env.missile_just_expired:
            await self._broadcast({"type": "event", "step": step, "message": "Enemy missile expired \u2014 dodge!", "level": "success"})
        if env.enemy_hit:
            await self._broadcast({"type": "event", "step": step, "message": "\U0001F3AF Enemy hit! Direct missile impact.", "level": "success"})
        if env.agent_hit:
            await self._broadcast({"type": "event", "step": step, "message": "\U0001F4A5 Agent hit! Missile impact sustained.", "level": "critical"})

        self._state.prev_missile_fired = bool(env.missile_fired)
        self._state.prev_enemy_missile_fired = bool(env.enemy_missile_fired)

    async def _run_loop(self) -> None:
        """Background task: ticks the env forward with real PPO actions while ACTIVE."""
        try:
            while True:
                async with self._lock:
                    status = self._state.status
                if status != "ACTIVE":
                    if status in ("STOPPED", "VICTORY", "DEFEAT", "TIMEOUT"):
                        return
                    await asyncio.sleep(self.tick_interval)
                    continue

                tick_start = time.time()
                await self._step_once()

                elapsed = time.time() - tick_start
                await asyncio.sleep(max(0.0, self.tick_interval - elapsed))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Simulation loop crashed")
            async with self._lock:
                self._state.status = "STOPPED"
            await self._broadcast({"type": "error", "message": "Simulation loop encountered an internal error and was stopped."})

    async def _step_once(self) -> Dict:
        """Advance the environment exactly one step using a REAL PPO action."""
        if self._obs is None:
            self._obs, _info = self.env.reset()

        try:
            action = self.model_service.predict(self._obs, deterministic=True)
        except ModelLoadError as exc:
            async with self._lock:
                self._state.status = "STOPPED"
            await self._broadcast({"type": "error", "message": f"Model inference failed: {exc}"})
            raise

        obs, reward, terminated, truncated, _info = self.env.step(np.asarray(action, dtype=np.float32))
        self._obs = obs
        self._state.last_reward = float(reward)
        self._state.episode_reward += float(reward)

        await self._emit_transition_events()

        async with self._lock:
            if terminated and self.env.enemy_hit:
                self._state.status = "VICTORY"
            elif terminated and self.env.agent_hit:
                self._state.status = "DEFEAT"
            elif truncated:
                self._state.status = "TIMEOUT"

        state_message = self._build_state_message(
            raw_action=np.asarray(action, dtype=np.float32),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
        await self._broadcast(state_message)

        if terminated or truncated:
            outcome = self._state.status
            await self._broadcast({
                "type": "event",
                "step": int(self.env.steps),
                "message": f"Simulation ended \u2014 {outcome}",
                "level": "success" if outcome == "VICTORY" else "critical" if outcome == "DEFEAT" else "warning",
            })

        return state_message
