"""
EvaluationService reproduces the notebook's `evaluate_agent()` logic
exactly (same win/loss classification, same metrics) but runs it as a
cancellable background asyncio task that streams progress, instead of a
single blocking call. It always uses a brand-new `AircraftCombatEnv` and
the same loaded PPO model used by live simulations - never fabricated
numbers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from statistics import mean
from typing import List, Literal, Optional

import numpy as np

from backend.model.model_service import ModelService
from backend.simulation.combat_env import AircraftCombatEnv

logger = logging.getLogger("aircraft_combat.evaluation_service")


class EvaluationBusyError(RuntimeError):
    pass


@dataclass
class EvaluationState:
    running: bool = False
    completed: int = 0
    total: int = 0
    wins: int = 0
    losses: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    episode_outcomes: List[str] = field(default_factory=list)
    done: bool = False
    error: Optional[str] = None


class EvaluationService:
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self._state = EvaluationState()
        self._subscribers: List[asyncio.Queue] = []
        self._task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._state.running

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    async def _broadcast(self, message: dict) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                pass

    def snapshot(self) -> dict:
        return {
            "running": self._state.running,
            "completed": self._state.completed,
            "total": self._state.total,
            "wins": self._state.wins,
            "losses": self._state.losses,
            "done": self._state.done,
            "error": self._state.error,
        }

    async def start(self, episodes: int) -> dict:
        if self._state.running:
            raise EvaluationBusyError("An evaluation is already running.")

        if not self.model_service.is_loaded:
            self.model_service.load()

        self._state = EvaluationState(running=True, total=episodes)
        self._task = asyncio.create_task(self._run(episodes))
        return self.snapshot()

    async def _run(self, episodes: int) -> None:
        eval_env = AircraftCombatEnv()
        try:
            for ep in range(episodes):
                obs, _info = eval_env.reset()
                terminated = truncated = False
                ep_reward = 0.0

                while not (terminated or truncated):
                    action = self.model_service.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _info = eval_env.step(np.asarray(action, dtype=np.float32))
                    ep_reward += float(reward)
                    # Yield control so the event loop can service WebSocket I/O.
                    await asyncio.sleep(0)

                self._state.episode_rewards.append(ep_reward)

                # Same win/loss classification as the notebook's evaluate_agent().
                if eval_env.enemy_hit:
                    outcome = "win"
                    self._state.wins += 1
                elif eval_env.agent_hit:
                    outcome = "loss"
                    self._state.losses += 1
                else:
                    if ep_reward > 0:
                        outcome = "win"
                        self._state.wins += 1
                    else:
                        outcome = "loss"
                        self._state.losses += 1

                self._state.episode_outcomes.append(outcome)
                self._state.completed = ep + 1

                await self._broadcast({
                    "type": "progress",
                    "completed": self._state.completed,
                    "total": episodes,
                    "wins": self._state.wins,
                    "losses": self._state.losses,
                })

            rewards = self._state.episode_rewards
            win_rate = (self._state.wins / episodes) * 100
            loss_rate = (self._state.losses / episodes) * 100

            result = {
                "type": "result",
                "episodes": episodes,
                "wins": self._state.wins,
                "losses": self._state.losses,
                "win_rate": round(win_rate, 2),
                "loss_rate": round(loss_rate, 2),
                "avg_reward": round(mean(rewards), 4) if rewards else 0.0,
                "max_reward": round(max(rewards), 4) if rewards else 0.0,
                "min_reward": round(min(rewards), 4) if rewards else 0.0,
                "episode_rewards": [round(r, 4) for r in rewards],
                "episode_outcomes": self._state.episode_outcomes,
            }
            self._state.done = True
            await self._broadcast(result)
        except Exception as exc:  # pragma: no cover
            logger.exception("Evaluation run failed")
            self._state.error = str(exc)
            await self._broadcast({"type": "error", "message": f"Evaluation failed: {exc}"})
        finally:
            self._state.running = False
