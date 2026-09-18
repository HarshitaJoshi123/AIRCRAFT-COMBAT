from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

SimulationStatus = Literal["IDLE", "ACTIVE", "PAUSED", "VICTORY", "DEFEAT", "TIMEOUT", "STOPPED"]


class Vector3(BaseModel):
    x: float
    y: float
    z: float

    @classmethod
    def from_list(cls, v) -> "Vector3":
        return cls(x=float(v[0]), y=float(v[1]), z=float(v[2]))


class EntityState(BaseModel):
    position: List[float] = Field(..., min_length=3, max_length=3)
    velocity: List[float] = Field(..., min_length=3, max_length=3)
    direction: List[float] = Field(..., min_length=3, max_length=3)


class MissileState(BaseModel):
    active: bool
    position: Optional[List[float]] = None


class ActionState(BaseModel):
    throttle: float
    pitch: float
    yaw: float
    fire: bool


class SimulationStateMessage(BaseModel):
    """Matches the WebSocket wire format described in the project spec."""

    type: Literal["state"] = "state"
    step: int
    max_steps: int
    status: SimulationStatus
    reward: float
    episode_reward: float
    distance: float

    agent: EntityState
    enemy: EntityState
    missile: MissileState
    enemy_missile: MissileState
    action: ActionState

    agent_hit: bool
    enemy_hit: bool
    terminated: bool
    truncated: bool


class SimulationEventMessage(BaseModel):
    type: Literal["event"] = "event"
    step: int
    message: str
    level: Literal["info", "warning", "critical", "success"] = "info"


class SimulationErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str


class SimulationCommandRequest(BaseModel):
    """Body for REST simulation control endpoints (all fields optional)."""

    pass


class SimulationStatusResponse(BaseModel):
    status: SimulationStatus
    running: bool
    step: int
    max_steps: int
    episode_reward: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_path: str
    active_simulation: bool


class EvaluationRequest(BaseModel):
    episodes: int = Field(default=300, ge=1, le=5000)


class EvaluationProgress(BaseModel):
    type: Literal["progress"] = "progress"
    completed: int
    total: int
    wins: int
    losses: int


class EvaluationResult(BaseModel):
    type: Literal["result"] = "result"
    episodes: int
    wins: int
    losses: int
    win_rate: float
    loss_rate: float
    avg_reward: float
    max_reward: float
    min_reward: float
    episode_rewards: List[float]
    episode_outcomes: List[Literal["win", "loss"]]
