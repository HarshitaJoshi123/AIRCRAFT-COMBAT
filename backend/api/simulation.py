from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from backend.model.model_service import ModelLoadError, ModelService
from backend.schemas.simulation import EvaluationRequest, HealthResponse, SimulationStatusResponse
from backend.simulation.evaluation_service import EvaluationBusyError, EvaluationService
from backend.simulation.simulation_manager import (
    SimulationBusyError,
    SimulationManager,
    SimulationNotRunningError,
)

logger = logging.getLogger("aircraft_combat.api")

router = APIRouter()

model_service = ModelService.get_instance()
simulation_manager = SimulationManager(model_service)
evaluation_service = EvaluationService(model_service)


# ---------------------------------------------------------------------- #
# Health
# ---------------------------------------------------------------------- #
@router.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_service.is_loaded or True else "degraded",
        model_loaded=model_service.is_loaded,
        model_path=model_service.model_path,
        active_simulation=simulation_manager.snapshot()["running"],
    )


# ---------------------------------------------------------------------- #
# REST simulation controls
# ---------------------------------------------------------------------- #
@router.post("/api/simulation/start", response_model=SimulationStatusResponse)
async def start_simulation():
    try:
        snap = await simulation_manager.start()
    except SimulationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return snap


@router.post("/api/simulation/pause", response_model=SimulationStatusResponse)
async def pause_simulation():
    try:
        return await simulation_manager.pause()
    except SimulationNotRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/api/simulation/resume", response_model=SimulationStatusResponse)
async def resume_simulation():
    try:
        return await simulation_manager.resume()
    except SimulationNotRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/api/simulation/reset", response_model=SimulationStatusResponse)
async def reset_simulation():
    try:
        state = await simulation_manager.reset()
    except SimulationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return simulation_manager.snapshot()


@router.post("/api/simulation/stop", response_model=SimulationStatusResponse)
async def stop_simulation():
    return await simulation_manager.stop()


@router.get("/api/simulation/status", response_model=SimulationStatusResponse)
async def simulation_status():
    return simulation_manager.snapshot()


# ---------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------- #
@router.post("/api/evaluation/start")
async def start_evaluation(req: EvaluationRequest):
    try:
        return await evaluation_service.start(req.episodes)
    except EvaluationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/api/evaluation/status")
async def evaluation_status():
    return evaluation_service.snapshot()


# ---------------------------------------------------------------------- #
# WebSockets
# ---------------------------------------------------------------------- #
@router.websocket("/ws/simulation")
async def ws_simulation(websocket: WebSocket):
    await websocket.accept()
    queue = simulation_manager.subscribe()
    try:
        # Send an immediate snapshot so a late-joining client isn't blind
        # until the next tick.
        await websocket.send_json({"type": "status", **simulation_manager.snapshot()})

        sender_task = asyncio.create_task(_pump_queue_to_socket(queue, websocket))
        try:
            while True:
                # We don't require incoming messages, but reading keeps the
                # connection's disconnect detection responsive.
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
        finally:
            sender_task.cancel()
    finally:
        simulation_manager.unsubscribe(queue)


@router.websocket("/ws/evaluation")
async def ws_evaluation(websocket: WebSocket):
    await websocket.accept()
    queue = evaluation_service.subscribe()
    try:
        await websocket.send_json({"type": "status", **evaluation_service.snapshot()})
        sender_task = asyncio.create_task(_pump_queue_to_socket(queue, websocket))
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("action") == "run":
                    episodes = int(data.get("episodes", 300))
                    try:
                        await evaluation_service.start(episodes)
                    except EvaluationBusyError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                    except ModelLoadError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            pass
        finally:
            sender_task.cancel()
    finally:
        evaluation_service.unsubscribe(queue)


async def _pump_queue_to_socket(queue: asyncio.Queue, websocket: WebSocket) -> None:
    while True:
        message = await queue.get()
        await websocket.send_json(message)
