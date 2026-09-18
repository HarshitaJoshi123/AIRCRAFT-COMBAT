from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.simulation import model_service, router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("aircraft_combat.main")

app = FastAPI(
    title="AI Aircraft Combat Simulator API",
    description="Serves real-time PPO-controlled dogfight simulation state over REST + WebSocket.",
    version="1.0.0",
)

allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def on_startup():
    try:
        model_service.load()
        logger.info("PPO model loaded successfully at startup.")
    except Exception as exc:
        # Don't crash the whole API if the model is missing - /api/health will
        # report it and /api/simulation/start will fail with a clear 503
        # instead of the whole server refusing to boot.
        logger.error("PPO model failed to load at startup: %s", exc)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/")
async def root():
    return {"name": "AI Aircraft Combat Simulator API", "status": "running", "docs": "/docs"}
