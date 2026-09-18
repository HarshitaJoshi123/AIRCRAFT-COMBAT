# AI Aircraft Combat Simulator

A real-time 3D dogfight visualizer for a PPO-trained reinforcement learning
agent, built from an original Google Colab research notebook
(`AIRCRAFT_COMBAT_NEW.ipynb`) and its trained model (`ppo_aircraft_model.zip`).

The **actual trained PPO policy** controls the agent aircraft. Every value
you see in the UI &mdash; positions, missiles, telemetry, rewards,
victory/defeat &mdash; comes straight from a live `AircraftCombatEnv`
instance being driven by real `model.predict()` calls. Nothing is
scripted, randomized, or pre-recorded.

---

## Features

- Real-time 3D dogfight rendered with React Three Fiber / Three.js
- Backend FastAPI service that owns the authoritative simulation state
- WebSocket streaming of every simulation tick (position, velocity, missiles, reward, PPO action)
- Start / Pause / Resume / Reset / Stop controls
- Four camera modes: Follow Agent, Follow Enemy, Free Orbit, Top-Down
- Live combat event log (missile launches, hits, dodges, episode end)
- Victory / Defeat / Timeout result screen with full episode stats
- Model Information page describing the actual PPO/env configuration
- Model Evaluation dashboard: run N real episodes and see win rate, reward
  distribution, and per-episode reward chart
- Dockerized backend + frontend for one-command deployment

---

## Architecture

```
BROWSER
  |
  v
React + TypeScript  --------------------------\
  |                                             |
  v                                             v
Three.js / React Three Fiber        REST (start/pause/reset/stop, eval)
  |                                             |
  '-------------------- WebSocket -------------'
                          |
                          v
                 Python FastAPI (backend/main.py)
                          |
                          v
              SimulationManager (backend/simulation/simulation_manager.py)
                          |
                          v
              AircraftCombatEnv (backend/simulation/combat_env.py)
                          |
                          v
          ModelService -> Trained PPO Model (models/ppo_aircraft_model.zip)
                          |
                          v
              Real-time simulation state -> WebSocket -> Three.js
```

---

## Tech Stack

**Backend:** Python, FastAPI, Uvicorn, Gymnasium, Stable-Baselines3, NumPy, Pydantic, WebSockets
**Frontend:** React, TypeScript, Vite, Three.js, React Three Fiber, @react-three/drei, Tailwind CSS, Recharts
**ML:** PPO (Proximal Policy Optimization), custom `AircraftCombatEnv`

---

## ML Model

| | |
|---|---|
| Algorithm | PPO (`MlpPolicy`), Stable-Baselines3 |
| Observation space | `Box(-inf, inf, shape=(13,), float32)` |
| Action space | `Box(-1, 1, shape=(4,), float32)` |
| Training timesteps | 500,000 (script default) / 501,760 (saved checkpoint) |
| Saved with | stable-baselines3 2.9.0, gymnasium 1.3.0, torch 2.11.0, numpy 2.1.3 |

### Observation (13 values)
`[agent_pos(3), agent_vel(3), enemy_pos(3), enemy_vel(3), distance(1)]`

### Action (4 values, each in [-1, 1])
`[throttle_raw, pitch, yaw, fire_raw]`
- `throttle = (throttle_raw + 1) / 2`
- `fire = (fire_raw + 1) / 2`, fires when `> 0.5`

### Reward shaping (unchanged from the notebook)
Proximity shaping, missile-fire encouragement/discouragement by range,
speed incentive, survival bonus, dodge bonus, danger-zone penalty, idle
penalty, edge/boundary penalty, and terminal +100 / -100 for a
hit/being-hit.

`backend/simulation/combat_env.py` is a byte-for-byte port of the
notebook's `AircraftCombatEnv` class &mdash; the physics, reward terms,
and termination conditions were not modified.

---

## Project Structure

```
aircraft-combat/
├── backend/
│   ├── main.py                     FastAPI app entrypoint
│   ├── requirements.txt
│   ├── api/simulation.py           REST + WebSocket routes
│   ├── simulation/
│   │   ├── combat_env.py           AircraftCombatEnv (ported verbatim)
│   │   ├── simulation_manager.py   Lifecycle + PPO-driven tick loop
│   │   ├── evaluation_service.py   Real-episode evaluation runner
│   │   └── state_serializer.py     Env state -> JSON wire format
│   ├── model/model_service.py      Loads ppo_aircraft_model.zip once
│   ├── schemas/simulation.py       Pydantic models
│   └── tests/                      pytest suite
│
├── frontend/
│   ├── package.json, vite.config.ts, index.html
│   └── src/
│       ├── App.tsx, main.tsx
│       ├── components/             Aircraft, Missile, Explosion, Arena,
│       │                           CameraRig, CombatScene, TelemetryPanel,
│       │                           ControlsPanel, EventLog, ResultOverlay, NavBar
│       ├── pages/                  DashboardPage, ModelInfoPage, EvaluationPage
│       ├── hooks/                  useSimulationSocket, useEvaluationSocket
│       ├── services/               api.ts (REST), websocket.ts
│       ├── simulation/             interpolation.ts (visual smoothing only)
│       └── types/simulation.ts
│
├── models/
│   └── ppo_aircraft_model.zip      Your actual trained model
│
├── training/
│   ├── train.py                    Standalone training script (not used in prod)
│   └── evaluate_offline.py         Notebook's matplotlib tools, offline use only
│
├── pytest.ini
├── Dockerfile                      Backend image
├── frontend/Dockerfile             Frontend image (nginx)
├── docker-compose.yml
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.11+
- Node.js 20+
- (optional) Docker + Docker Compose

### Backend Setup
```bash
cd aircraft-combat
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Frontend Setup
```bash
cd aircraft-combat/frontend
npm install
cp .env.example .env                # adjust VITE_API_URL / VITE_WS_URL if needed
```

### Model Setup
The trained model is already included at `models/ppo_aircraft_model.zip`.
If you retrain it (see below), just overwrite that file, or point the
backend at a different location with:
```bash
export PPO_MODEL_PATH=/absolute/path/to/your_model.zip
```

---

## Running Locally

**Terminal 1 - backend:**
```bash
cd aircraft-combat
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 - frontend:**
```bash
cd aircraft-combat/frontend
npm run dev
```

Open **http://localhost:5173**. The frontend talks to the backend at
`http://localhost:8000` / `ws://localhost:8000` (from `.env`).

API docs (FastAPI auto-generated): **http://localhost:8000/docs**

### Running the simulation
1. Open the Dashboard tab.
2. Click **START**. The backend loads the PPO model (if not already
   loaded), resets `AircraftCombatEnv`, and begins ticking at ~12Hz,
   streaming real state over `/ws/simulation`.
3. Use **PAUSE/RESUME/RESET/STOP** and the camera buttons freely.
4. When an episode ends you'll see a VICTORY / DEFEAT / TIMEOUT overlay
   with real episode stats and a RUN AGAIN button.

### Running an evaluation
1. Open the **Evaluation** tab.
2. Choose an episode count (e.g. 300) and click **RUN EVALUATION**.
3. Progress streams live over `/ws/evaluation`; when finished you get
   real win rate / loss rate / average / max / min reward plus charts.
   This runs on a separate `AircraftCombatEnv` instance so it never
   interferes with a live dashboard simulation.

---

## Retraining (optional)
Training is intentionally isolated from the production backend.
```bash
cd aircraft-combat
python training/train.py --timesteps 500000 --output models/ppo_aircraft_model
```
This reproduces the notebook's exact training call
(`PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs")`,
`model.learn(total_timesteps=500000)`). The production backend never
calls this script itself and never retrains on startup.

For an offline matplotlib-based evaluation/animation preview (as in the
original notebook, not used by the live web app):
```bash
python training/evaluate_offline.py --episodes 300 --animate-steps 150 --animate-output preview.gif
```

---

## Testing
```bash
cd aircraft-combat
pytest
```
This runs, among others, the critical acceptance test
`test_model_can_run_full_episode`, which verifies the actual
`ppo_aircraft_model.zip` can drive the refactored `AircraftCombatEnv`
end-to-end with no shape/behavior mismatch.

---

## Deployment

### Docker Compose (recommended)
```bash
docker compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:80

Override URLs for a real deployment by passing build args / env vars:
```bash
VITE_API_URL=https://api.yourdomain.com \
VITE_WS_URL=wss://api.yourdomain.com \
docker compose build frontend
```

### Manual deployment
- **Backend:** any host that can run `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
  (Fly.io, Render, a VM, ECS, etc). Set `PPO_MODEL_PATH` and `CORS_ORIGINS`.
- **Frontend:** `npm run build` produces `frontend/dist/`, deployable to any
  static host (Vercel, Netlify, S3+CloudFront, nginx). Set `VITE_API_URL`
  and `VITE_WS_URL` at build time to point at your backend's public URL
  (use `wss://` in production, since browsers require secure WebSockets
  from an HTTPS page).

### Large model files
`ppo_aircraft_model.zip` here is ~150KB, well under GitHub's limits. If
you retrain a much larger model:
- Use [Git LFS](https://git-lfs.github.com/) for the `models/` directory, **or**
- Store the model in object storage (S3/GCS) and have the backend download
  it at startup into the path pointed to by `PPO_MODEL_PATH` (add a small
  startup hook in `backend/main.py`'s `on_startup` before `model_service.load()`).

---

## How the pieces connect

- **Frontend never computes physics.** `CombatScene.tsx` renders exactly
  the `agent`/`enemy`/`missile` fields from the latest WebSocket message.
  `simulation/interpolation.ts` only smooths the *visual* position between
  two real backend ticks (12Hz backend, 60fps render) &mdash; it never
  changes what actually happened in the environment.
- **PPO is the only decision-maker.** `SimulationManager._step_once()`
  calls `ModelService.predict(obs, deterministic=True)` every tick and
  feeds that action straight into `env.step()`. There is no fallback to
  random or scripted actions.
- **Evaluation reuses the same model and env class**, just in a separate
  `AircraftCombatEnv` instance and a background asyncio task, so it can
  run hundreds of episodes without blocking the live dashboard's
  WebSocket.

---

## Known Limitations

- Only one live simulation runs at a time (by design - `SimulationManager`
  rejects a second `start()` while one is `ACTIVE`/`PAUSED` with HTTP 409).
- The 3D aircraft are procedural low-poly meshes (cones/boxes), not
  textured GLTF models, per the "no external assets required" requirement.
- The evaluation dashboard blocks on CPU-bound PPO inference inside an
  `asyncio` task; for very large episode counts (5,000+) this will take
  noticeably longer since it's single-process. Progress still streams
  live so the UI won't appear frozen.
- No authentication/authorization - this is a single-user demo app.

---

## Future Improvements

- Multi-session support (one simulation per WebSocket client instead of one global simulation)
- Replay/scrubbing of past episodes
- Swap-in support for alternate trained checkpoints from the UI
- GLTF aircraft models with more detailed materials/animations
- Server-authoritative reconnect/resync (currently a rejoining client only
  gets a status snapshot, not full episode replay-to-date)
