# AI Aircraft Combat Simulator

A real-time 3D dogfight visualizer powered by a PPO-trained reinforcement
learning agent. Built from an original Google Colab research notebook and
its trained model — every move you see (positions, missiles, victory/defeat)
comes from a live simulation being driven by real `model.predict()` calls.
Nothing is scripted or pre-recorded.

---

## 📓 Original Research Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MaxFtg5U0TL3NgOQkWD9eBdwdVKYMoum?usp=sharing)

This is where the environment and PPO model were originally built and trained.

---

## 🚀 Features

- Real-time 3D dogfight rendered with React Three Fiber / Three.js
- FastAPI backend running the actual simulation + PPO model
- Live WebSocket streaming of every tick (position, missiles, reward, actions)
- Start / Pause / Resume / Reset / Stop controls with 4 camera modes
- Victory / Defeat / Timeout screen with full episode stats
- Model Evaluation dashboard — run N episodes and see win rate & reward charts
- Dockerized for one-command deployment

---

## 🧠 The Model

| | |
|---|---|
| Algorithm | PPO (`MlpPolicy`), Stable-Baselines3 |
| Observation | 13 values — agent & enemy position/velocity + distance |
| Action | 4 values — throttle, pitch, yaw, fire |
| Training timesteps | 500,000 |

Reward shaping includes proximity, smart missile timing, dodging, survival
bonuses, and boundary/danger penalties, with +100 for a hit and -100 for
being hit.

---

## 🛠️ Tech Stack

**Frontend:** React, TypeScript, Three.js, React Three Fiber, Tailwind CSS
**Backend:** Python, FastAPI, Gymnasium, Stable-Baselines3, WebSockets

---

## 📁 Project Structure

```text
aircraft-combat/
├── backend/          FastAPI app, simulation logic, PPO model service
├── frontend/          React + Three.js dashboard
├── models/            Trained ppo_aircraft_model.zip
├── training/          Standalone training/eval scripts (not used in prod)
└── docker-compose.yml
```

---

## 🌟 Impact

This project shows a full end-to-end RL pipeline in action — not just a
trained model sitting in a notebook, but a live, interactive system where
you can actually watch the agent think and fight in real time. It bridges
research (Colab, PPO training) with a production-style web app (FastAPI +
React + WebSockets), making the agent's behavior transparent and explorable
rather than a black box.
