# 🚁️ Aircraft Combat RL Environment

This repository provides a custom 3D aircraft combat simulation environment using Gymnasium, built for training reinforcement learning agents with PPO (Stable-Baselines3). The agent learns to engage and defeat an enemy aircraft in a 3D space.

---

## Notebooks

- [🚀 Run the Aircraft Combat RL environment on Google Colab](https://colab.research.google.com/drive/1c9JY-esaPhUODCKQOkJnPh5ScEaf8kFe?usp=sharing)

---

## 📆 Features

- ✅ Custom Gymnasium environment with relative motion and reward shaping.
- ✈️ Simplified missile pursuit logic for engaging the enemy.
- 📊 3D visualization of agent and enemy trajectories.
- 🤖 Pretrained PPO agent support.
- 🎥 HTML-based animation rendering for simulations.

---

## 🧠 Environment Overview

- **Observation Space**: 10D vector — agent pos, enemy pos, relative pos, and distance.
- **Action Space**: 2D continuous — agent movement direction deltas.
- **Goal**: Minimize distance to enemy; "win" if within a small radius.
- **Done**: Agent "hits" the target or time expires.

---

## 📁 Project Structure

```
aircraft-combat-rl/
│
├── combat_env/               # Custom environment definition
│   ├── __init__.py
│   └── aircraft_env.py
│
├── utils/                    # Visualization and utilities
│   ├── __init__.py
│   └── visualization.py
│
├── train.py                  # PPO training script
├── evaluate.py               # Evaluate trained agent
├── animate.py                # Generate 3D animated simulation
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/your-username/aircraft-combat-rl.git
cd aircraft-combat-rl
pip install -r requirements.txt
```

**requirements.txt:**

```txt
gymnasium
numpy
matplotlib
stable-baselines3
```

---

### 2. Train the Agent

```bash
python train.py
```

This will train a PPO agent on the `AircraftCombatEnv` and save the model to `ppo_aircraft_model.zip`.

---

### 3. Evaluate the Agent

```bash
python evaluate.py
```

Runs the trained agent and prints the success rate and average reward over test episodes.

---

### 4. Visualize Simulation

```bash
python animate.py
```

Generates an interactive 3D animation (`combat_animation.html`) showing the agent's and enemy's trajectories.

---

## 📈 Example Output

- 🕦 Blue path: Agent  
- 🔴 Red path: Enemy  
- ✅ Animation file: `combat_animation.html`

---
