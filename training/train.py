"""
Standalone training script for the AircraftCombatEnv PPO agent.

This reproduces the training workflow from the original Colab notebook
(`AIRCRAFT_COMBAT_NEW.ipynb`) using the exact same environment, algorithm,
and hyperparameters (`PPO("MlpPolicy", ...)`, 500,000 timesteps).

This script is NEVER imported or run by the production backend. The
backend only loads an already-trained `models/ppo_aircraft_model.zip`.
Run this file only if you want to retrain the agent from scratch.

Usage:
    python training/train.py
    python training/train.py --timesteps 1000000 --output models/ppo_aircraft_model
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow running this script directly (`python training/train.py`) from the
# repo root without installing the project as a package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.simulation.combat_env import AircraftCombatEnv


def main():
    parser = argparse.ArgumentParser(description="Train the PPO AircraftCombatEnv agent.")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps (notebook default: 500000).")
    parser.add_argument("--output", type=str, default="models/ppo_aircraft_model", help="Path (without .zip) to save the trained model.")
    parser.add_argument("--tensorboard-log", type=str, default="./ppo_logs", help="TensorBoard log directory.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    args = parser.parse_args()

    # Validate the environment against the Gymnasium API before training,
    # exactly as the notebook does.
    check_env(AircraftCombatEnv())

    env = DummyVecEnv([lambda: AircraftCombatEnv()])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"\nSaved trained model to: {args.output}.zip")
    print("To use it in the live app, copy/rename it to models/ppo_aircraft_model.zip")


if __name__ == "__main__":
    main()
