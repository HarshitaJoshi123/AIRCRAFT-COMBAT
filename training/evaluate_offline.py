"""
Offline evaluation + matplotlib animation tools, preserved from the
original Colab notebook for local/offline analysis only.

These are NOT used by the production backend (which uses
`backend/simulation/evaluation_service.py` for the live web dashboard, and
Three.js for visualization instead of matplotlib - see Phase 17 of the
project spec). This file exists purely so the notebook's original
analysis workflow (`evaluate_agent`, `animate_agent`, `display_animation`)
remains available for offline use, e.g. in a Jupyter notebook or a local
`python -i` session.

Usage:
    python training/evaluate_offline.py --episodes 300 --model models/ppo_aircraft_model
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import mean

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.simulation.combat_env import AircraftCombatEnv


def evaluate_agent(model, episodes: int = 300, verbose: bool = True):
    """Identical logic to the notebook's evaluate_agent()."""
    wins, losses, rewards = 0, 0, []
    eval_env = AircraftCombatEnv()

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        terminated = truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)

        if eval_env.enemy_hit:
            wins += 1
        elif eval_env.agent_hit:
            losses += 1
        else:
            if ep_reward > 0:
                wins += 1
            else:
                losses += 1

    win_rate = (wins / episodes) * 100
    loss_rate = (losses / episodes) * 100
    avg_reward = mean(rewards)

    if verbose:
        print(f"\nEvaluation over {episodes} episodes:")
        print(f"- Wins: {wins}")
        print(f"- Losses: {losses}")
        print(f"- Win Rate: {win_rate:.2f}%")
        print(f"- Loss Rate: {loss_rate:.2f}%")
        print(f"- Average Reward: {avg_reward:.2f}")
        print(f"- Max Reward: {max(rewards):.2f}")
        print(f"- Min Reward: {min(rewards):.2f}")

    return win_rate, avg_reward


def draw_scene(env, ax, agent_pos, enemy_pos, missile_pos=None, enemy_missile_pos=None,
               agent_hit=False, enemy_hit=False):
    ax.clear()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Aircraft Combat 3D (offline matplotlib preview)")

    ax.scatter(*agent_pos, color='blue', s=200, label='Agent')
    if hasattr(env, 'path_history') and len(env.path_history) > 1:
        path = np.array(env.path_history)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='blue', linewidth=1.5, label='Agent Path')

    ax.scatter(*enemy_pos, color='red', s=200, label='Enemy')
    if hasattr(env, 'enemy_path_history') and len(env.enemy_path_history) > 1:
        path = np.array(env.enemy_path_history)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=1.5, label='Enemy Path')

    if missile_pos is not None:
        ax.scatter(*missile_pos, color='black', s=60, marker='x', label='Agent Missile')
    if enemy_missile_pos is not None:
        ax.scatter(*enemy_missile_pos, color='orange', s=60, marker='x', label='Enemy Missile')

    if agent_hit:
        ax.scatter(*agent_pos, color='purple', s=300, marker='*', label='Agent Hit!')
    if enemy_hit:
        ax.scatter(*enemy_pos, color='yellow', s=300, marker='*', label='Enemy Hit!')

    ax.legend()


def animate_agent(env, model, steps: int = 50):
    import matplotlib.pyplot as plt

    obs, _ = env.reset()
    frames = []
    for _ in range(steps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        draw_scene(env, ax, env.agent_pos, env.enemy_pos, env.missile_pos, env.enemy_missile_pos,
                   agent_hit=env.agent_hit, enemy_hit=env.enemy_hit)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(img)
        plt.close(fig)

        if terminated or truncated:
            for _ in range(5):
                frames.append(img)
            break
    return frames


def save_animation_gif(frames, output_path: str, interval_ms: int = 150):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()
    im = plt.imshow(frames[0])

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    ani.save(output_path, writer="pillow")
    plt.close(fig)
    print(f"Saved animation to {output_path}")


def main():
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(description="Offline PPO evaluation + optional matplotlib animation export.")
    parser.add_argument("--model", type=str, default="models/ppo_aircraft_model", help="Path (without .zip) to the trained model.")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--animate-steps", type=int, default=0, help="If > 0, also render a matplotlib GIF of this many steps.")
    parser.add_argument("--animate-output", type=str, default="dogfight_preview.gif")
    args = parser.parse_args()

    model = PPO.load(args.model)

    print("Single Evaluation Run:")
    evaluate_agent(model, episodes=args.episodes, verbose=True)

    if args.animate_steps > 0:
        env = AircraftCombatEnv()
        frames = animate_agent(env, model, steps=args.animate_steps)
        save_animation_gif(frames, args.animate_output)


if __name__ == "__main__":
    main()
