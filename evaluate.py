from stable_baselines3 import PPO
from combat_env import AircraftCombatEnv
from statistics import mean

def evaluate_agent(model_path="ppo_aircraft_model", episodes=100):
    model = PPO.load(model_path)
    eval_env = AircraftCombatEnv()

    wins = 0
    losses = 0
    rewards = []

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated

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

    print(f"\n🎯 Evaluation over {episodes} episodes:")
    print(f"- Wins: {wins}")
    print(f"- Losses: {losses}")
    print(f"- Avg reward: {mean(rewards):.2f}")
    print(f"- Max reward: {max(rewards):.2f}")
    print(f"- Min reward: {min(rewards):.2f}")

if __name__ == "__main__":
    evaluate_agent()
