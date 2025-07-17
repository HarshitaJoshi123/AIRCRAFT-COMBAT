from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from combat_env import AircraftCombatEnv

def train_agent(total_timesteps=500_000, model_path="ppo_aircraft_model"):
    # Wrap environment with DummyVecEnv
    env = DummyVecEnv([lambda: AircraftCombatEnv()])

    # Initialize PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs")

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_agent()
