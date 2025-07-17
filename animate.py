from stable_baselines3 import PPO
from combat_env import AircraftCombatEnv
from utils.visualization import animate_agent, display_animation

def run_animation(model_path="ppo_aircraft_model", steps=150):
    # Load trained model
    model = PPO.load(model_path)

    # Create environment
    env = AircraftCombatEnv()

    # Animate the agent in the environment
    frames = animate_agent(env, model, steps=steps)

    # Display animation in notebook
    return display_animation(frames)

if __name__ == "__main__":
    # If running in a notebook, just call run_animation() in a cell
    animation_html = run_animation()
    with open("combat_animation.html", "w") as f:
        f.write(animation_html.data)
    print("🎞️ Animation saved as combat_animation.html")
