import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import HTML

def draw_scene(env, ax, agent_pos, enemy_pos, missile_pos=None, enemy_missile_pos=None,
               agent_hit=False, enemy_hit=False):
    ax.clear()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_zlim([0, 100])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Aircraft Combat 3D")

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
        ax.scatter(agent_pos[0], agent_pos[1], agent_pos[2], color='purple', s=300, marker='*', label='💥 Agent Hit!')
    if enemy_hit:
        ax.scatter(enemy_pos[0], enemy_pos[1], enemy_pos[2], color='yellow', s=300, marker='*', label='🎯 Enemy Hit!')

    ax.legend()

def animate_agent(env, model, steps=50):
    obs, _ = env.reset()
    frames = []

    for _ in range(steps):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        draw_scene(env, ax, env.agent_pos, env.enemy_pos,
                   env.missile_pos, env.enemy_missile_pos,
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

def display_animation(frames, interval=150):
    fig = plt.figure()
    im = plt.imshow(frames[0])
    
    def update(i):
        im.set_data(frames[i])
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())
