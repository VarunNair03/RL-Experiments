import gymnasium as gym

# Customizing the CartPole environment
env = gym.make(
    "CartPole-v1",
    render_mode="human",
    max_episode_steps=500,  # Extend the episode duration # Update the reward threshold
)

state, info = env.reset()
for step in range(300):  # Limit the simulation to 300 steps
    env.render()
    
    # Simple policy: push cart in the direction of the pole's angle
    pole_angle = state[2]  # Extracting the pole angle
    action = 1 if pole_angle > 0 else 0  # Push cart to the right if pole leans right, else left
    
    state, reward, done, truncated, info = env.step(action)
    print(f"Step: {step}, State: {state}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
    
    if done or truncated:
        print("Episode finished. Resetting environment.")
        state, info = env.reset()

env.close()
