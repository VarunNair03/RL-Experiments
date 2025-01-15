import gymnasium as gym

env = gym.make("CartPole-v1", render_mode = "human")
state = env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample()  # Random action
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        env.reset()

env.close()