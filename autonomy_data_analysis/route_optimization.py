import gym
import numpy as np
from stable_baselines3 import PPO

# Create a simple gym environment for vehicle routing
class VehicleEnv(gym.Env):
    def __init__(self):
        super(VehicleEnv, self).__init__()
        self.state = np.random.rand(2)  # Simulated (x, y) position
        self.goal = np.array([1.0, 1.0])  # Destination point

    def step(self, action):
        self.state += action * 0.1  # Move in small steps
        reward = -np.linalg.norm(self.goal - self.state)  # Negative distance to goal
        done = np.linalg.norm(self.goal - self.state) < 0.1  # Reached goal?
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.random.rand(2)
        return self.state

# Train RL model
env = VehicleEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)

# Test trained model
obs = env.reset()
for _ in range(20):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break
