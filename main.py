import gym
from gym_ros2.envs import gym_ros2
from stable_baselines3 import PPO

env = gym_ros2.ROS2Env()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1e6)
model.save("ppo_gym_ros2")

del model

model = PPO.load("ppo_gym_ros2")
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)