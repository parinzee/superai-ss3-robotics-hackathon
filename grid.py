import gym
from gym_ros2.envs import gym_ros2_grid
from sb3_contrib import MaskablePPO

env = gym_ros2_grid.ROS2Env(x_goal=3.5, y_goal=-3.5, min_range=0.05)
model = MaskablePPO("MlpPolicy", env, verbose=2)

try:
    while True:
        model.learn(total_timesteps=1e2)
        model.save("grid_mask_gym_ros2")
        print("===Saved===")
except KeyboardInterrupt:
    pass

# del model

# model = PPO2.load("ppo_gym_ros2")
# obs = env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)