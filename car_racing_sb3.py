import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = gym.make("CarRacing-v3", render_mode="rgb_array")
env = Monitor(env)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device="cpu"
)

model.learn(total_timesteps=50000)

model.save("ppo_car_racing")

env.close()
