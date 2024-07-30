from livermore_gym.envs import LivermoreEnv
from stable_baselines3 import PPO
import os

models_ppo = "models/PPO"
logdir = "logs"

os.makedirs(models_ppo, exist_ok=True)
os.makedirs(logdir, exist_ok=True)


env = LivermoreEnv("../DATA/processed/NVDA.csv")

TIMESTEPS = 100000


def learn(model, name):
  for i in range(1, 10):
    model.learn(
      total_timesteps=TIMESTEPS,
      reset_num_timesteps=False,
      tb_log_name=name,
      progress_bar=True,
    )
    model.save(f"models/{name}/{TIMESTEPS * i}")


def load(model):
  for i in range(5):
    vec_env = model.get_env()
    obs = vec_env.reset()
    done = False
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, dones, something, info = vec_env.step(action)
      vec_env.render()


learn(PPO("MlpPolicy", env, tensorboard_log=logdir), "PPO")
# load(PPO.load("models/PPO/90000", env=env))
