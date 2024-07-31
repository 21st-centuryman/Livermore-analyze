from livermore_gym.envs import LivermoreEnv
from stable_baselines3 import PPO
import os

logdir = "logs"

os.makedirs("models/PPO", exist_ok=True)
os.makedirs(logdir, exist_ok=True)

TIMESTEPS = 1000000


def learn(model):
  for i in range(1, 10):
    print(i)
    model.learn(
      total_timesteps=TIMESTEPS,
      reset_num_timesteps=False,
      tb_log_name="PPO",
      progress_bar=True,
    )
    model.save("models/PPO")


def load(model):
  for i in range(5):
    vec_env = model.get_env()
    obs = vec_env.reset()
    done = False
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, _ = vec_env.step(action)
      vec_env.render()


# ---------------------------------------------------------
#   This is needed for the loop or if you want to run one stock value
# ---------------------------------------------------------

print("Initial training")
directory = "../data/processed/"
initial_env = LivermoreEnv(f"{directory}/NVDA.csv", 30)
learn(PPO("MlpPolicy", initial_env, tensorboard_log=logdir))


# ---------------------------------------------------------
#   If you want to run the model for each file in a dir
# ---------------------------------------------------------

# index = 0
# for filename in os.listdir(directory):
#  if filename.endswith(".csv"):
#    print(f"{index}/{len(os.listdir(directory))}: {filename}")
#    env = LivermoreEnv(f"{directory}/{filename}", 30)
#
#    learn(PPO.load("models/PPO/livermore", env, tensorboard_log=logdir))
#    index += 1


# ---------------------------------------------------------
#   If you want to run a loaded value
# ---------------------------------------------------------
# load_path = "./models/PPO/"
# load_env = LivermoreEnv(f"{directory}/NVDA.csv", 30)
# load(PPO.load(load_path, load_env))
