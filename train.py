from livermore_gym.envs import LivermoreEnv
from stable_baselines3 import PPO
import os

logdir = "logs"
modeldir = "models/PPO"

os.makedirs(modeldir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

TIMESTEPS = 1000000
directory = "/path/to/dir"


def learn(model, name):
  for i in range(1, 10):
    model.learn(
      total_timesteps=TIMESTEPS,
      reset_num_timesteps=False,
      tb_log_name=name,
      progress_bar=True,
    )
    model.save(f"models/{name}")


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
#   This is needed for the loop
#   or if you want to run one stock value
# ---------------------------------------------------------

# print("Initial training")
# initial_env = LivermoreEnv(f"{directory}/NVDA.csv", 30)
# learn(PPO("MlpPolicy", initial_env, tensorboard_log=logdir), "30")

# ---------------------------------------------------------
#   if you want to load a model and run it
# ---------------------------------------------------------

# print("Loaded training")
# env = LivermoreEnv(f"{directory}/NVDA.csv", 30)
# learn(PPO.load(f"{modeldir}.zip", env, tensorboard_log=logdir))

# ---------------------------------------------------------
#   If you want to run the model for each file in a dir
# ---------------------------------------------------------

# index = 0
# for filename in os.listdir(directory):
#  if filename.endswith(".csv"):
#    print(f"{index}/{len(os.listdir(directory))}: {filename}")
#    env = LivermoreEnv(f"{directory}/{filename}", 30)
#
#    learn(ppo.load(f"{modeldir}.zip", env, tensorboard_log=logdir))
#    index += 1


# ---------------------------------------------------------
#   If you want to run a loaded value
# ---------------------------------------------------------
# load_env = LivermoreEnv(f"{directory}/NVDA.csv", 30)
# load(PPO.load(f"{modeldir}.zip", load_env))
