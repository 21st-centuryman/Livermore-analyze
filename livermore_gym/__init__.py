__version__ = "0.1.0"
from gymnasium.envs.registration import register

register(
  id="Livermore-gym-v0",
  entry_point="livermore_gym.envs:LivermoreEnv",
)
