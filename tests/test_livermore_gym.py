from stable_baselines3.common.env_checker import check_env
from livermore_gym.envs import LivermoreEnv

env = LivermoreEnv("ADD STOCK", 30)
check_env(env)
