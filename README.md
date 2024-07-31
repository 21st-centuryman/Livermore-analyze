<div align="center">

# Livermore-analyze
##### Trade stocks
[![Python](https://img.shields.io/badge/python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)]()
[![Pytorch](https://img.shields.io/badge/stable_baselines3-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![openai](https://img.shields.io/badge/gymnasium-412991.svg?style=for-the-badge&logo=openai&logoColor=white)]()
<p align="center">
  <img width="533" height="300" src="./assets/animation.gif">
  <br/>
  <i>A trained agent not going bankrupt due to inflation</i>
  <i>In order to profit I would need to train it longer</i>
</p>
</div>

## ⇁  Introduction

***WIP this still needs work. I have to adjust the reward and observations until I see meaningful results.***

Welcome to Livermore analyze. This is a gymnasium environment for trading and selling stocks.

The agent will have to perform an action for each data point. The possible actions are:

0: action; 0 Hold, 1 Sell, 2 Buy

The agent gets the following observations: 
1. Data points up to the current number
2. A list of purchases; Only able top buy one stock at a time as of now

The input data needed for this environment can be collected using [Livermore-fetch](https://github.com/21st-centuryman/Livermore-fetch) a program that both collects and process the data for this environment. Your data needs to be one-dimensional and a list of numbers, shown below:

```console
$ head -5 data/processed/NVDA.csv
TAPE
0.454
0.454
0.451
0.451
```

## ⇁  Instalation
```console
git clone https://github.com/21st-centuryman/Livermore-analyze
```

## ⇁  Usage
```python
import gymnasium as gym
from livermore_gym.envs import LivermoreEnv

fetch_file = /path/to/stock/data.csv
env = livermoreenv(fetch_file)

vec_env = model.get_env()
obs = vec_env.reset()
done = False
while not done:
    action = ... # Your agent code here
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
```
