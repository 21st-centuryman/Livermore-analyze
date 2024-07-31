import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import pygame


class LivermoreEnv(gym.Env):
  """
  The goal of this env is to get a profit given market data.

  ### Action Space
  0: actions 0 Hold, 1 Sell, 2 Buy
  """

  def __init__(self, stock, size):
    super(LivermoreEnv, self).__init__()
    self.size = size
    self.data = (
      pl.read_csv(f"{stock}")["TAPE"].cast(pl.Float32).to_numpy().flatten().tolist()
    )  # make it a variable we call
    self.observation_space = spaces.Box(low=0, high=max(self.data), shape=((len(self.data) + 1),), dtype=np.float32)
    self.action_space = spaces.Discrete(n=3)

  def step(self, action):
    prev_bank = self.bank
    amount, transaction = self.do_action(self.length, self.data, action, self.purchases, self.bank)
    self.bank += amount
    self.bank -= self.bank * self.inflation  # inflation removes value not increase it.
    self.purchases = transaction
    self.reward = prev_bank - self.bank  # We need to work on this reward function
    # TapeRender(self.data, self.bank, self.purchases, action, self.length)  # Needs to be added in the render function
    self.length += 1
    info = {}
    if self.length == len(self.data):
      self.terminated = True
    if self.bank < 0:
      self.bank = 0
      self.reward -= 5000  # Don't spend all your money at once
    observation = np.append(np.array(self.data[self.length - self.size : self.length + 1]), self.purchases).astype(
      np.float32
    )
    return observation, self.reward, self.terminated, False, info

  def reset(self, seed=None, options=None):
    if seed is not None:
      np.random.seed(seed)
    self.purchases = np.zeros(len(self.data) - self.size)
    self.length = self.size
    self.terminated = False
    self.bank = 5000  # make it a variable we call
    self.inflation = 0.1 / 365  # inflation is static and high
    observation = np.append(np.array(self.data[0 : self.length + 1]), self.purchases).astype(np.float32)
    self.reward = 0
    return observation, {}

  def do_action(self, frame, data, act, purchases, bank):
    if act == 1:
      return (data[frame] - purchases[0]), np.insert(purchases[1:], len(purchases) - 1, 0)
    elif act == 2 and bank > data[frame]:
      return -data[frame], np.insert(purchases[:-1], 0, data[frame])
    else:
      return 0, purchases


class TapeRender:
  def __init__(self, data, bank, purchases, act, frame):
    self.render(bank, purchases, frame, data, act)

  def draw_line_graph(self, heights, widths, frame, data, screen):
    GREEN = (5, 132, 31)
    RED = (190, 30, 56)
    for i in range(min(frame, len(data) - 1)):
      color = RED if data[i + 1] - data[i] < 0 else GREEN
      pygame.draw.line(
        screen,
        color,
        (widths[i], heights[i]),
        (widths[i + 1], heights[i + 1]),
        2,
      )

  def draw_frame_number(self, bank, frame, action, purchases, data, screen):
    WHITE = (255, 255, 255)
    font = pygame.font.SysFont("Arial", 24)
    screen.blit(font.render(f"BANK: ${bank}", True, WHITE), (10, 10))
    screen.blit(font.render(f"Stock quote: ${data[frame]}", True, WHITE), (10, 34))
    screen.blit(font.render(f"Action: {action}", True, WHITE), (10, 58))
    screen.blit(
      font.render(
        f"Owned_stocks: {purchases[0:6]} ...",
        True,
        WHITE,
      ),
      (10, 82),
    )
    screen.blit(font.render("Last 30 quotes:", True, WHITE), (10, 106))
    current = frame
    for i in range(3):
      screen.blit(
        font.render(f"{data[current - 10 : current]}", True, WHITE),
        (10, 130 + i * 24),
      )
      current = current - 10

  def render(self, bank, purchases, frame, data, act):
    # Size of screen and colors
    WIDTH, HEIGHT = 1066, 600
    BLACK = (0, 0, 0)

    # This is for easier calculations in draw_line_graph
    scale_y = (HEIGHT / max(data)) * 0.65
    scale_x = (WIDTH / len(data)) * 0.98
    heights = [HEIGHT - data[i] * scale_y - 10 for i in range(len(data))]
    widths = [i * scale_x + 10 for i in range(len(data))]

    # For Pygame to work
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # png = 0  # for creating the pngs
    screen.fill(BLACK)
    self.draw_line_graph(heights, widths, frame, data, screen)
    self.draw_frame_number(bank, frame, act, purchases, data, screen)
    pygame.display.flip()

    # For gif creation, will be removed
    # if frame % 20 == 0:
    #  pygame.image.save(screen, f"frames/frame_{frame:04d}.png")

    pygame.display.flip()
