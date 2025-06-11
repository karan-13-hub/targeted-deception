import random

"""Random Agent."""
class RandomAgent:
  """Agent that takes random legal actions."""

  def __init__(self, config):
    """Initialize the agent."""
    self.config = config

  def get_action(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] == 0:
      return random.choice(observation['legal_moves'])
    else:
      return None
