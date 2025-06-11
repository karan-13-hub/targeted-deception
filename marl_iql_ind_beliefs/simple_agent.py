import numpy as np


class SimpleAgent:
  """Agent that applies a simple heuristic."""

  def __init__(self, config):
    """Initialize the agent."""
    self.config = config
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.information_tokens
  @staticmethod
  def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']]

  def get_action(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] != 0:
      return None

    # Check if there are any pending hints and play the card corresponding to
    # the hint.
    for card_index, hint in enumerate(observation['card_knowledge'][0]):
      if hint['color'] is not None or hint['rank'] is not None:
        return {'action_type': 'PLAY', 'card_index': card_index}

    # Check if it's possible to hint a card to your colleagues.
    fireworks = observation['fireworks']       
    if observation['information_tokens'] > 0:
      # Check if there are any playable cards in the hands of the opponents.
      # print("\n\nNum player : %d \n\n"%observation['num_players'])
      for player_offset in range(1, observation['num_players']):
        player_hand = observation['observed_hands'][player_offset]
        # print("\n\nPlayer ID : %d"%player_offset)
        # print("Hand ID : %d, %s"%(player_offset, player_hand))
        player_hints = observation['card_knowledge'][player_offset]
        # print("Hints ID : %d, %s" %(player_offset, player_hints))
        
        # Check if the card in the hand of the opponent is playable.
        for card, hint in zip(player_hand, player_hints):
          c_r = np.random.randint(2)
          if c_r == 0 and SimpleAgent.playable_card(card,
                                        fireworks) and hint['color'] is None:
              return {
                  'action_type': 'REVEAL_COLOR',
                  'color': card['color'],
                  'target_offset': player_offset
              }
          
          if c_r == 1 and SimpleAgent.playable_card(card,
                                       fireworks) and hint['rank'] is None:
            return {
                'action_type': 'REVEAL_RANK',
                'rank': card['rank'],
                'target_offset': player_offset
            }

    # If no card is hintable then discard or play.
    card_index = np.random.randint(5)
    if observation['information_tokens'] < self.max_information_tokens:
      return {'action_type': 'DISCARD', 'card_index': card_index}
    else:
      return {'action_type': 'PLAY', 'card_index': card_index}



    # action_types = ['PLAY', 'DISCARD', 'REVEAL_COLOR', 'REVEAL_RANK']
    # colors = ['R', 'Y', 'G', 'W', 'B']
    # ranks = [1, 2, 3, 4, 5]
    # player_offset = np.random.randint(1, observation['num_players'])
    # action_type = np.random.choice(action_types, 1).item()
    # if 'REVEAL_COLOR' in action_type :
    #   return {
    #     'action_type': action_type,
    #     'color': np.random.choice(colors, 1).item(),
    #     'target_offset': player_offset
    #     }
    # elif 'REVEAL_RANK' in action_type:
    #   return {
    #     'action_type': action_type,
    #     'rank': np.random.choice(ranks, 1).item(),
    #     'target_offset': player_offset
    #     }
    # else :
    #   return {'action_type': 'PLAY', 'card_index': card_index}
 
