# kuhn-poker-gym

![Kuhn's poker tree](https://raw.githubusercontent.com/Danielhp95/gym-kuhn-poker/master/Kuhn_poker_tree.png)

I have implemented the famous Kuhn Poker game in accordance to the [OpenAI gym interface](https://www.gymlibrary.dev/). The industry standard API for reinforcement learning.

## What is Kuhn Poker?

Kuhn Poker is a simplified version of poker that focuses on exploring the strategic aspects of decision-making in a limited information setting. In this game, two players participate in a single hand using a standard deck of cards without the ranking of suits. The game revolves around three cards: K (King), Q (Queen), and J (Jack). One card is dealt to each player, which may place bets similarly to a standard poker. If both players bet or both players pass, the player with the higher card wins, otherwise, the betting player wins. The intriguing aspect of Kuhn Poker is that players have limited information, and they can either bet or check. Betting increases the pot, while checking allows the hand to progress without adding any more chips to the pot. The game concludes when a player either bets or checks after the initial round of betting, which triggers a showdown. Kuhn Poker provides an interesting platform for studying decision-making and the interplay of strategy, deception, and risk assessment in a simplified poker setting.

[Wikipedia Explanation of Kuhn's poker](https://www.wikiwand.com/en/Kuhn_poker)

## Action space

Two actions, `[Pass, Bet]` make up the action space.

## State space

The state space is the concatention of several vectors:

+ `Current player`. 
+ `Player hand`.
+ `Betting history`. History of whether **each** player `PASS`ed or `BET`ted.
+ `Pot contributions`. A vector `p = [p_1, p_2]`, where `p_1` corresponds to the contribution
of player 1 to the pot. It includes player's `antes`.

## Observation space
Each player has a subset of the state space that representes their oberservation space. 

The observation for player `i` contains:
+ `Player id`
+ `Dealt card`
+ `Betting history`
+ `Pot contributions`

## Reward

Each transition to each state is associated with a reward vector `r = [r_1, r_2]`, where each index i is the reward associated to player i. The reward function works the same as standard poker.

## How to play


### Installing via cloning this repository

```bash
git clone https://github.com/abhivetukuri/kuhn_poker_gym.git
cd kuhn_poker_gym
pip install -e .
```

To play this game `gym` must be installed. And Kuhn's poker environment can be created via running inside a `python` interpreter:

```python
>>> import gym
>>> import gym_kuhn_poker
>>> env = gym.make('KuhnPoker-v0', **dict()) # optional secondary argument
```

The `dict()` in the expression above includes keyword arguments for the underlying environment:
+ `number_of_players`: (Default 2).
+ `deck_size`: (Default 3).
+ `betting_rounds`: (Default: 2).
+ `ante`: (Default 1).

Have fun playing this simple game with your friends!
