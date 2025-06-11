import numpy as np
# import gym
# import gym_kuhn_poker
from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv

if __name__ == "__main__":
    env = KuhnPokerEnv(number_of_players=3, deck_size=5, betting_rounds=2, ante=1) # optional secondary argument

    game_num = 0
    while game_num < 5:
        obs = env.reset()
        done = False
        print(f"Game Number : {game_num+1}")
        while not done:
            action = np.random.choice(env.action_space.n)
            print(f"\nCurrent Player: {env.current_player}")
            obs, reward, done, info = env.step(action)
            print(f" Action: {action}, Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
        print(f"The winner is {env.winner}")
        game_num += 1
        print("=========================================")
        print("\n\n")