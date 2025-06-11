import torch
import numpy as np
import random

def save(args, save_name, model, ep=None):
    import os
    save_dir = './trained_models/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + "_" + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")


def collect_episode(env, agent, belief_networks, dataset, config, policy_flag):
    state = env.reset()
    player_hands = []
    for i in range(config.num_players):
        player_hands.append(state[i][config.num_players:config.num_players+config.deck_size])
    steps = 0
    done = False
    ep_rew = 0
    while not done:
        curr_plyr = env.current_player
        obs = state[curr_plyr]
        obs_arr = np.array(obs, dtype=np.float32)
        belief = belief_networks[curr_plyr].get_belief(obs_arr).detach().cpu().numpy()
        belief_arr = np.array(belief, dtype=np.float32).reshape(-1)
        if policy_flag :
            act = agent.get_action(obs_arr, belief_arr, config.eps, eval=False)
        else :
            available_actions = np.arange(env.action_space_size, dtype=np.int32)
            act = random.choice(available_actions)
        
        next_state, rew, done, _ = env.step(act)
        
        nxt_plyr = env.current_player
        nobs = next_state[nxt_plyr]        
        ep_rew += max(rew)

        dataset.add(state, obs, act, rew[curr_plyr], nobs, done, player_hands)
        state = next_state.copy()
        steps += 1
    obs = np.zeros_like(obs_arr)
    nobs = np.zeros_like(obs_arr)
    dataset.add(state, obs, 0, 0, nobs, done, player_hands)
    return ep_rew, steps    


def collect_train_episodes(env, agent, belief_networks, dataset, config, num_episodes, policy_flag = False):
    collected_steps = 0
    collected_episodes = 0
    print("Collecting initial warmup data ...")
    while collected_episodes < num_episodes :
        train_ep_rew, steps = collect_episode(env, agent, belief_networks, dataset, config, policy_flag)
        collected_steps+=steps
        collected_episodes+=1
    print("Number of steps collected so far.... %d"%collected_steps)
    print("Number of episodes collected so far.... %d"%collected_episodes)
    return train_ep_rew, collected_steps, collected_episodes


def eval_policy(env, agent, belief_networks, config, num_eval_episodes):
    print("Evaluating the policy ...")
    num_episodes = 0
    num_samples = 0
    mean_episode_reward = 0
    best_episode_reward = -np.inf
    win_rates = [0]*config.num_players
    while num_episodes < num_eval_episodes:
        state = env.reset()
        done = False
        ep_rew = 0
        episode_steps = 0
        while not done:
            curr_plyr = env.current_player
            obs = state[curr_plyr]
            obs_arr = np.array(obs, dtype=np.float32)
            belief = belief_networks[curr_plyr].get_belief(obs_arr).detach().cpu().numpy()
            belief_arr = np.array(belief, dtype=np.float32).reshape(-1)
            act = agent.get_action(obs_arr, belief_arr, eval=True)
    
            next_state, rew, done, _ = env.step(act)
            
            state = next_state.copy()

            ep_rew += max(rew)

            episode_steps += 1

        win_rates[env.winner] += 1
        mean_episode_reward += ep_rew
        if ep_rew > best_episode_reward:
            best_episode_reward = ep_rew
        num_episodes += 1
        num_samples += episode_steps
    mean_episode_reward /= num_episodes
    return mean_episode_reward, best_episode_reward, num_samples, win_rates
