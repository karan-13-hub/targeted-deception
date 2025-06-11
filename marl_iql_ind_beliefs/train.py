import os
import time
import numpy as np
from collections import deque
from collections import OrderedDict
import torch
# import wandb
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_train_episodes, eval_policy
import random
from dqn_agent_belief_network import DQNAgentBelief
from belief_network import AgentBeliefNetwork
from random_agent import RandomAgent
from simple_agent import SimpleAgent
from tensorboardX import SummaryWriter
from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv
import pickle
import signal
import shutil

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="bo3-5p", help="Run name, default: QL-DQRN")
    
    parser.add_argument('--num_players', type=int, default=5)
    parser.add_argument('--deck_size', type=int, default=11)
    parser.add_argument('--betting_rounds', type=int, default=5)
    parser.add_argument('--ante', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--env", type=str, default="Kuhn-poker", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--epochs", type=int, default=50000, help="Number of epochs, default: 200")
    parser.add_argument("--logdir", type=str, default="./data/", help="Folder for logging data")
    parser.add_argument("--buffer_size", type=int, default=131072, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training, default: cuda:0")
    
    parser.add_argument("--eps", type=float, default=1.00, help="Epsilon, default: 1")
    parser.add_argument("--min_eps", type=float, default=1.5e-5, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=float, default=2e3, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e5")
    
    parser.add_argument("--seed", type=int, default=7, help="Seed, default: 1")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate, default: 1e-3")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.9999, help="Discount factor, default: 0.99")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor, default: 0.99")
    parser.add_argument("--tau", type=float, default=1e-3, help="Discount factor, default: 0.99")
    parser.add_argument("--clip_grad_norm", type=float, default=5.0, help="Clip Grad Norm, default: 5.0")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size, default: 256")
    parser.add_argument("--gru_hidden_size", type=int, default=512, help="GRU hidden size, default: 256")
    parser.add_argument("--first_order", type=bool, default=False, help="Whether to use first order belief or not")
    parser.add_argument("--second_order", type=bool, default=False, help="Whether to use second order belief or not")

    parser.add_argument("--num_episodes_warmup", type=int, default=1000)
    parser.add_argument("--num_train_eps_per_epoch", type=int, default=1)
    parser.add_argument("--num_eval_eps_per_epoch", type=int, default=10)
    parser.add_argument("--eval_after", type=int, default=50)
    parser.add_argument("--num_train_iters_per_epoch", type=int, default=1)
    parser.add_argument("--belief_order", type=int, default=3, help="Belief order, default: 2")
    parser.add_argument("--num_collect_agents", type=int, default=5, help="Number of agents to collect data, default: 1")
    parser.add_argument("--update_collect_agents", type=int, default=100, help="Number of epochs to update the collect agents, default: 100")

    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_model_every", type=int, default=1000, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--save_ckpt_every", type=int, default=100, help="Saves the network every x epochs, default: 25")    
    parser.add_argument("--load_from_checkpoint", type=bool, default=False, help="Whether to load from checkpoint ot not")    
    parser.add_argument("--model_ckpt_fname", type=str, default="./model_ckpt/", help="model checkpoint filename")    
    parser.add_argument("--data_ckpt_fname", type=str, default="./data_ckpt/", help="model checkpoint filename")    
    args = parser.parse_args()
    return args

def save_checkpoint(log, config, agent, buffer):
    save_dir = './model_ckpt/' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename=save_dir+"checkpoint_"+config.run_name+"_agent.pth"
    if os.path.exists(filename):
        shutil.move(filename, filename.replace(".pth", "_old.pth"))
    agent.save_checkpoint(log, filename)
    
    save_dir = './data_ckpt/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename=save_dir+"buffer_"+config.run_name+".pickle"
    if os.path.exists(filename):
        shutil.move(filename, filename.replace(".pickle", "_old.pickle"))
    buffer.save_checkpoint(filename)

def load_from_checkpoint(config, agent, buffer, log):
    checkpoint_file = glob.glob(config.model_ckpt_fname+"checkpoint_"+config.run_name+"_old.pth")
    log = agent.load_checkpoint(checkpoint_file)
    print("\nLoaded model checkpoint file: {}".format(checkpoint_file))

    checkpoint_file = glob.glob(config.data_ckpt_fname+"buffer_"+config.run_name+"_old.pickle")    
    with open(checkpoint_file, 'rb') as handle:
        data = pickle.load(handle)
    buffer.load(data)
    print("\nLoaded buffer checkpoint file: {}".format(checkpoint_file))

    return log

def train_belief_networks(belief_networks, states, player_hands, valid, log):
    #states are B, seq_len, num_players, obs_size
    all_beliefs = []
    for i in range(config.num_players):
        all_beliefs.append(belief_networks[i].get_belief_batch(states[:, :, i, :].detach().cpu().numpy()))

    # all beliefs[i] is B, seq_len, num_players, belief_order, deck_size
    B = all_beliefs[0].shape[0]
    seq_len = all_beliefs[0].shape[1]

    print("Training belief networks ...")
    
    # Initialize per-sequence step losses - these will be returned
    batch_belief_losses = torch.zeros(B, seq_len, config.num_players, config.num_players, config.belief_order, device=all_beliefs[0].device)
    
    # Training - one optimizer step per network after all losses are accumulated
    for i in range(config.num_players):
        # Zero gradients once at the beginning
        belief_networks[i].optimizer.zero_grad()
        network_total_loss = 0
        
        valid_mask = valid.view(B, seq_len, 1)
        # First-order belief training
        player_targets = player_hands.reshape(B, seq_len, config.num_players, config.deck_size)  # [B, seq_len, num_players, deck_size]
        predictions = all_beliefs[i][:, :, :, 0, :].reshape(B, seq_len, config.num_players, config.deck_size)  # [B, seq_len, num_players, deck_size]
        player_targets = player_targets.view(-1, config.deck_size)
        predictions = predictions.view(-1, config.deck_size)
        first_loss = torch.nn.functional.cross_entropy(predictions, player_targets, reduction='none')
        first_loss = first_loss.view(B, seq_len, config.num_players)
        first_loss = first_loss * valid_mask
        batch_belief_losses[:, :, i, :, 0] = first_loss

        network_total_loss += first_loss.mean()
        
        # Higher-order beliefs training for this network
        if config.belief_order > 1:
            for j in range(config.num_players):
                for k in range(1, config.belief_order):
                    
                    # Get target beliefs from previous agent
                    predictions = all_beliefs[i][:, :, j, k, :].reshape(B, seq_len, config.deck_size)
                    targets = all_beliefs[j][:, :, i, k-1, :].detach().reshape(B, seq_len, config.deck_size)
                    
                    predictions = predictions.view(-1, config.deck_size)
                    targets = targets.view(-1, config.deck_size)
                    
                    higher_loss = torch.nn.functional.cross_entropy(predictions, targets, reduction='none')
                    
                    higher_loss = higher_loss.view(B, seq_len)
                    higher_loss = higher_loss * valid_mask.squeeze(-1)
                    batch_belief_losses[:, :, i, j, k] = higher_loss
                    network_total_loss += higher_loss.mean()
            
        # Update network once after all losses are accumulated
        network_total_loss.backward()
        belief_networks[i].optimizer.step()

    # Detailed logging
    first_order_mean = batch_belief_losses[:, :, :, :, 0].mean().item()
    log["First Order Belief Loss"] = first_order_mean
    # log["First Order Belief Loss Per Agent"] = {f"Agent {i}": first_order_losses[i].item() for i in range(config.num_players)}
    
    if config.belief_order > 1:
        higher_order_mean = batch_belief_losses[:, :, :, :, 1:].mean().item()
        log["Higher Order Belief Loss"] = higher_order_mean
        
        # Log per-order losses
        for j in range(1, config.belief_order):
            order_mean = batch_belief_losses[:, :, :, :, j].mean().item()
            log[f"{j+1}-Order Belief Loss"] = order_mean
            
        # Log per-agent higher order losses
        # for i in range(config.num_players):
        #     agent_higher_mean = higher_order_losses[i].mean().item()
        #     log[f"Agent {i} Higher Order Belief Loss"] = agent_higher_mean
    else:
        log["Higher Order Belief Loss"] = 0
    
    # Total loss across all networks
    log["Total Belief Network Loss"] = first_order_mean + (higher_order_mean if config.belief_order > 1 else 0)
    
    return log, batch_belief_losses

def train_agents(agent, config, buffer, belief_networks, ep_num, log):

    print("Training agents ...")
    for _ in range(config.num_train_iters_per_epoch):
        (states, obs, actions, rewards, next_obses, dones, curr_player, player_hands, valid) = buffer.sample()
        #train belief networks
        
        log, batch_belief_losses = train_belief_networks(belief_networks, states, player_hands, valid, log)
        #batch_belief_losses is B, seq_len, num_players, num_players, belief_order
        B = states.shape[0]
        seq_len = states.shape[1]
        seq_indices = np.arange(seq_len)
        curr_player_indices = seq_indices % config.num_players
        next_player_indices = (curr_player_indices + 1) % config.num_players

        # Initialize belief tensors
        curr_plyr_beliefs = []
        next_plyr_beliefs = []

        # Loop through each time step to get the correct player's beliefs

        for t in range(seq_len):
            curr_idx = curr_player_indices[t]
            next_idx = next_player_indices[t]
            
            # Get beliefs for current player at this timestep
            curr_belief = belief_networks[curr_idx].get_belief_batch(obs[:, t:t+1].detach().cpu().numpy()).detach()
            
            # Get beliefs for next player at this timestep
            next_belief = belief_networks[next_idx].get_belief_batch(next_obses[:, t:t+1].detach().cpu().numpy()).detach()
            
            curr_plyr_beliefs.append(curr_belief)
            next_plyr_beliefs.append(next_belief)
        

        # Concatenate along the sequence dimension
        curr_plyr_beliefs = torch.stack(curr_plyr_beliefs, dim=1).squeeze(2)
        next_plyr_beliefs = torch.stack(next_plyr_beliefs, dim=1).squeeze(2)

        # For later use in belief reward calculation
        curr_player = torch.tensor(curr_player_indices, device=states.device)

        #belief reward is the sum of the batch belief losses of other players' belief about the current player
        current_players = curr_player.reshape(1, seq_len, 1, 1, 1).expand(B, seq_len, config.num_players, 1, 1)
        player_indices = torch.arange(config.num_players, device=states.device).view(1, 1, -1, 1, 1).expand(B, seq_len, config.num_players, 1, 1)
        mask = (player_indices != current_players).squeeze(-1).float()

        # Get the belief losses about current players at each timestep for all belief orders
        # Index using the current player at each timestep
        belief_loss_about_current = batch_belief_losses.gather(3, current_players.expand(B, seq_len, config.num_players, 1, config.belief_order))[:, :, :, 0, :]

        # Apply mask to zero out the current player's beliefs about themselves
        mask = mask.expand(B, seq_len, config.num_players, config.belief_order)
        masked_losses = belief_loss_about_current * mask
        
        # Sum across all belief orders and other players
        belief_reward = masked_losses.sum(dim=(-1, -2)) / (config.num_players - 1)
        belief_reward = belief_reward.detach()
        experiences = (obs, actions, rewards, next_obses, dones, valid, curr_plyr_beliefs, next_plyr_beliefs, belief_reward)
        bellman_error, g_norm = agent.learn(experiences)
        log["Bellmann error"] = bellman_error
        log["Grad Norm"] = g_norm
        log["Buffer Size"] = buffer.__len__()
        log["Epoch"] = ep_num
    return log

def fill_buffer(env, collect_agents, collect_belief_networks, config, buffer, num_episodes, policy_flag):
    total_train_ep_rew = 0
    total_train_steps = 0
    total_train_eps = 0
    for i in range(len(collect_agents)):
        train_ep_rew, train_steps, train_eps = collect_train_episodes(env=env, agent=collect_agents[i], belief_networks=collect_belief_networks[i], dataset=buffer, config=config, num_episodes=num_episodes, policy_flag=policy_flag)
        total_train_ep_rew += train_ep_rew
        total_train_steps += train_steps
        total_train_eps += train_eps
    total_train_ep_rew = total_train_ep_rew / len(collect_agents)
    return total_train_ep_rew, total_train_steps, total_train_eps

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    # Make the RL environment
    env = KuhnPokerEnv(number_of_players=config.num_players, deck_size=config.deck_size, betting_rounds=config.betting_rounds, ante=config.ante) # optional secondary argument
    
    logdir = config.logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    logdir = logdir + config.run_name + '_' + str(config.seed) + '_'+ time.strftime("%d-%m-%Y_%H-%M-%S")
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    #initialize the summary writer
    summ_writer = SummaryWriter(logdir, flush_secs=1, max_queue=1)

    # Observation and action sizes
    obses = env.reset()
    state_size = len(obses[0])
    action_size = env.action_space_size
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    buffer = ReplayBuffer(config, device=device)

    average10_mean = deque(maxlen=10)
    average10_best = deque(maxlen=10)
    all_logs = []

    # Initialize the agents for data collection
    #keep a queue of config.num_collect_agents agents
    collect_agents = deque(maxlen=config.num_collect_agents)
    collect_belief_networks = deque(maxlen=config.num_collect_agents)
    list_of_belief_networks = []
    for _ in range(config.num_collect_agents):
        collect_agents.append(DQNAgentBelief(state_size=state_size, action_size=action_size, config=config, device=device))
        for _ in range(config.num_players):
            list_of_belief_networks.append(AgentBeliefNetwork(state_size=state_size, action_size=action_size, config=config, device=device))
        collect_belief_networks.append(list_of_belief_networks)

    # Initialize the agents for training
    agent = DQNAgentBelief(state_size=state_size, action_size=action_size, config=config, device=device) 
    #Initialize num_agents belief networks
    belief_networks = []
    for _ in range(config.num_players):
        belief_networks.append(AgentBeliefNetwork(state_size=state_size, action_size=action_size, config=config, device=device))

    all_logs = []
        
    log = OrderedDict()
    init_eps = config.eps
    d_eps = init_eps - config.min_eps
    average10_mean = deque(maxlen=10)
    average10_best = deque(maxlen=10)
    total_train_steps = 0
    total_train_eps = 0
    if config.load_from_checkpoint:
        log = load_from_checkpoint(config, agent, buffer, log)
        total_train_steps = log["Train Steps collected"]
        total_train_eps = log["Train Episodes collected"]
        start_epoch = log["Epoch"]    
    else :
        _, train_steps, train_eps = fill_buffer(env, collect_agents, collect_belief_networks, config, buffer, config.num_episodes_warmup, policy_flag=True)
        start_epoch = 1
    for ep_num in range(start_epoch, config.epochs+1):
        print("\n\n==========================================================")
        print("Epoch : %d"%ep_num)

        #train agents
        log = train_agents(agent, config, buffer, belief_networks, ep_num, log)

        #collect more train samples with the updated policy
        train_ep_rew, train_steps, train_eps = fill_buffer(env, collect_agents, collect_belief_networks, config, buffer, config.num_train_eps_per_epoch, policy_flag=True)
        #evaluate the policy
        if ep_num % config.eval_after == 0:
            mean_episode_reward, best_episode_reward, eval_steps, win_rates = eval_policy(env=env, agent=agent, belief_networks=belief_networks, config=config, num_eval_episodes=config.num_eval_eps_per_epoch)

            average10_mean.append(mean_episode_reward)
            average10_best.append(best_episode_reward)

            average10_mean_arr = np.array(average10_mean, dtype=np.float32)
            average10_mean_arr = np.mean(average10_mean_arr, axis=0)

            average10_best_arr = np.array(average10_best, dtype=np.float32)
            average10_best_arr = np.mean(average10_best_arr, axis=0)

            log["Eval Steps"] = eval_steps
            log['Last 10 Average Mean Return'] = average10_mean_arr
            log['Last 10 Average Best Return'] = average10_best_arr
            log['Mean Eval Episodic Reward'] = mean_episode_reward
            log['Best Eval Episodic Reward'] = best_episode_reward

            # print win rates in percent format
            win_rates = np.array(win_rates, dtype=np.float32)
            win_rates = win_rates / sum(win_rates)
            win_rates = win_rates * 100

            for i in range(len(win_rates)):
                log['Win Rate Player %d'%i] = win_rates[i]

        total_train_steps+=train_steps
        total_train_eps+=train_eps

        config.eps = max(init_eps - ((ep_num*d_eps)/config.eps_frames), config.min_eps)

        log['Train Episodic Reward'] = train_ep_rew
        log["Train Steps collected"] = total_train_steps
        log["Train Episodes collected"] = total_train_eps
        log['Epsilon'] = config.eps

        all_logs.append(log)
        if ep_num % config.eval_after == 0:
            print("\nBest Eval Episodic Reward : {} | Mean Eval Episodic Reward : {}| Bellmann error: {}| Belief Loss: {}".format(best_episode_reward, mean_episode_reward, log["Bellmann error"], log["Total Belief Network Loss"]))       
            print("Win rates: ", win_rates)                 
        else :
            print("\nTrain Episodic Reward : {}| Bellmann error: {}| Belief Loss: {}".format(train_ep_rew, log["Bellmann error"], log["Total Belief Network Loss"]))        
 
        if ep_num % config.save_model_every == 0:
            save(config, save_name="QL-DQN", model=agent.network, ep=ep_num)
            for i in range(config.num_players):
                save(config, save_name=f"belief_network_{i}", model=belief_networks[i].network, ep=ep_num)
        # if ep_num % config.save_data_every == 0:
        #     buffer.save(ep_num)
        
        if ep_num % config.save_ckpt_every == 0:
            save_checkpoint(log, config, agent, buffer)
        
        if ep_num % config.update_collect_agents == 0:
            #remove the oldest agent and belief networks
            collect_agents.popleft()
            collect_belief_networks.popleft()
            #add the new agent and belief networks of the current agent and belief networks
            collect_agents.append(agent)
            collect_belief_networks.append(belief_networks)
        perform_logging(all_logs, summ_writer, ep_num)

def perform_logging(all_logs, summ_writer, ep_num) :
    log = all_logs[-1]
    for key, value in log.items():
        name = key+"/"
        summ_writer.add_scalar('{}'.format(name), value, ep_num)


if __name__ == "__main__":
    config = get_config()
    train(config)
