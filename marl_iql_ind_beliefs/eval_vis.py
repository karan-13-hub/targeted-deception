import os
import torch
import numpy as np
import argparse
import glob
from collections import defaultdict
import pandas as pd

from dqn_agent_belief_network import DQNAgentBelief
from belief_network import AgentBeliefNetwork
from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv


def get_config():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--deck_size', type=int, default=3, help='Deck size')
    parser.add_argument('--betting_rounds', type=int, default=2, help='Number of betting rounds')
    parser.add_argument('--ante', type=int, default=1, help='Ante')

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate, default: 1e-3")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.9999, help="Discount factor, default: 0.99")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor, default: 0.99")
    parser.add_argument("--tau", type=float, default=1e-3, help="Discount factor, default: 0.99")
    parser.add_argument("--clip_grad_norm", type=float, default=5.0, help="Clip Grad Norm, default: 5.0")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size, default: 256")
    parser.add_argument("--gru_hidden_size", type=int, default=512, help="GRU hidden size, default: 256")
    parser.add_argument("--belief_order", type=int, default=2, help="Belief order, default: 2")

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device')

    parser.add_argument('--models_dir', type=str, default='./trained_models/', help='Models directory')
    parser.add_argument('--num_eval_episodes', type=int, default=1000, help='Number of episodes for evaluation')
    parser.add_argument('--model_prefix', type=str, default=None, help='Prefix for model selection (e.g., "van-kp", "fo-kp", "so-kp")')
    parser.add_argument('--model_epoch', type=int, default=0, help='Specific model epoch to evaluate (0 for latest)')
    parser.add_argument('--all_against_all', action='store_true', help='Evaluate all models against each other')
    
    return parser.parse_args()


def load_models(model_path, config, state_size, action_size):
    """Load agent and belief networks from the specified path"""
    print(f"Loading models from {model_path}")
    
    # Set belief order based on model type
    model_name = os.path.basename(model_path)
    if "vn-" in model_name:
        config.belief_order = 1
    elif "fo-" in model_name:
        config.belief_order = 2
    elif "so-" in model_name:
        config.belief_order = 3
    else:
        print("Warning: Unknown model type, using default belief order")
    
    # Create agent and belief networks
    agent = DQNAgentBelief(state_size, action_size, config, device=config.device)
    belief_networks = []
    for _ in range(config.num_players):
        belief_networks.append(AgentBeliefNetwork(state_size, action_size, config, device=config.device))
    
    # Load agent state dict
    agent_path = model_path
    agent_checkpoint = torch.load(agent_path, map_location=config.device)
    agent.network.load_state_dict(agent_checkpoint)
    
    # Load belief networks state dicts
    base_path = os.path.dirname(model_path)
    model_prefix = os.path.basename(model_path).split('QL-DQN')[0]  # Get prefix like 'fo-kp-2'
    epoch = os.path.basename(model_path).split('_')[1].split('.')[0]  # Get epoch number
    
    for i in range(config.num_players):
        belief_path = os.path.join(base_path, f"{model_prefix}belief_network_{i}_{epoch}.pth")
        print(f"Loading belief network from: {belief_path}")
        if os.path.exists(belief_path):
            belief_checkpoint = torch.load(belief_path, map_location=config.device)
            belief_networks[i].network.load_state_dict(belief_checkpoint)
        else:
            print(f"Warning: Belief network file not found at {belief_path}")
    
    return agent, belief_networks


def get_latest_model(models_dir, prefix):
    """Get the path of the latest model with the given prefix"""
    pattern = os.path.join(models_dir, f"{prefix}*")
    model_files = glob.glob(pattern)
    
    # Filter only files containing QL-DQN
    model_files = [f for f in model_files if "QL-DQN" in f]
    
    if not model_files:
        return None
    
    # Extract epoch numbers and find the latest
    def extract_epoch(filename):
        try:
            return int(''.join(filter(str.isdigit, os.path.basename(filename))))
        except:
            return 0
    
    return max(model_files, key=extract_epoch)


def get_specific_model(models_dir, prefix, epoch):
    """Get a specific model with the given prefix and epoch"""
    model_path = os.path.join(models_dir, f"{prefix}QL-DQN{epoch}.pth")
    if os.path.exists(model_path):
        return model_path
    return None


def get_all_model_types(models_dir):
    """Get the latest model for each type (vn, fo, so)"""
    model_types = ["vn-kp", "fo-kp", "so-kp"]
    model_paths = {}
    
    for model_type in model_types:
        latest_model = get_latest_model(models_dir, model_type)
        if latest_model:
            model_paths[model_type] = latest_model
    
    return model_paths

def create_info_states():
    strat_info_set = {
        'k' : {'b':0, 'p':0},
        'q' : {'b':0, 'p':0},
        'j' : {'b':0, 'p':0},
        'kpb' : {'b':0, 'p':0},
        'qpb' : {'b':0, 'p':0},
        'jpb' : {'b':0, 'p':0},
        'kp' : {'b':0, 'p':0},
        'kb' : {'b':0, 'p':0},
        'qp' : {'b':0, 'p':0},
        'qb' : {'b':0, 'p':0},
        'jp' : {'b':0, 'p':0},
        'jb' : {'b':0, 'p':0},
    }
    return strat_info_set
    
def evaluate_self_play(env, agent, belief_networks, config, num_episodes):
    """Evaluate a model playing against itself"""
    print(f"Evaluating self-play for {num_episodes} episodes...")
    
    num_episodes_completed = 0
    win_counts = [0] * config.num_players
    strat_info_set = create_info_states()
    
    while num_episodes_completed < num_episodes:
        state = env.reset()
        done = False
        prev_moves = None
        info_set = None
        act = {1:'b', 0:'p'}
        hand = {2:'k', 1:'q', 0:'j'}
        while not done:
            curr_player = env.current_player
            print(f"\nCurrent Player: {curr_player}")
            obs = state[curr_player]
            plyr_hand = obs[config.num_players:config.num_players+config.deck_size].index(1)
            obs_arr = np.array(obs, dtype=np.float32)
            
            # Get beliefs for current player
            curr_belief = belief_networks[curr_player].get_belief(obs_arr.reshape(1, -1))
            curr_belief = curr_belief.detach().cpu().numpy()
            # Get action using agent with beliefs
            action = agent.get_action(obs_arr, curr_belief, eval=True)
            
            if info_set is None:
                info_set = hand[plyr_hand]
            else:
                info_set = hand[plyr_hand]+prev_moves
            strat_info_set[info_set][act[action]]+=1
            if prev_moves is not None:
                prev_moves = prev_moves+act[action]
            else:
                prev_moves = act[action]

            next_state, rewards, done, _ = env.step(action)
            print(f"Player Hand: {plyr_hand}")
            print(f" Action: {action}, Reward: {rewards}, Done: {done}")
            state = next_state.copy()
        
        # Track winner
        winner = env.winner
        print(f"The winner is {winner}")
        win_counts[winner] += 1
        num_episodes_completed += 1
        print("=========================================")
        print("\n\n")
        
        if num_episodes_completed % 100 == 0:
            print(f"Completed {num_episodes_completed}/{num_episodes} episodes")
    
    # Calculate win rates
    win_rates = [count / num_episodes for count in win_counts]
    for key in strat_info_set.keys():
        sum = 1e-10
        for act in strat_info_set[key]:
            print(act, key)
            sum = sum + strat_info_set[key][act]
        for act in strat_info_set[key]:
            strat_info_set[key][act] = strat_info_set[key][act]/sum
    df = pd.DataFrame.from_dict(strat_info_set, orient='index')
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print(df)
    return win_rates


def evaluate_cross_play(env, agent1, belief_networks1, agent2, belief_networks2, config, num_episodes):
    """Evaluate two different agents playing against each other"""
    print(f"Evaluating cross-play for {num_episodes} episodes...")
    
    agents = [agent1, agent2]
    belief_networks = [belief_networks1, belief_networks2]
    num_episodes_completed = 0
    win_counts = [0] * config.num_players
    
    while num_episodes_completed < num_episodes:
        state = env.reset()
        done = False
        
        while not done:
            curr_player = env.current_player
            obs = state[curr_player]
            obs_arr = np.array(obs, dtype=np.float32)
            
            # Get beliefs for current player
            curr_belief = belief_networks[curr_player][curr_player].get_belief(obs_arr.reshape(1, -1))
            curr_belief = curr_belief.detach().cpu().numpy()
            
            # Select action using the appropriate agent for the current player
            action = agents[curr_player].get_action(obs_arr, curr_belief, eval=True)
            
            next_state, rewards, done, _ = env.step(action)
            state = next_state.copy()
        
        # Track winner
        winner = env.winner
        win_counts[winner] += 1
        num_episodes_completed += 1
        
        if num_episodes_completed % 100 == 0:
            print(f"Completed {num_episodes_completed}/{num_episodes} episodes")
    
    # Calculate win rates
    win_rates = [count / num_episodes for count in win_counts]
    return win_rates


def main():
    config = get_config()
    
    # Create environment
    env = KuhnPokerEnv(
        number_of_players=config.num_players,
        deck_size=config.deck_size,
        betting_rounds=config.betting_rounds,
        ante=config.ante
    )
    
    # Get state and action dimensions
    state_size = len(env.reset()[0])
    action_size = env.action_space_size
    
    results = defaultdict(dict)
    
    if config.all_against_all:
        # Get the latest model for each type
        model_paths = get_all_model_types(config.models_dir)
        print(model_paths)
        
        
        if len(model_paths) < 2:
            print("Not enough models found for all-against-all evaluation")
            return
        
        # Load all models
        agents = {}
        belief_networks = {}
        for model_type, model_path in model_paths.items():
            agent, beliefs = load_models(model_path, config, state_size, action_size)
            agents[model_type] = agent
            belief_networks[model_type] = beliefs
            
            # Evaluate self-play for each model
            win_rates = evaluate_self_play(env, agent, beliefs, config, config.num_eval_episodes)
            results[f"{model_type} vs {model_type}"] = {
                f"Player {i} Win Rate": win_rates[i] for i in range(config.num_players)
            }
        
        # Evaluate cross-play for each pair of models
        for i, (model1, agent1) in enumerate(agents.items()):
            for model2, agent2 in list(agents.items())[i+1:]:
                win_rates = evaluate_cross_play(env, agent1, belief_networks[model1], agent2, belief_networks[model2], config, config.num_eval_episodes)
                results[f"{model1} vs {model2}"] = {
                    f"{model1} Win Rate": win_rates[0],
                    f"{model2} Win Rate": win_rates[1]
                }
                # Also evaluate with positions swapped
                win_rates_swapped = evaluate_cross_play(env, agent2, belief_networks[model2], agent1, belief_networks[model1], config, config.num_eval_episodes)
                results[f"{model2} vs {model1}"] = {
                    f"{model2} Win Rate": win_rates_swapped[0],
                    f"{model1} Win Rate": win_rates_swapped[1]
                }
    else:
        # Evaluate a specific model or latest of a type
        if config.model_prefix is not None and config.model_epoch > 0:
            model_path = get_specific_model(config.models_dir, config.model_prefix, config.model_epoch)
        else:
            model_path = get_latest_model(config.models_dir, config.model_prefix)
        
        if not model_path:
            print(f"No model found with prefix {config.model_prefix}")
            return
        
        agent, belief_networks = load_models(model_path, config, state_size, action_size)
        
        # Evaluate self-play
        win_rates = evaluate_self_play(env, agent, belief_networks, config, config.num_eval_episodes)
        model_name = os.path.basename(model_path).split('.')[0]
        results[f"{model_name} vs {model_name}"] = {
            f"Player {i} Win Rate": win_rates[i] for i in range(config.num_players)
        }
    
    # Display results
    print("\n===== Evaluation Results =====")
    
    # Separate self-play and cross-play results
    self_play_results = {}
    cross_play_results = {}
    
    for key, value in results.items():
        if " vs " in key:
            model1, model2 = key.split(" vs ")
            if model1 == model2:
                # Self-play results
                self_play_results[key] = value
            else:
                # Cross-play results
                cross_play_results[key] = value
    
    # Display self-play results
    print("\n----- Self-Play Results -----")
    self_play_df = pd.DataFrame.from_dict(self_play_results, orient='index')
    self_play_df = self_play_df[['Player 0 Win Rate', 'Player 1 Win Rate']]
    self_play_df = self_play_df.round(3)  # Round to 3 decimal places
    print(self_play_df.to_string())
    
    # Display cross-play results
    print("\n----- Cross-Play Results -----")
    cross_play_df = pd.DataFrame.from_dict(cross_play_results, orient='index')
    # Reorder columns to show vn-kp, fo-kp, so-kp in that order
    column_order = ['vn-kp Win Rate', 'fo-kp Win Rate', 'so-kp Win Rate']
    cross_play_df = cross_play_df.reindex(columns=column_order)
    cross_play_df = cross_play_df.round(3)  # Round to 3 decimal places
    print(cross_play_df.to_string())
    
    # Save results to Excel
    results_file = "evaluation_results.xlsx"
    with pd.ExcelWriter(results_file) as writer:
        self_play_df.to_excel(writer, sheet_name='Self-Play')
        cross_play_df.to_excel(writer, sheet_name='Cross-Play')
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()