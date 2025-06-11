import os
import torch
import numpy as np
import argparse
import glob
from collections import defaultdict
import pandas as pd

from dqn_agent_belief import DQNAgentBelief
from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv


def get_config():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--num_players', type=int, default=2, help='Number of players')
    parser.add_argument('--deck_size', type=int, default=5, help='Deck size')
    parser.add_argument('--betting_rounds', type=int, default=2, help='Number of betting rounds')
    parser.add_argument('--ante', type=int, default=1, help='Ante')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for the networks')
    parser.add_argument('--gru_hidden_size', type=int, default=512, help='GRU hidden size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='Device')
    parser.add_argument('--first_order', type=bool, default=False)
    parser.add_argument('--second_order', type=bool, default=False)
    parser.add_argument('--models_dir', type=str, default='./trained_models/', help='Models directory')
    parser.add_argument('--num_eval_episodes', type=int, default=1000, help='Number of episodes for evaluation')
    parser.add_argument('--model_prefix', type=str, default='', help='Prefix for model selection (e.g., "van-kp", "fo-kp", "so-kp")')
    parser.add_argument('--model_epoch', type=int, default=0, help='Specific model epoch to evaluate (0 for latest)')
    parser.add_argument('--all_against_all', action='store_true', help='Evaluate all models against each other')
    
    return parser.parse_args()


def load_model(model_path, config, state_size, action_size):
    """Load a model from the specified path"""
    print(f"Loading model from {model_path}")
    
    # Determine model type from filename to set appropriate flags
    model_name = os.path.basename(model_path)
    if "fo-" in model_name:
        config.first_order = True
        config.second_order = False
    elif "so-" in model_name:
        config.first_order = False
        config.second_order = True
    else:  # vanilla model
        config.first_order = False
        config.second_order = False
    
    # Create agent
    agent = DQNAgentBelief(state_size, action_size, config, device=config.device)
    
    # Load state dict
    checkpoint = torch.load(model_path, map_location=config.device)
    
    # # Handle different checkpoint formats
    # if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
    #     agent.network.load_state_dict(checkpoint['network_state_dict'])
    #     if 'target_net_state_dict' in checkpoint:
    #         agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    # else:
    agent.network.load_state_dict(checkpoint)
    # agent.target_net.load_state_dict(checkpoint)
    
    return agent


def get_latest_model(models_dir, prefix):
    """Get the path of the latest model with the given prefix"""
    pattern = os.path.join(models_dir, f"{prefix}*")
    model_files = glob.glob(pattern)
    
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
    """Get the latest model for each type (van, fo, so)"""
    model_types = ["van-kp", "fo-kp", "so-kp"]
    model_paths = {}
    
    for model_type in model_types:
        latest_model = get_latest_model(models_dir, model_type)
        if latest_model:
            model_paths[model_type] = latest_model
    
    return model_paths


def evaluate_self_play(env, agent, config, num_episodes):
    """Evaluate a model playing against itself"""
    print(f"Evaluating self-play for {num_episodes} episodes...")
    
    num_episodes_completed = 0
    win_counts = [0] * config.num_players
    
    while num_episodes_completed < num_episodes:
        state = env.reset()
        done = False
        
        while not done:
            curr_player = env.current_player
            obs = state[curr_player]
            obs_arr = np.array(obs, dtype=np.float32)
            action = agent.get_action(obs_arr, eval=True)
            
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


def evaluate_cross_play(env, agent1, agent2, config, num_episodes):
    """Evaluate two different agents playing against each other"""
    print(f"Evaluating cross-play for {num_episodes} episodes...")
    
    agents = [agent1, agent2]
    num_episodes_completed = 0
    win_counts = [0] * config.num_players
    
    while num_episodes_completed < num_episodes:
        state = env.reset()
        done = False
        
        while not done:
            curr_player = env.current_player
            obs = state[curr_player]
            obs_arr = np.array(obs, dtype=np.float32)
            
            # Select action using the appropriate agent for the current player
            action = agents[curr_player].get_action(obs_arr, eval=True)
            
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
        
        if len(model_paths) < 2:
            print("Not enough models found for all-against-all evaluation")
            return
        
        # Load all models
        agents = {}
        for model_type, model_path in model_paths.items():
            agents[model_type] = load_model(model_path, config, state_size, action_size)
            
            # Evaluate self-play for each model
            win_rates = evaluate_self_play(env, agents[model_type], config, config.num_eval_episodes)
            results[f"{model_type} vs {model_type}"] = {
                f"Player {i} Win Rate": win_rates[i] for i in range(config.num_players)
            }
        
        # Evaluate cross-play for each pair of models
        for i, (model1, agent1) in enumerate(agents.items()):
            for model2, agent2 in list(agents.items())[i+1:]:
                win_rates = evaluate_cross_play(env, agent1, agent2, config, config.num_eval_episodes)
                results[f"{model1} vs {model2}"] = {
                    f"{model1} Win Rate": win_rates[0],
                    f"{model2} Win Rate": win_rates[1]
                }
                # Also evaluate with positions swapped
                win_rates_swapped = evaluate_cross_play(env, agent2, agent1, config, config.num_eval_episodes)
                results[f"{model2} vs {model1}"] = {
                    f"{model2} Win Rate": win_rates_swapped[0],
                    f"{model1} Win Rate": win_rates_swapped[1]
                }
    else:
        # Evaluate a specific model or latest of a type
        if config.model_epoch > 0:
            model_path = get_specific_model(config.models_dir, config.model_prefix, config.model_epoch)
        else:
            model_path = get_latest_model(config.models_dir, config.model_prefix)
        
        if not model_path:
            print(f"No model found with prefix {config.model_prefix}")
            return
        
        agent = load_model(model_path, config, state_size, action_size)
        
        # Evaluate self-play
        win_rates = evaluate_self_play(env, agent, config, config.num_eval_episodes)
        model_name = os.path.basename(model_path).split('.')[0]
        results[f"{model_name} vs {model_name}"] = {
            f"Player {i} Win Rate": win_rates[i] for i in range(config.num_players)
        }
    
    # Display results
    print("\n===== Evaluation Results =====")
    df = pd.DataFrame.from_dict(results, orient='index')
    print(df)
    
    # Save results to CSV
    results_file = "evaluation_results.csv"
    df.to_csv(results_file)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()