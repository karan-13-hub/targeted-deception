#!/usr/bin/env python
import subprocess
import os
import pandas as pd
import glob
import re
from collections import defaultdict

def get_model_epochs(models_dir, prefix):
    """Get all available epochs for a specific model prefix"""
    pattern = os.path.join(models_dir, f"{prefix}QL-DQN*.pth")
    model_files = glob.glob(pattern)
    
    epochs = []
    for file in model_files:
        # Extract epoch number from filename
        match = re.search(r'(\d+)\.pth$', file)
        if match:
            epochs.append(int(match.group(1)))
    
    return sorted(epochs)

def run_self_play_evaluation(model_prefix, epoch, num_episodes=1000):
    """Run self-play evaluation for a specific model"""
    cmd = [
        "python", "eval_models.py",
        "--model_prefix", model_prefix,
        "--model_epoch", str(epoch),
        "--num_eval_episodes", str(num_episodes)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_cross_play_evaluation(num_episodes=1000):
    """Run cross-play evaluation for all model types"""
    cmd = [
        "python", "eval_models.py",
        "--all_against_all",
        "--num_eval_episodes", str(num_episodes)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_training_progression_evaluation(model_prefix, epochs, num_episodes=500):
    """Evaluate a model's performance at different training stages"""
    results = {}
    
    for epoch in epochs:
        print(f"\n=== Evaluating {model_prefix} at epoch {epoch} ===")
        cmd = [
            "python", "eval_models.py",
            "--model_prefix", model_prefix,
            "--model_epoch", str(epoch),
            "--num_eval_episodes", str(num_episodes)
        ]
        subprocess.run(cmd)
        
        # Read the results file
        df = pd.read_csv("evaluation_results.csv")
        model_name = f"{model_prefix}QL-DQN{epoch}"
        row = df.loc[df.index.str.contains(model_name)]
        
        if not row.empty:
            results[epoch] = {col: row[col].values[0] for col in row.columns if col != 'Unnamed: 0'}
    
    # Save progression results
    progression_df = pd.DataFrame.from_dict(results, orient='index')
    progression_df.index.name = 'Epoch'
    progression_df.to_csv(f"{model_prefix}_progression.csv")
    print(f"\nProgression results saved to {model_prefix}_progression.csv")

def run_milestone_cross_play(model_prefixes, milestone_epochs, num_episodes=500):
    """Run cross-play evaluations between milestone epochs of different model types"""
    results = defaultdict(dict)
    
    # Create all combinations of models and milestone epochs
    for i, prefix1 in enumerate(model_prefixes):
        for epoch1 in milestone_epochs:
            # Self-play for this milestone
            model1 = f"{prefix1}QL-DQN{epoch1}"
            cmd = [
                "python", "eval_models.py",
                "--model_prefix", prefix1,
                "--model_epoch", str(epoch1),
                "--num_eval_episodes", str(num_episodes)
            ]
            subprocess.run(cmd)
            
            # Load results
            df = pd.read_csv("evaluation_results.csv")
            row = df.loc[df.index.str.contains(f"{model1} vs {model1}")]
            if not row.empty:
                results[f"{model1} vs {model1}"] = {col: row[col].values[0] for col in row.columns if col != 'Unnamed: 0'}
            
            # Cross-play with other models
            for prefix2 in model_prefixes[i+1:]:
                for epoch2 in milestone_epochs:
                    model2 = f"{prefix2}QL-DQN{epoch2}"
                    
                    # Create temporary script for this specific cross-play
                    with open("temp_cross_play.py", "w") as f:
                        f.write(f"""
import torch
import numpy as np
from eval_models import load_model, evaluate_cross_play, get_config
from gym_kuhn_poker.envs.kuhn_poker_env import KuhnPokerEnv

def run():
    config = get_config()
    env = KuhnPokerEnv(
        number_of_players=config.num_players,
        deck_size=config.deck_size,
        betting_rounds=config.betting_rounds,
        ante=config.ante
    )
    
    state_size = len(env.reset()[0])
    action_size = env.action_space_size
    
    # Load the two models
    model_path1 = "./trained_models/{prefix1}QL-DQN{epoch1}.pth"
    model_path2 = "./trained_models/{prefix2}QL-DQN{epoch2}.pth"
    
    agent1 = load_model(model_path1, config, state_size, action_size)
    agent2 = load_model(model_path2, config, state_size, action_size)
    
    # Evaluate agent1 as player 0, agent2 as player 1
    print(f"Evaluating {prefix1}QL-DQN{epoch1} vs {prefix2}QL-DQN{epoch2}")
    win_rates = evaluate_cross_play(env, agent1, agent2, config, {num_episodes})
    print(f"Win rates: {win_rates}")
    
    results = {{
        "{model1} Win Rate": win_rates[0],
        "{model2} Win Rate": win_rates[1]
    }}
    
    # Evaluate with positions swapped
    print(f"Evaluating {prefix2}QL-DQN{epoch2} vs {prefix1}QL-DQN{epoch1}")
    win_rates_swapped = evaluate_cross_play(env, agent2, agent1, config, {num_episodes})
    print(f"Win rates (swapped): {win_rates_swapped}")
    
    results_swapped = {{
        "{model2} Win Rate": win_rates_swapped[0],
        "{model1} Win Rate": win_rates_swapped[1]
    }}
    
    return results, results_swapped

if __name__ == "__main__":
    results, results_swapped = run()
    import json
    with open("cross_play_results.json", "w") as f:
        json.dump({{"normal": results, "swapped": results_swapped}}, f)
""")
                    
                    # Run the temp script
                    subprocess.run(["python", "temp_cross_play.py"])
                    
                    # Load results
                    import json
                    with open("cross_play_results.json", "r") as f:
                        cross_results = json.load(f)
                    
                    results[f"{model1} vs {model2}"] = cross_results["normal"]
                    results[f"{model2} vs {model1}"] = cross_results["swapped"]
    
    # Save milestone cross-play results
    milestone_df = pd.DataFrame.from_dict(results, orient='index')
    milestone_df.to_csv("milestone_cross_play_results.csv")
    print(f"\nMilestone cross-play results saved to milestone_cross_play_results.csv")
    
    # Clean up
    if os.path.exists("temp_cross_play.py"):
        os.remove("temp_cross_play.py")
    if os.path.exists("cross_play_results.json"):
        os.remove("cross_play_results.json")

def main():
    models_dir = "./trained_models/"
    model_prefixes = ["van-kp", "fo-kp", "so-kp"]
    
    # 1. Get available epochs for each model type
    available_epochs = {}
    for prefix in model_prefixes:
        epochs = get_model_epochs(models_dir, prefix)
        available_epochs[prefix] = epochs
        print(f"Available epochs for {prefix}: {epochs}")
    
    # 2. Run the current best cross-play evaluation
    print("\n=== Running cross-play evaluation for latest models ===")
    run_cross_play_evaluation(num_episodes=1000)
    
    # 3. Evaluate progression for each model type (every 5000 epochs)
    for prefix in model_prefixes:
        epochs = available_epochs[prefix]
        # Select epochs at 5000 intervals, plus the latest
        if epochs:
            milestone_epochs = [e for e in epochs if e % 5000 == 0 or e == epochs[-1]]
            if milestone_epochs:
                print(f"\n=== Evaluating progression for {prefix} at epochs {milestone_epochs} ===")
                run_training_progression_evaluation(prefix, milestone_epochs, num_episodes=500)
    
    # 4. Run milestone cross-play evaluation
    print("\n=== Running milestone cross-play evaluation ===")
    # Select common milestone epochs (5000, 10000, 15000, etc.)
    common_milestones = []
    for epoch in range(5000, 50001, 5000):
        if all(epoch in available_epochs[prefix] for prefix in model_prefixes):
            common_milestones.append(epoch)
    
    if common_milestones:
        print(f"Common milestone epochs for all model types: {common_milestones}")
        run_milestone_cross_play(model_prefixes, common_milestones, num_episodes=500)
    else:
        print("No common milestone epochs found for all model types")

if __name__ == "__main__":
    main() 