#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import seaborn as sns

def plot_self_play_progression(file_pattern="*_progression.csv"):
    """Plot win rates for each model type across different epochs"""
    progression_files = glob.glob(file_pattern)
    
    if not progression_files:
        print(f"No progression files found matching pattern: {file_pattern}")
        return
    
    plt.figure(figsize=(12, 8))
    
    for file in progression_files:
        model_type = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file)
        
        # Extract player win rates
        player_cols = [col for col in df.columns if "Player" in col and "Win Rate" in col]
        
        for col in player_cols:
            plt.plot(df['Epoch'], df[col], marker='o', label=f"{model_type} - {col}")
    
    plt.title("Self-Play Win Rates Across Training Epochs")
    plt.xlabel("Training Epoch")
    plt.ylabel("Win Rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig("self_play_progression.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Self-play progression plot saved as 'self_play_progression.png'")

def plot_cross_play_heatmap(file="milestone_cross_play_results.csv"):
    """Create a heatmap of cross-play win rates between different models"""
    if not os.path.exists(file):
        print(f"Cross-play results file not found: {file}")
        return
    
    df = pd.read_csv(file)
    
    # Extract model names from the index
    df['Match'] = df.iloc[:, 0]
    
    # Create a matrix for the heatmap
    matches = []
    for match in df['Match']:
        if " vs " in match:
            model1, model2 = match.split(" vs ")
            matches.append((model1, model2))
    
    unique_models = sorted(list(set([m for pair in matches for m in pair])))
    
    # Create a matrix of win rates
    matrix = np.zeros((len(unique_models), len(unique_models)))
    model_to_idx = {model: i for i, model in enumerate(unique_models)}
    
    for i, row in df.iterrows():
        match = row['Match']
        if " vs " in match:
            model1, model2 = match.split(" vs ")
            
            # Find the win rate for model1
            win_rate_col = next((col for col in df.columns if model1 in col and "Win Rate" in col), None)
            if win_rate_col:
                win_rate = row[win_rate_col]
                matrix[model_to_idx[model1], model_to_idx[model2]] = win_rate
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=unique_models, yticklabels=unique_models)
    plt.title("Cross-Play Win Rates (Row Model vs Column Model)")
    plt.tight_layout()
    
    plt.savefig("cross_play_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Cross-play heatmap saved as 'cross_play_heatmap.png'")

def plot_latest_cross_play(file="evaluation_results.csv"):
    """Plot bar chart of latest cross-play results"""
    if not os.path.exists(file):
        print(f"Evaluation results file not found: {file}")
        return
    
    df = pd.read_csv(file)
    
    # Extract model prefixes
    model_types = ["van-kp", "fo-kp", "so-kp"]
    
    # Filter for cross-play matches only
    cross_play_rows = []
    for i, row in df.iterrows():
        match = row.iloc[0]
        if " vs " in match:
            model1, model2 = match.split(" vs ")
            
            # Check if this is a match between different model types
            prefix1 = next((prefix for prefix in model_types if prefix in model1), None)
            prefix2 = next((prefix for prefix in model_types if prefix in model2), None)
            
            if prefix1 and prefix2 and prefix1 != prefix2:
                cross_play_rows.append(row)
    
    if not cross_play_rows:
        print("No cross-play matches found in results")
        return
    
    # Create DataFrame from filtered rows
    cross_play_df = pd.DataFrame(cross_play_rows)
    
    # Plot the results
    plt.figure(figsize=(15, 8))
    
    matches = cross_play_df.iloc[:, 0].tolist()
    x = np.arange(len(matches))
    width = 0.35
    
    # Extract win rates for each model in the match
    win_rates1 = []
    win_rates2 = []
    labels1 = []
    labels2 = []
    
    for i, match in enumerate(matches):
        model1, model2 = match.split(" vs ")
        
        win_rate_col1 = next((col for col in cross_play_df.columns if model1 in col and "Win Rate" in col), None)
        win_rate_col2 = next((col for col in cross_play_df.columns if model2 in col and "Win Rate" in col), None)
        
        if win_rate_col1 and win_rate_col2:
            win_rates1.append(cross_play_df.iloc[i][win_rate_col1])
            win_rates2.append(cross_play_df.iloc[i][win_rate_col2])
            labels1.append(model1)
            labels2.append(model2)
    
    # Plot bars
    plt.bar(x - width/2, win_rates1, width, label=[l.split('QL')[0] for l in labels1])
    plt.bar(x + width/2, win_rates2, width, label=[l.split('QL')[0] for l in labels2])
    
    plt.ylabel('Win Rate')
    plt.title('Cross-Play Win Rates')
    plt.xticks(x, [f"{l1.split('QL')[0]} vs {l2.split('QL')[0]}" for l1, l2 in zip(labels1, labels2)], rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.tight_layout()
    plt.savefig("latest_cross_play.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Latest cross-play results plot saved as 'latest_cross_play.png'")

def main():
    # Check if required files exist
    if not os.path.exists("evaluation_results.csv"):
        print("Main evaluation results file not found. Run eval_models.py first.")
        return
    
    # Plot self-play progression
    progression_files = glob.glob("*_progression.csv")
    if progression_files:
        plot_self_play_progression()
    else:
        print("No progression files found. Run run_evaluations.py to generate progression data.")
    
    # Plot cross-play heatmap
    if os.path.exists("milestone_cross_play_results.csv"):
        plot_cross_play_heatmap()
    else:
        print("Milestone cross-play results file not found. Run run_evaluations.py to generate milestone data.")
    
    # Plot latest cross-play results
    plot_latest_cross_play()
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 