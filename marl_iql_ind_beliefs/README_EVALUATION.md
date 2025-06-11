# Model Evaluation Tools

This directory contains tools to evaluate the performance of different belief models in the Kuhn Poker environment.

## Available Models

The codebase trains several types of models:

1. **Vanilla Model (`van-kp`)**: Basic DQN without belief modeling
2. **First-Order Belief Model (`fo-kp`)**: Adds modeling of opponent hand beliefs 
3. **Second-Order Belief Model (`so-kp`)**: Adds second-order beliefs (what the opponent thinks you have)

## Evaluation Scripts

### 1. `eval_models.py`

This script performs the basic evaluation of models, allowing both self-play and cross-play evaluations.

**Usage:**

```bash
# Evaluate a specific model in self-play
python eval_models.py --model_prefix "van-kp" --model_epoch 15000 --num_eval_episodes 1000

# Evaluate all models against each other
python eval_models.py --all_against_all --num_eval_episodes 1000
```

**Arguments:**
- `--model_prefix`: Prefix for the model to evaluate (e.g., "van-kp", "fo-kp", "so-kp")
- `--model_epoch`: Specific training epoch to evaluate (0 for latest)
- `--num_eval_episodes`: Number of episodes to use for evaluation
- `--all_against_all`: Evaluate all model types against each other
- `--num_players`: Number of players in the game (default: 2)
- `--deck_size`: Deck size to use (default: 5)
- `--betting_rounds`: Number of betting rounds (default: 2)
- `--ante`: Ante amount (default: 1)

### 2. `run_evaluations.py`

This script runs a comprehensive set of evaluations, including:
- Latest model cross-play 
- Training progression evaluation
- Milestone cross-play evaluations

**Usage:**

```bash
python run_evaluations.py
```

This will:
1. Find available models in the `trained_models` directory
2. Run cross-play evaluation for the latest models
3. Evaluate each model type's progression across training
4. Run milestone cross-play evaluations at common epochs (e.g., 5000, 10000, etc.)

### 3. `visualize_results.py`

This script creates visualizations from the evaluation results.

**Usage:**

```bash
python visualize_results.py
```

This produces several plots:
- Self-play win rates across training epochs
- Cross-play heatmap between different models
- Bar charts of latest cross-play results

## Output Files

The evaluation scripts produce the following output files:
- `evaluation_results.csv`: Results from the most recent evaluation
- `*_progression.csv`: Training progression results for each model type
- `milestone_cross_play_results.csv`: Results from milestone cross-play evaluations
- Various visualization PNG files

## Example Workflow

```bash
# 1. Run a basic evaluation of the latest models
python eval_models.py --all_against_all --num_eval_episodes 500

# 2. Run comprehensive evaluations
python run_evaluations.py

# 3. Generate visualizations
python visualize_results.py
```

## Interpreting Results

- **Self-play win rates**: In a fair game like Kuhn Poker, win rates in self-play should approach theoretical optimal values
- **Cross-play win rates**: Higher win rates against other models indicate stronger performance
- **Training progression**: Shows how model performance evolves during training 