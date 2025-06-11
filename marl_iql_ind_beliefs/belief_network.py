import torch
import torch.nn as nn
from networks import BeliefNetwork
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random

class AgentBeliefNetwork():
    def __init__(self, state_size, action_size, config, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = config.tau
        self.gamma = config.gamma
        self.clip_grad_norm = config.clip_grad_norm
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.lr = config.learning_rate
        self.lr_decay_gamma = config.lr_decay_gamma
        self.deck_size = config.deck_size
        self.num_players = config.num_players
        self.belief_order = config.belief_order
        self.network = BeliefNetwork(state_size=self.state_size,
                                     belief_order=self.belief_order,
                                     layer_size=self.hidden_size,
                                     num_players=self.num_players,
                                     deck_size=self.deck_size).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay_gamma)
        
    
    def get_belief(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0).unsqueeze(0)
        belief = self.network(obs)
        return belief
    
    def get_belief_batch(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        belief = self.network(obs)
        return belief
