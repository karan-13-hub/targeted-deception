import torch
import torch.nn as nn
from networks import DDQNBeliefNetwork
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random

class DQNAgentBelief():
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

        self.network = DDQNBeliefNetwork(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=self.hidden_size,
                            num_players=self.num_players,
                            deck_size=self.deck_size,
                            belief_order=self.belief_order).to(self.device)

        self.target_net = DDQNBeliefNetwork(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=self.hidden_size,
                            num_players=self.num_players,
                            deck_size=self.deck_size,
                            belief_order=self.belief_order).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay_gamma)
        
    
    def get_action(self, obs, belief, epsilon=0.0, eval=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0).unsqueeze(0)
        belief = torch.from_numpy(belief).float().to(self.device)
        belief = belief.unsqueeze(0).unsqueeze(0)
        if eval:
            self.network.eval()
            with torch.no_grad():
                q_values = self.network(obs, belief)
            q_values = q_values.squeeze()
            action = torch.argmax(q_values).item()
            return action
        else:
            q_values = self.network(obs, belief)
            if random.random() > epsilon:
                q_values = q_values.squeeze()
                action = torch.argmax(q_values).item()
                return action
            else:
                # Select a random action
                available_actions = torch.arange(self.action_size, dtype=torch.long).to(self.device)
                action = random.choice(available_actions)
        return action.detach().cpu().item()

    def cql_loss(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()

    def learn(self, experiences):   
        obs, actions, rewards, next_obses, dones, valid, curr_plyr_beliefs, next_plyr_beliefs, belief_reward = experiences

        Q_a_s = self.network(obs, curr_plyr_beliefs)
        Q_expected = Q_a_s.gather(2, actions.long()).squeeze(-1)
        Q_expected = Q_expected * valid.squeeze(-1)
        # print("Q_values: ", Q_expected[0])
            
        with torch.no_grad():
            Q_targets_next = self.target_net(next_obses, next_plyr_beliefs)
            Q_targets_next = Q_targets_next.detach().max(2)[0]
            Q_targets = rewards.squeeze(-1) + ((self.gamma**self.num_players) * Q_targets_next * (~dones.squeeze(-1)))
            Q_targets = Q_targets + belief_reward
            Q_targets = Q_targets * valid.squeeze(-1)
            Q_targets = Q_targets.detach()

        cql1_loss = self.cql_loss(Q_a_s, actions)*0

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellman_error

        self.optimizer.zero_grad()
        q1_loss.backward()
        g_norm = clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return bellman_error.detach().item(), g_norm.detach().item()
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


    def save_checkpoint(self, log, filename):
        torch.save({
        'log': log,
        'network_state_dict': self.network.state_dict(),
        'target_net_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict()
        }, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['log']
