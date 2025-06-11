import torch
import torch.nn as nn
from networks import DDQNBelief
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
        self.first_order = config.first_order
        self.second_order = config.second_order

        self.network = DDQNBelief(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=self.hidden_size,
                            num_players=self.num_players,
                            deck_size=self.deck_size,
                            ).to(self.device)

        self.target_net = DDQNBelief(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=self.hidden_size,
                            num_players=self.num_players,
                            deck_size=self.deck_size,
                            ).to(self.device)
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay_gamma)
        
    
    def get_action(self, obs, epsilon=0.0, eval=False):
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0).unsqueeze(0)
        if eval:
            self.network.eval()
            with torch.no_grad():
                q_values, _ = self.network(obs)
            q_values = q_values.squeeze()
            action = torch.argmax(q_values).item()
            return action
        else:
            q_values, _ = self.network(obs)
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
        next_states, obs, actions, rewards, next_states, dones, curr_player, player_hands, valid = experiences

        Q_a_s, belief_first_order = self.network(obs)
        Q_expected = Q_a_s.gather(2, actions.long()).squeeze(-1)
        Q_expected = Q_expected * valid.squeeze(-1)
        print("Q_values: ", Q_expected[0])

        #if first order belief: my belief about other player hands
        loss_first_order = torch.tensor([0.0], device=self.device)
        if self.first_order:
            #belief_first_order B x seq_len x num_players x deck_size
            belief_first_order = valid.unsqueeze(-1) * belief_first_order
            player_hands_fo = valid.unsqueeze(-1) * player_hands

            # reduce belief_first_order and player hands to two dimensions
            belief_first_order = belief_first_order.view(-1, self.num_players*self.deck_size)
            player_hands_fo = player_hands_fo.view(-1, self.num_players*self.deck_size)

            #cross entropy loss between first hand belief and player hands
            loss_first_order = F.cross_entropy(belief_first_order, player_hands_fo, reduction='none')
            loss_first_order = loss_first_order.view(next_states.shape[0], next_states.shape[1])
            print('First order belief: ', belief_first_order[0])
            print('Player hands GT: ', player_hands_fo[0])
        #if second order belief: my belief about other players belief about my own hand
        loss_second_order = torch.tensor([0.0], device=self.device)
        if self.second_order:
            _, belief_second_order = self.network(next_states)
            #belief_second_order B x seq_len x num_players x num_players x deck_size

            # Get current player indices and expand dimensions to align with belief_second_order
            batch_size, seq_len, _ = curr_player.shape
            curr_player_indices = curr_player.expand(batch_size, seq_len, self.num_players)
            
            # Create batch indices and sequence indices
            batch_indices = torch.arange(batch_size, device=self.device).view(-1, 1, 1).expand(batch_size, seq_len, self.num_players)
            seq_indices = torch.arange(seq_len, device=self.device).view(1, -1, 1).expand(batch_size, seq_len, self.num_players)
            player_indices = torch.arange(self.num_players, device=self.device).view(1, 1, -1).expand(batch_size, seq_len, self.num_players)
            
            # Extract beliefs directly using advanced indexing
            # Shape: batch_size x seq_len x num_players x deck_size
            belief_second_order = belief_second_order[batch_indices, seq_indices, player_indices, curr_player_indices]

            # Reshape to match player_hands shape
            belief_second_order = valid.unsqueeze(-1) * belief_second_order
            player_hands_so = valid.unsqueeze(-1) * player_hands
            
            # Reduce belief_second_order and player_hands to two dimensions
            belief_second_order = belief_second_order.view(-1, self.num_players*self.deck_size)
            player_hands_so = player_hands_so.view(-1, self.num_players*self.deck_size)

            # Compute the cross-entropy loss
            loss_second_order = F.cross_entropy(belief_second_order, player_hands_so, reduction='none')
            loss_second_order = loss_second_order.view(next_states.shape[0], next_states.shape[1])
            
        with torch.no_grad():
            Q_targets_next, _ = self.target_net(next_states)
            Q_targets_next = Q_targets_next.detach().max(2)[0]
            Q_targets = rewards.squeeze(-1) + ((self.gamma**self.num_players) * Q_targets_next * (~dones.squeeze(-1)))
            print("Rewards: ", rewards[0])
            print("Q_target: ", Q_targets[0])
            if self.first_order:
                Q_targets = Q_targets - loss_first_order
            
            if self.second_order:
                Q_targets = Q_targets + loss_second_order

            Q_targets = Q_targets * valid.squeeze(-1)
            Q_targets = Q_targets.detach()

        cql1_loss = self.cql_loss(Q_a_s, actions)*0

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellman_error

        if self.first_order:
            print("Loss first order: ", loss_first_order.mean())
            q1_loss = q1_loss + loss_first_order.mean()
        
        if self.second_order:
            q1_loss = q1_loss - loss_second_order.mean()
        self.optimizer.zero_grad()
        q1_loss.backward()
        g_norm = clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_net)
        return bellman_error.detach().item(), loss_first_order.mean().item(), loss_second_order.mean().item(), g_norm.detach().item()
        
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
