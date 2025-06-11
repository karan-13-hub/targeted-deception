import torch
import torch.nn as nn

class DDQRN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, gru_hidden_size):
        super(DDQRN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.ff_1 = nn.Linear(self.input_shape, layer_size)
        self.ff_2 = nn.Linear(layer_size, layer_size//2)
        self.gru = nn.GRU(layer_size//2, gru_hidden_size, batch_first=True)
        self.ff_3 = nn.Linear(gru_hidden_size, action_size)

    def forward(self, input, hidden_state=None):
        """
        x: (batch_size, seq_len, state_size)
        hidden_state: (batch_size, 1, hidden_size)
        Returns Q-values for each time step, plus updated hidden state.
        """
        batch_size, seq_len, _ = input.size()

        # Flatten partial MLP across time
        x = torch.relu(self.ff_1(input))
        x = torch.relu(self.ff_2(x))
        
        # GRU expects (batch_size, seq_len, feature_size)
        out, hidden_state = self.gru(x, hidden_state)

        # Flatten out -> (batch_size * seq_len, hidden_size)
        out = out.contiguous().view(-1, out.size(-1))
        q_values = self.ff_3(out)

        # Reshape back to (batch_size, seq_len, action_size)
        q_values = q_values.view(batch_size, seq_len, -1)
        return q_values, hidden_state
    
class DDQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size):
        super(DDQN, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size//2)
        self.ff_2 = nn.Linear(layer_size//2, action_size)

    def forward(self, input):
        """
        x: (batch_size, state_size)
        Returns Q-values for each action.
        """
        x = torch.relu(self.head_1(input))
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        return out

class DDQNBelief(nn.Module):
    def __init__(self, state_size, action_size, layer_size, num_players, deck_size):
        super(DDQNBelief, self).__init__()
        
        # Network parameters
        self.input_shape = state_size
        self.action_size = action_size
        self.deck_size = deck_size
        self.num_players = num_players
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(self.input_shape, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size//2),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = nn.Linear(layer_size//2, action_size)
        
        # Belief head
        self.belief_head = nn.Linear(layer_size//2, num_players * deck_size)

    def forward(self, input):
        """
        input: (batch_size, state_size)
        Returns Q-values for actions and belief distribution.
        """
        # Extract features
        features = self.features(input)
        
        # Get action values
        action_values = self.action_head(features)
        
        # Get belief distribution
        belief_logits = self.belief_head(features)
        # bls = belief_logits.shape[:-1]
        # Reshape only the last channel in belief_logits
        belief_logits = belief_logits.view(*belief_logits.shape[:-1], self.num_players, self.deck_size)
        # belief = nn.functional.softmax(belief_logits, dim=-1)
        belief = belief_logits
        
        return action_values, belief


class BeliefNetwork(nn.Module):
    def __init__(self, state_size, belief_order, layer_size, num_players, deck_size):
        super(BeliefNetwork, self).__init__()
        self.input_shape = state_size
        self.belief_order = belief_order
        self.deck_size = deck_size
        self.num_players = num_players

        self.features = nn.Sequential(
            nn.Linear(self.input_shape, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size//2),
            nn.ReLU()
        )

        self.belief_head = nn.Linear(layer_size//2, num_players * self.belief_order * self.deck_size)
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
        input: (batch_size, seq_len, state_size)
        Returns Belief matrix of shape (batch_size, seq_len, num_players, belief_order, deck_size)
        """
        batch_size = input.size(0)
        seq_len = input.size(1)
        features = self.features(input)
        belief = self.belief_head(features)
        belief = belief.view(batch_size, seq_len, self.num_players, self.belief_order, self.deck_size)
        belief = self.Softmax(belief)
        return belief

class DDQNBeliefNetwork(nn.Module):
    #This network takes in the state and the belief matrix and returns the Q-values for each action
    def __init__(self, state_size, action_size, layer_size, num_players, deck_size, belief_order):
        super(DDQNBeliefNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.num_players = num_players
        self.deck_size = deck_size
        self.belief_order = belief_order

        #concatenate the state and the belief matrix
        self.input_shape = state_size +  num_players * belief_order * deck_size

        self.features = nn.Sequential(
            nn.Linear(self.input_shape, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size//2),   
            nn.ReLU(),
            nn.Linear(layer_size//2, layer_size//2),
            nn.ReLU()
        )

        self.action_head = nn.Linear(layer_size//2, action_size)

    def forward(self, state, belief):
        """
        input: (batch_size, seq_len, state_size + num_players * belief_order * deck_size)
        Returns Q-values for each action.
        """
        batch_size = state.size(0)
        seq_len = state.size(1)
        belief = belief.view(batch_size, seq_len, -1)

        #concatenate the state and the belief matrix
        input = torch.cat((state, belief), dim=-1)

        features = self.features(input)
        action_values = self.action_head(features)
        return action_values
