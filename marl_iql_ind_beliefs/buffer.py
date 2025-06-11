import numpy as np
import random
import torch
import pickle
from collections import deque, namedtuple
import os
import shutil

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, config, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.num_players = config.num_players
        self.device = device
        self.sequence = []
        self.memory = deque(maxlen=config.buffer_size)  
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.experience = namedtuple("Experience", field_names=["state", "obs", "action", "reward", "next_obs", "done", "player_hands"])
    
    def add(self, state, obs, action, reward, next_obs, done, player_hands):
        """Add a new experience to memory."""        
        e = self.experience(state, obs, action, reward, next_obs, done, player_hands)
        self.sequence.append(e)
        if done:
            self.memory.append(self.sequence)
            self.sequence = []
        
    def sample(self):
        """Randomly sample a batch of sequences from memory."""
        sequences = random.sample(self.memory, k=self.batch_size)
        # sampled batch shape : Batch x Seq x State
        max_seq_len = max([len(seq) for seq in sequences])
        states = np.zeros((self.batch_size, max_seq_len, len(sequences[0][0].state), len(sequences[0][0].state[0])))
        obs = np.zeros((self.batch_size, max_seq_len, len(sequences[0][0].obs)))
        actions = np.zeros((self.batch_size, max_seq_len, 1))
        rewards = np.zeros((self.batch_size, max_seq_len, 1))
        next_obses = np.zeros((self.batch_size, max_seq_len, len(sequences[0][0].next_obs)))
        dones = np.zeros((self.batch_size, max_seq_len, 1))
        bootstrap = np.zeros((self.batch_size, max_seq_len, 1))
        curr_player = np.zeros((self.batch_size, max_seq_len, 1))
        valid = np.zeros((self.batch_size, max_seq_len, 1))
        player_hands = np.zeros((self.batch_size, max_seq_len, len(sequences[0][0].player_hands), len(sequences[0][0].player_hands[0])))

        for i, seq in enumerate(sequences):
            valid_idx = 0
            for j, e in enumerate(seq):
                states[i,j] = e.state
                obs[i,j] = e.obs
                actions[i,j] = e.action
                rewards[i,j] = e.reward
                next_obses[i,j] = e.next_obs
                dones[i,j] = e.done
                if e.done:
                    valid_idx = j
                curr_player[i,j] = j%self.num_players
                player_hands[i,j] = e.player_hands
            valid[i] = np.array([1 if k <= valid_idx else 0 for k in range(max_seq_len)]).reshape(-1,1)       

        #Change rewards based on the number of players
        gamma_pow = self.gamma ** np.arange(1, self.num_players)
        adjustment = np.zeros_like(rewards)
        for p in range(1, self.num_players):
            shifted = np.zeros_like(rewards)
            shifted[:, :-p] = rewards[:, p:]
            adjustment += gamma_pow[p-1] * shifted
        rewards = rewards - adjustment

        #Change the next obses based on the number of players
        p = self.num_players-1
        next_obses[:, :-p] = next_obses[:, p:]
        dones[:, :-p] = dones[:, p:]

        #Change the dones based on the number of players
        # import pdb
        # pdb.set_trace()
        states = torch.from_numpy(states).float().to(self.device)
        obs = torch.from_numpy(obs).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_obses = torch.from_numpy(next_obses).float().to(self.device)
        dones = torch.from_numpy(dones).bool().to(self.device)
        curr_player = torch.from_numpy(curr_player).long().to(self.device)
        player_hands = torch.from_numpy(player_hands).float().to(self.device)
        valid = torch.from_numpy(valid).bool().to(self.device)
        return (states, obs, actions, rewards, next_obses, dones, curr_player, player_hands, valid)
    

    def save(self, ep_num):
        save_dir = './dataset/' 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename=save_dir+"data_"+str(ep_num)+".pickle"
        if os.path.exists(filename):
            shutil.move(filename, filename.replace(".pickle", "_old.pickle"))
            
        data = []
        for seq in self.memory:
            sequence = {'states':[], 'obs':[], 'actions':[], 'rewards':[], 'next_obses':[], 'dones':[], 'player_hands':[]}
            for j, e in enumerate(seq):
                sequence['states'].append(e.state)
                sequence['obs'].append(e.obs)
                sequence['actions'].append(e.action)
                sequence['rewards'].append(e.reward)
                sequence['next_obses'].append(e.next_obs)
                sequence['dones'].append(e.done)
                sequence['player_hands'].append(e.player_hands)
            data.append(sequence)
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_checkpoint(self, filename):
        data = []        
        for seq in self.memory:
            sequence = {'states':[], 'obs':[], 'actions':[], 'rewards':[], 'next_obses':[], 'dones':[], 'player_hands':[]}
            for j, e in enumerate(seq):
                sequence['states'].append(e.state)
                sequence['obs'].append(e.obs)
                sequence['actions'].append(e.action)
                sequence['rewards'].append(e.reward)
                sequence['next_obses'].append(e.next_obs)
                sequence['dones'].append(e.done)
                sequence['player_hands'].append(e.player_hands)
            data.append(sequence)
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, data):
        for seq in data:
            self.sequence = []
            for i in range(len(seq['states'])):
                state = seq['states'][i]
                obs = seq['obs'][i]
                action = seq['actions'][i]
                reward = seq['rewards'][i]
                next_obs = seq['next_obses'][i]
                done = seq['dones'][i]
                player_hands = seq['player_hands'][i]
                e = self.experience(state, obs, action, reward, next_obs, done, player_hands)
                self.sequence.append(e)
                if done:
                    self.memory.append(self.sequence)
                    self.sequence = []
            if self.sequence:  # In case the sequence didn't end with done=True
                self.memory.append(self.sequence)
                self.sequence = []

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)