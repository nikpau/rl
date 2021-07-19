import numpy as np 
import torch
from helper import zipf_quantiles, MinHeap
import random

# Uniform replay buffer from which uniform sampling is performed
class ReplayBufferUniform:
    def __init__(self, action_dim, state_dim, buffer_length, batch_size, device):
        self.max_size   = buffer_length
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device
    
        self.name = "Uniform"
        
        self.state  = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action  = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.n_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones  = np.zeros((self.max_size, 1))
    
    def add(self, state, action, reward, n_state, done):
        """state and n_state are np.arrays of shape (state_dim,)."""
        self.state[self.ptr]  = state
        self.action[self.ptr]  = action
        self.reward[self.ptr]  = reward
        self.n_state[self.ptr] = n_state
        self.dones[self.ptr]  = done

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self):
        ind = np.random.randint(low=0, high=self.size, size=self.batch_size)

        return (torch.tensor(self.state[ind]).to(self.device), 
                torch.tensor(self.action[ind]).to(self.device), 
                torch.tensor(self.reward[ind]).to(self.device), 
                torch.tensor(self.n_state[ind]).to(self.device), 
                torch.BoolTensor(self.dones[ind]).to(self.device))

    def len(self):
        return self.size

# Rank based Priority Experience Replay Buffer after Schaul et al. (2016)
class Rank_Based_PER_Buffer:
    def __init__(self, state_dim, action_dim, alpha, beta, beta_inc, max_size, batch_size, device) -> None:
        
        # Input args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.max_size = max_size
        self.batch_size = batch_size
        self.max_TD_error = 1.0
        self.ptr = 0
        self.name = "RankPER"

        # assert device in ["cpu", "cuda"], "Unknown device"
        self.device = device
 
        # Init heap

        # 6th (5 due to zero indexing) element is the key for the heap (td_error)
        self.heap = MinHeap(max_size = self.max_size, key_idx = 5)

        # Split the heap into groups with equal probability.
        # Implemented as batch_size quantiles from the zipf distribution.
        self.segments, self.probs = zipf_quantiles(self.max_size, self.batch_size, self.alpha)
        
    def len(self):
        return self.ptr

    def add(self, state, action, reward, next_state, done):

        # Build transition for heap. Add highest td_error seen so far
        tranistion = (state, action, reward, next_state, done, -self.max_TD_error)

        # Insert into heap
        self.heap.insert(tranistion)

        self.ptr = min(self.ptr + 1, self.max_size)

    def sample(self):

        # increase beta
        self.beta = min(1., self.beta + self.beta_inc)

        transition_list = []
        weights = []
        index_list = []

        for start,end in self.segments:
            index = random.randint(start,end) # Get one index per batch_size chunks of heap
            index_list.append(index) # Store index in list
            transition = self.heap.heaplist[index] # Extract transitions from heap
            transition_list.append(transition) # Append transition to already sampled ones
            weights.append((self.max_size * self.probs[index - 1])** -self.beta) # Append IS weights to list

        max_weight = max(weights)
        weights = [x/max_weight for x in weights] # In sample weights
            
        states, actions, rewards, next_states, dones, _ = zip(*transition_list)

        return (torch.FloatTensor(states).reshape(self.batch_size, self.state_dim).to(self.device),
                torch.FloatTensor(actions).reshape(self.batch_size, self.action_dim).to(self.device),
                torch.FloatTensor(rewards).reshape(self.batch_size,1).to(self.device),
                torch.FloatTensor(next_states).reshape(self.batch_size, self.state_dim).to(self.device),
                torch.BoolTensor(dones).reshape(self.batch_size,1).to(self.device),
                torch.FloatTensor(weights).to(self.device),
                index_list)

    def update_prio(self, idx, TD_error):

        TD_error = TD_error.detach().cpu().numpy().reshape(self.batch_size)

        for i, buf_index in enumerate(idx):
            # Replace the key of transition at buf_index in the heap with the new td_error
            self.heap.replace_key(buf_index,-TD_error[i])

            # Update maximum td_error ever seen
            self.max_TD_error = min(self.max_TD_error,-TD_error[i])
