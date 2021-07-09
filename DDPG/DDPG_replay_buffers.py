import numpy as np 
import torch

# Uniform replay buffer from which uniform sampling is performed
class ReplayBufferUniform:
    def __init__(self, action_dim, state_dim, buffer_length, batch_size, device):
        self.max_size   = buffer_length
        self.batch_size = batch_size
        self.ptr        = 0
        self.size       = 0
        self.device     = device
        
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
class Rank_PER_Buffer:
    def __init__(self) -> None:
        pass