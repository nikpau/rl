"""
This replay buffer stores tuples of type (state, action, reward, new_state, done).

The uniform buffers uses uniform sampling from the deque, while the [IMPL PrioReplay]
"""
from numpy.random import beta
import bin_heap
import collections
import random
import numpy as np 

# Uniform replay buffer from which uniform sampling is performed
class ReplayBufferUniform:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=self.buffer_size) # Initialize buffer
        self.batch_size = batch_size

        # Give type for distinguishing in the agent functions 
        self.type = "UNI"
        
    def add(self, transition):
        self.buffer.append(transition) # Append to buffer

    def len(self):
        return len(self.buffer)
        
    def sample(self):
        indices = np.random.choice(self.buffer_size, self.batch_size, replace = False) # replace = False makes it O(n)
        states, actions, rewards, n_states, dones = zip(*[self.buffer[index] for index in indices])
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), \
            np.array(rewards, dtype=np.float32), np.array(n_states, dtype=np.float32), np.array(dones)
            
class PropotionalPERBuffer:
    def __init__(self, buffer_size: int, alpha: float, beta_init: float, beta_inc: float, heapify_freq: int,
                 batch_size: int):
        
        self.buffer_size = buffer_size
        self.currSize = 0

        # Priority and IS params
        self.alpha = alpha
        self.beta = beta_init
        self.beta_inc = beta_inc
        self.max_prio = 1.0
        
        # Give type for distinguishing in the agent functions
        self.type = "PER"

        
        # This is the binary heap for the transition tuple
        # (state, action, reward, n_state, done, prio)
        self.heap = bin_heap.MinHeap(max_size=self.buffer_size, key_idx=5) # Heap init
        self.heap_buffer = [(0,0,0,0,0,-self.max_prio)] * self.buffer_size
        self.heap.build_heap(self.heap_buffer) # Create MinHeap from Buffer List
        self.heapify_freq = heapify_freq # Frequency to re-heapify to avoid getting the too unbalanced

        # Pointer to index of current filling
        self.fill_ptr = 0

        # Define the batch size beforehand
        self.batch_size = batch_size

        chunk_gen = self.chunk(self.heap.heaplist, self.buffer_size // self.batch_size)
    
    # Generator function for splitting the heap into chunks of size batch_size
    def chunk(self, chunk_size):
        for i in range(0,len(self.buffer_size), chunk_size):
            yield self.heap.heaplist[i:i+chunk_size]
        
    def len(self):
        return self.currSize

    def add(self, state, action, reward, n_state, done):

        # Set the priority to the maximum seen so far
        prio = self.max_prio ** self.alpha
        
        # Add transition tuple to the heap at point fill_ptr
        self.heap.insert((state, action, reward, n_state, done, -prio)) # -prio because I use a min heap
        # TODO: Implement a max heap to get rid of the inverse prios
        
        # Increase fill ptr by one
        self.fill_ptr = (self.fill_ptr + 1) % self.buffer_size
        self.currSize = min(self.currSize + 1, self.buffer_size)
        
    def sample(self):
        
        
        # Increase beta by beta_inc
        self.beta = min(1.,self.beta + self.beta_inc)
        
        # Sample from the buffer.
        # We do not need to filter for priorities as the Heap is already organized
        batch = self.heap.heaplist[:self.batch_size]
        
        # Take the sampled batch to calculate IS weights
        prios = [-x[5] for x in batch] # Extract prios and reverse sign as they are saved with flipped sign
        weights = [(x * self.currSize)**(-self.beta) for x in prios]

        # TODO: Finish Impl....
        
class ProportionalPERBuffer_noHeap:
    def __init__(self, buffer_size, alpha, beta_init, beta_inc, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.currSize = 0

        # Priority and IS params
        self.alpha = alpha
        self.beta = beta_init
        self.beta_inc = beta_inc
        
        # Give type for distinguishing in the agent functions
        self.type = "PER"
        
        # Environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.max_prio = 1.0
        self.fill_ptr = 0
        
        # This gets filled with (|td_error| + e) ** alpha
        self.priorities_alpha = np.zeros(self.buffer_size)

        # Every transition tuple element gets its own list
        dtype = np.float32
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=dtype)
        self.action = np.zeros(self.buffer_size,dtype=np.int64)
        self.reward = np.zeros(self.buffer_size,dtype=dtype)
        self.n_state = np.zeros((self.buffer_size, self.state_dim),dtype=dtype)
        self.dones = np.zeros(self.buffer_size,dtype=dtype)
            
    def len(self):
        return self.currSize
    
    def add(self, state, action, reward, n_state, done):
        
        # Update the priority of current transition to the highest seen so far
        self.priorities_alpha[self.fill_ptr] = self.max_prio ** self.alpha
        
        # Fill buffer with current transition
        self.states[self.fill_ptr] = state
        self.action[self.fill_ptr] = action
        self.reward[self.fill_ptr] = reward
        self.n_state[self.fill_ptr] = n_state
        self.dones[self.fill_ptr] = done
         
        # Increase fill ptr by one
        self.fill_ptr = (self.fill_ptr + 1) % self.buffer_size
        self.currSize = min(self.currSize + 1, self.buffer_size)

    def sample(self, batch_size):
        # Increase beta by beta_inc
        self.beta = min(1.,self.beta + self.beta_inc)
        
        # Calculate the probabilities of sampling for all 
        # current transitions in the buffer
        current_prios = self.priorities_alpha[:self.currSize]
        current_probs = current_prios / np.sum(current_prios)
        
        # Get the indices of the sampled transitions based 
        # on their probabilties
        sample_idx = np.random.choice(self.currSize, batch_size, p = current_probs)

        # Calculate importance sampling weights 
        sampled_probs = current_probs[sample_idx]
        weights = (self.currSize * sampled_probs) ** (-self.beta)
        
        # Normalize weights 
        max_weights = (np.min(current_probs) * self.currSize) ** (-self.beta)
        weights = weights / max_weights
        
        return (self.states[sample_idx],self.action[sample_idx],self.reward[sample_idx],
                self.n_state[sample_idx],self.dones[sample_idx],weights,sample_idx)
        
    def update_priority(self, idx, td_error):
        
        # Transform td_error to numpy array
        td_error = td_error.detach().numpy()
        
        # Small epsilon to avoid zero priority
        eps = 1e-5
        
        for index, buffer_index in enumerate(idx):
           tmp_error = td_error[index]
           
           # Update the priorities with alpha
           self.priorities_alpha[buffer_index] = (tmp_error + eps) ** self.alpha 
           
           # Update maxium ever seen priority
           self.max_prio = max(self.max_prio, tmp_error)