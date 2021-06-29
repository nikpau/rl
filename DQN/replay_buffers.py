"""
This replay buffer stores tuples of type (state, action, reward, new_state, done).

The uniform buffers uses uniform sampling from the deque, while the [IMPL PrioReplay]
"""

import collections
import numpy as np 

# Uniform replay buffer from which uniform sampling is performed
class ReplayBufferUniform:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=self.buffer_size) # Initialize buffer
        
    def add(self, transition):
        self.buffer.append(transition) # Append to buffer

    def len(self):
        return len(self.buffer)
        
    def sample(self, batch_size):
        indices = np.random.choice(self.buffer_size, batch_size, replace = False) # replace = False makes it O(n)
        states, actions, rewards, n_states, dones = zip(*[self.buffer[index] for index in indices])
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), \
            np.array(rewards, dtype=np.float32), np.array(n_states, dtype=np.float32), np.array(dones)

