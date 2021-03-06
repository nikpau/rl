import gym
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import loss
import torch.nn.functional as F 
import numpy as np 
import argparse
import collections

# HYPERPARAMETERS ---------------------------------

# Env
ENV_NAME = "CartPole-v1"
ACTIONS = {0: 0, 1: 1}
BUFFER_SIZE = 50000
REWARD_BOUND = 475

# Training
EPSILON_START = 1.0
GAMMA = 0.999
EPSILON_DECAY = 0.9999
EPSILON_FINAL = 0.02
BATCH_SIZE = 256
TARGET_NET_UPDATE = 256
LEARNING_RATE = 0.001

# Network -------------------------------------

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_obs[0],32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,n_actions)
        )
        
    def forward(self, x):
        return self.net(x)
    
# Experience Replay Buffer ----------------
class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_SIZE) # Initialize buffer
        
    def add(self, transition):
        self.buffer.append(transition) # Append to buffer

    def len(self):
        return len(self.buffer)
        
    def sample(self, BATCH_SIZE):
        indices = np.random.choice(BUFFER_SIZE, BATCH_SIZE, replace = False) # replace = False makes it O(n)
        states, actions, rewards, n_states, dones = zip(*[self.buffer[index] for index in indices])
        return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64), \
            np.array(rewards, dtype=np.float32), np.array(n_states, dtype=np.float32), np.array(dones)
 
# Agent -----------------------------------    
class Agent:
    def __init__(self, env, net, tgt_net, buffer, device ="cpu"):
        self.net = net
        self.tgt_net = tgt_net
        self.buffer = buffer
        self.env = env
        
        assert device in ["cpu","cuda"], "Unknown device."

        self.device = torch.device(device)
        
    def select_action(self, state, epsilon):
        # Epsison greedy policy
        if np.random.rand() < epsilon: # take random action
            return np.random.randint(self.env.action_space.n)
        
        q_vals = self.net(torch.tensor(np.array(state,dtype=np.float32)).to(self.device)) # get q vals via network
        return torch.argmax(q_vals).item()
    
    def memorize(self, state, action, reward, n_state, done):
        self.buffer.add((state, action, reward, n_state, done))
        
    
    def calculate_loss(self, batch):
        states, actions, rewards, n_states, dones = batch # unpack batch to seperate lists 
        
        # convert to tensors
        states_t = torch.tensor(states).to(self.device)
        new_states_t = torch.tensor(n_states).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones)
        
        # estimate working q_vals with main network
        q_vals = self.net(states_t.float()).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        
        # Get next Q with target net and detach gradients for target network
        new_q_vals = self.tgt_net(new_states_t.float()).max(1)[0]
        new_q_vals[dones] = 0.0 # if episode terminated set the q values to zero as there is no next time step
        new_q_vals = new_q_vals.detach() # detach the computational graph to avoid gradients form tgt to flow to normal net
        
        # calculate expected Q value
        q_target = rewards_t + GAMMA * new_q_vals # Bellman update
        
        return nn.MSELoss()(q_vals,q_target)
    
    def update_target_net(self):
        self.tgt_net.load_state_dict(self.net.state_dict())
        
if __name__ == "__main__":

    # Initialize env
    env = gym.make(ENV_NAME)
    state = env.reset() #first state
    
    # Init Replay Buffer    
    buffer = ReplayBuffer()

    # Init networks
    net = DQN(env.observation_space.shape,env.action_space.n)
    tgt_net = DQN(env.observation_space.shape,env.action_space.n)
    
    # Optimizer init
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    # Init agent
    agent = Agent(env,net,tgt_net, buffer)

    rewards_list = []
    episode_reward = 0.0
    iter_no = 0
    episode_no = 0

    # Epsilon for trainig
    eps = EPSILON_START

    # Actual training loop
    while True:
        
        iter_no += 1
        eps = max(EPSILON_FINAL, eps * EPSILON_DECAY)
        
        # Get action from action space
        action = agent.select_action(state,eps)
        
        # take step in env and receive new state and reward
        new_state, reward, done, _ = env.step(action)
        
        # Add transition to replay buffer
        agent.memorize(state,action,reward, new_state,done)
        
        state = new_state
        episode_reward += reward
        
        if done:
            episode_no += 1
            rewards_list.append(episode_reward)
            mean_reward = np.mean(rewards_list[-100:])
            print(f"Iteration: {iter_no}. Episode: {episode_no}. Mean reward: {mean_reward}. Epsilon: {round(eps,2)}")
            
            episode_reward = 0.0
            state = env.reset()
            
        if buffer.len() < BUFFER_SIZE:
            continue
        
        if iter_no % TARGET_NET_UPDATE == 0:
            agent.update_target_net()
        
        if mean_reward > REWARD_BOUND:
            print("SOLVED")
            torch.save(net.state_dict(), "DQN/" + ENV_NAME +  "_weights.dat")
            break
        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = agent.calculate_loss(batch)
        loss_t.backward()
        optimizer.step()
        
        