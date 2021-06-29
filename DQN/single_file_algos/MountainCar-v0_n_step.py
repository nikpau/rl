"""
This is nearly the same Agent as in the CartPole-v1_n_step.py but for convenience it's a seperate file.
Apart from the tensorboard logging and the adapted rewards for the env it is not really different. 
Also new net with more nodes in the second hidden layer. (I am not yet sure why this works)
"""

import gym
from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import loss
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import argparse # Maybe sometimes
import collections

# HYPERPARAMETERS ---------------------------------

# Env
ENV_NAME = "MountainCar-v0"
ACTIONS = {0: 0, 1: 1, 2: 2}
BUFFER_SIZE = 5000
REWARD_BOUND = -110

# Training
EPSILON_START = 1.0
GAMMA = 0.999
EPSILON_DECAY = 0.9995
EPSILON_FINAL = 0.001
BATCH_SIZE = 256
TARGET_NET_UPDATE = 256
LEARNING_RATE = 0.001
N_STEPS = 4

# Network -------------------------------------

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_obs[0],32),
            nn.ReLU(),
            nn.Linear(32,128),
            nn.ReLU(),
            nn.Linear(128,64),
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
    def __init__(self, env, net, tgt_net, buffer, double=False, device ="cpu"):
        self.net = net
        self.tgt_net = tgt_net
        self.buffer = buffer
        self.env = env
        self.double = double
        
        assert device in ["cpu","cuda"], "Unknown device."

        self.device = torch.device(device)
        
    def select_action(self, state, epsilon):
        # Epsison greedy policy
        if np.random.rand() < epsilon: # take random action
            return np.random.randint(self.env.action_space.n)
        
        state = torch.FloatTensor(state).to(self.device)
        q_vals = self.net(state) # get q vals via network
        return torch.argmax(q_vals).item()
    
    def memorize(self, state, action, reward, n_state, done):
        self.buffer.add((state, action, reward, n_state, done))

    # Play n steps 
    def play_n_steps(self, state, n_steps, epsilon, gamma):
        state_trajectory = [state] #store origin state in array
        action_trajectory = [] # Array for the n actions. Besides the first index it is never used. TODO: improve this.
        disc_reward = 0.0 # Discounted reward init
        for idx, _ in enumerate(range(n_steps)):
            action = self.select_action(state, epsilon)
            new_state, reward, done, _ = self.env.step(action)

            # Discount reward according to trajectory length
            disc_reward += gamma**(idx) *reward

            state_trajectory.append(new_state)
            action_trajectory.append(action)

            if done: # Check if episode is done before the end of the n steps. Return the episode so far
                return (state_trajectory[0], action_trajectory[0], disc_reward, state_trajectory[-1],  done)

            state = new_state # override current state

        # return tuple of starting state, starting action, discounted reward, and ending state and dones.
        return (state_trajectory[0], action_trajectory[0], disc_reward, state_trajectory[-1], done)
            
        
    # Calculate loss (MSE). If n_steps = 1, n_states is the next state (vanilla DQN), otherwise it is the n-th next state. 
    # This means if n_steps != 1, the tuple this function returns becomes
    # (state, action, discounted reward (G_t), final_state(after n steps))
    def calculate_loss(self, batch, n_steps):

        assert n_steps >= 1, "Cannot take less than one step"

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
        with torch.no_grad(): # detach the computational graph to avoid gradients form tgt to flow to normal net
            if self.double:    
                n_actions = self.net(new_states_t.float()).max(1)[1]
                new_q_vals = self.tgt_net(new_states_t.float()).gather(1, n_actions.unsqueeze(-1)).squeeze(-1)
            else:
                new_q_vals = self.tgt_net(new_states_t.float()).max(1)[0]

        new_q_vals[dones] = 0.0 # if episode terminated set the q values to zero as there is no next time step
        
        # calculate expected Q value
        q_target = rewards_t + GAMMA**(n_steps) * new_q_vals # Bellman update with discount if callled
        
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
    agent = Agent(env,net,tgt_net, buffer, double=True)
    
    # Init Logger
    writer = SummaryWriter() # This is what uses the torch.utils.tensorboard btw 

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
        
        # Play n steps and return a transition tuple
        state, action, reward, new_state, done = agent.play_n_steps(state,N_STEPS, eps, GAMMA)

        # Add transition to replay buffer
        agent.memorize(state,action,reward, new_state,done)
        
        state = new_state
        episode_reward += reward
        
        if done:
            episode_no += 1
            rewards_list.append(episode_reward)
            mean_reward = np.mean(rewards_list[-100:])
            print(f"Iteration: {iter_no}. Episode: {episode_no}. Mean reward: {mean_reward:.2f}. Epsilon: {eps:.2f}")

            # TODO: implement Summary Writer
            writer.add_scalar("Reward per iteration", episode_reward, iter_no)
            writer.add_scalar("Reward per episode", episode_reward, episode_no)
            writer.add_scalar("Mean Reward at iteration", mean_reward, iter_no)
            
            
            episode_reward = 0.0
            state = env.reset()
            
        if buffer.len() < BUFFER_SIZE:
            continue
        
        if iter_no % TARGET_NET_UPDATE == 0:
            agent.update_target_net()
        
        if mean_reward > REWARD_BOUND:
            print("SOLVED")
            torch.save(net.state_dict(), "DQN/" + ENV_NAME + "_nstep_weights.dat")
            break
        
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = agent.calculate_loss(batch,N_STEPS)
        loss_t.backward()
        optimizer.step()
        
        