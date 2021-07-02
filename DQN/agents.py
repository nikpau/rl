import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class NoisyAgent:
    def __init__(self, env, net, tgt_net, buffer, double=False, device ="cpu"):
        self.net = net
        self.tgt_net = tgt_net
        self.buffer = buffer
        self.env = env
        self.double = double
        
        assert device in ["cpu","cuda"], "Unknown device."

        self.device = torch.device(device)
        
    # Selects an action by just sampling from the noisy net. 
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        q_vals = self.net(state) # get q vals via network
        return torch.argmax(q_vals).item()
    
    def memorize(self, state, action, reward, n_state, done):
        self.buffer.add((state, action, reward, n_state, done))

    # Play n steps 
    def play_n_steps(self, state, n_steps, gamma):
        state_trajectory = [state] #store origin state in array
        action_trajectory = [] # Array for the n actions. Besides the first index it is never used. TODO: improve this.
        disc_reward = 0.0 # Discounted reward init
        for idx, _ in enumerate(range(n_steps)):
            action = self.select_action(state)
            new_state, reward, done, _ = self.env.step(action)

            # Discount reward according to trajectory length
            disc_reward += gamma**(idx) *reward

            state_trajectory.append(new_state)
            action_trajectory.append(action)

            if done: # Check if episode is done before the end of the n steps. Return the episode so far
                return (state_trajectory[0], action_trajectory[0], disc_reward, state_trajectory[-1], done)

            state = new_state # override current state

        # return tuple of starting state, starting action, discounted reward, and ending state and dones.
        return (state_trajectory[0], action_trajectory[0], disc_reward, state_trajectory[-1], done)
            
        
    # Calculate loss (MSE). If n_steps = 1, n_states is the next state (vanilla DQN), otherwise it is the n-th next state. 
    # This means if n_steps != 1, the tuple this function returns becomes
    # (state, action, discounted reward (G_t), final_state(after n steps))
    def calculate_loss(self, batch, n_steps, gamma):

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
        q_target = rewards_t + gamma**(n_steps) * new_q_vals # Bellman update with discount if callled
        
        return F.smooth_l1_loss(q_vals,q_target)
    
    def update_target_net(self):
        self.tgt_net.load_state_dict(self.net.state_dict())
        
# Using e-greedy policy for action selecion
class EpsGreedyAgent:
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
        self.buffer.add(state, action, reward, n_state, done)

    # Play n steps 
    def play_n_steps(self, state, n_steps, epsilon, gamma):
        state_trajectory = [state] #store origin state in array
        action_trajectory = []
        disc_reward = 0.0
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
            
        
    def calculate_loss(self, batch, n_steps, gamma):

        assert n_steps >= 1, "Cannot take less than one step"

        if self.buffer.type == "PER":
            *transition, weights, idx = batch
            states, actions, rewards, n_states, dones = transition
            weights = torch.tensor(weights)
        else:
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
        q_target = rewards_t + gamma**(n_steps) * new_q_vals # Bellman update
        
        # Update priorities if the buffer uses prioritization
        if self.buffer.type == "PER":
            td_error = torch.abs(q_target - q_vals)
            self.buffer.update_priority(idx = idx, td_error = td_error)
            cum_sq_error = (q_vals - q_target) ** 2
            return torch.mean(cum_sq_error * weights) # Weigh each loss with its importance sampling weight
        
        return nn.MSELoss()(q_vals,q_target)
    
    def update_target_net(self):
        self.tgt_net.load_state_dict(self.net.state_dict())
 