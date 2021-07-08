import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(Actor,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400,300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400,300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, states, actions):
        cat_input = torch.cat([states,actions],dim=1) # Concat states and actions
        return self.net(cat_input)
