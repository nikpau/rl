import torch.nn as nn 
import torch.nn.functional as F 
from noisy_linear_layer import NoisyLinear

class NoisyNetwork(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(NoisyNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_obs[0],64),
            nn.ReLU(),
            nn.Linear(64,512),
            nn.ReLU(),
            NoisyLinear(512,n_actions,bias=True)
        )

        self.feature = nn.Linear(n_obs[0],128)
        self.noisy1 = NoisyLinear(128,128,bias=True)
        self.noisy2 = NoisyLinear(128,n_actions,bias=True)
        
    def forward(self, x):
        return self.net(x)
        
'''     def forward(self,x):
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy1(feature))
        return self.noisy2(hidden) '''


class NormalNetwork(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(NormalNetwork, self).__init__()
        
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