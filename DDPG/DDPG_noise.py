import numpy as np 
import torch

class GaussianNoise:
    def __init__(self, action_dim, mu=0.0, sigma=0.3):
        self.action_dim = action_dim

        self.mu = mu
        self.sigma = sigma

    def sample(self):
        noise_array = np.random.normal(loc=self.mu,scale=self.sigma, size=self.action_dim)
        return torch.from_numpy(noise_array)