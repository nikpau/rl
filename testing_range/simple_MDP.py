"""
This is a toy environment based on figure 6.5 from Sutton & Barto's book "Reinforcement Learning: An Introduction"
"""

import numpy as np

class ToyMDP:
    def __init__(self, mu, gamma) -> None:
        self.mu = mu
        self.gamma = gamma

        self.s = "A"

        self.Q_star = {("A","l"): self.mu * self.gamma, ("A","r"): 0}

        for i in range(8):
            self.Q_star[("B",i)] = self.mu

    # returns tuple (n_state,reward,done)
    def step(self, a):

        if self.s == "A" and a == "l":
            self.s = "B"
            return (self.s,0,False)

        elif self.s == "A" and a == "r":
            self.s = "end"
            return (self.s,0,True)

        elif self.s == "B":
            self.s = "end"
            return (self.s,np.random.normal(loc = self.mu, scale=1),True)
    
    def reset(self):
        self.s = "A"
        return self.s