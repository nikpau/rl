"""
Here I implement a toy Q-Learning agent from Watkins (1989) and
a double Q-Learning agent after van Hasselt (2010).
"""
import random
import copy

class QAgent:

    def __init__(self, alpha, epsilon, gamma) -> None:
        
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # Table for the Q-Function, saved as a Dict of form
        # {(state, action): Q-Value}
        self.Q_table = {("A","l"): 0.0, ("A","r"): 0.0}
        for i in range(8):
            self.Q_table[("B",i)] = 0.0

    # Return only those Dict entries that belong to state "state".
    def _get_state_Q(self, state):
        Qs = []
        for key in list(self.Q_table.keys()):
            if key[0] == state:
                Qs.append(state)
        
        return {key: self.Q_table[key] for key in Qs}

    # Calculate max_a' Q(s,a')
    def _maxQ(self, state):
        state_Qs = self._get_state_Q(state)
        return max(state_Qs.values())

    # Calculate argmax_a' Q(s,a')
    def _argmax_Q(self, state):
        state_Q = self._get_state_Q(state)
        max_Q = self._maxQ(state)
        max_actions = [k[1] for k,v in state_Q.items() if v == max_Q]

        return random.choice(max_actions)

    def select_action(self, state):
        if random.random() < self.epsilon: # Explore
            state_Q = self._get_state_Q(state)
            return random.choice(list(state_Q))[1]
        else: # Greedy
            return self._argmax_Q(state)

    def update(self, state,action,reward,n_state,done):
        if done:
            Q_max = 0
        else:
            Q_max = self._maxQ(n_state)

        self.Q_table[state,action] = self.alpha * (reward + self.gamma * Q_max - self.Q_table[state,action])

    def calc_bias(self, Q_star):
        bias = {}
        for key in Q_star.keys():
            bias[key] = self.Q_table[key] - Q_star[key]
        return bias

class DoubleQAgent(QAgent):
    def __init__(self, alpha, epsilon, gamma) -> None:
        super().__init__(alpha=alpha, epsilon=epsilon, gamma=gamma)

        self.Q_table1 = copy.deepcopy(super().Q_table)
        self.Q_table2 = copy.deepcopy(super().Q_table)

    
