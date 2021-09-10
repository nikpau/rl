import random
import scipy.stats as sc
import numpy as np 

"""
Multi-armed Bandit with an e-greedy Agent and a Bernoulli modeled reward structure.
"""
class OSSBandit():
    def __init__(self, arms, alpha, probs) -> None:
        self.arms = arms
        self.alpha = alpha
        self.probs = probs
        self.ptr = 0

        self.curr_estimator_var = 0.001
        self.curr_max_mu = 0

        self.exp_value = [0.0] * arms
        self.var = [0.001] * arms
        self.counts = [1] * arms

        self.norm_quant = sc.norm().ppf(self.alpha)

    def reset(self):
        self.exp_value = [0.0] * self.arms
        self.counts = [1] * self.arms
        self.var = [0.001] * self.arms

    def select_arm(self):
        arm_to_play = self.ptr
        self.ptr = (self.ptr + 1) % self.arms
        return arm_to_play

    def update(self,reward, chosen_arm):

        # Update expected value
        self.counts[chosen_arm] += 1
        curr_count = self.counts[chosen_arm]
        curr_exp = self.exp_value[chosen_arm]
        new_exp = _running_mean(curr_count,curr_exp,reward)

        # Update variance (Welford's online algorithm)
        curr_var = self.var[chosen_arm]
        if curr_count <= 2:
            new_var = 0.001
        else:
            #new_var = ((curr_count - 2) / (curr_count - 1)) * curr_var + ((1/curr_count) * (reward - curr_exp)**2)
            new_var = _running_variance(curr_count,curr_var,curr_exp,reward)
        self.exp_value[chosen_arm] = new_exp
        self.var[chosen_arm] = new_var

    def max_mu(self):
        max_arm = np.argmax(self.exp_value) # select the index of the highest estimate
        max_val = self.exp_value[max_arm] # select the highest estimate
        arms_to_compare = [0] * self.arms # binary coded vector indicating whether a value "passed" the test

        for arm in range(self.arms):
            t_stat = (self.exp_value[arm] - max_val) / np.sqrt((self.var[arm]/self.counts[arm]) \
                + (self.var[max_arm]/self.counts[max_arm]))
            if t_stat >= self.norm_quant:
                arms_to_compare[arm] = 1 # H_0 not rejected
            else:
                arms_to_compare[arm] = 0 # H_0 rejected

        # Multiply the indicator function with the expected values
        n_ones = arms_to_compare.count(1)
        values = [ind*mu for ind,mu in zip(arms_to_compare,self.exp_value)]

        exp_est = sum(values) / n_ones
        
        self.curr_estimator_var = self._estimator_variance(exp_est)
        self.curr_max_mu = exp_est

        return exp_est
    
    def _estimator_variance(self, new_max_mu):
        curr_count = self.counts[np.argmax(self.exp_value)]
        curr_var = self.curr_estimator_var
        curr_max_mu = self.curr_max_mu
        if curr_count <= 2:
            return 0.001
        else:
            new_var = _running_variance(curr_count,curr_var,curr_max_mu,new_max_mu)
        
        return new_var
        
    def error(self):
        true_prob = max(self.probs)
        max_mu_est = self.curr_max_mu
        rmse = np.sqrt((max_mu_est - true_prob)**2 + self.curr_estimator_var)
        bias = max_mu_est - true_prob
        return rmse, bias

    def play(self, chosen_arm):
        prob = self.probs[chosen_arm]
        return np.random.binomial(1,prob)

class MEBandit():
    def __init__(self, arms, probs) -> None:
        self.arms = arms
        self.probs = probs
        self.ptr = 0

        self.curr_estimator_var = 0.001
        self.curr_max_mu = 0
        
        self.exp_value = [0.0] * arms
        self.var = [0.001] * arms
        self.counts = [1] * arms

    def reset(self):
        self.exp_value = [0.0] * self.arms
        self.counts = [1] * self.arms
        self.var = [0.001] * self.arms

    def select_arm(self):
        arm_to_play = self.ptr
        self.ptr = (self.ptr + 1) % self.arms
        return arm_to_play

    def update(self,reward, chosen_arm):

        # Update expected value
        self.counts[chosen_arm] += 1
        curr_count = self.counts[chosen_arm]
        curr_exp = self.exp_value[chosen_arm]
        new_exp = _running_mean(curr_count,curr_exp,reward)

        # Update variance (Welford's online algorithm)
        curr_var = self.var[chosen_arm]
        if curr_count <= 2:
            new_var = 0.001
        else:
            new_var = _running_variance(curr_count,curr_var,curr_exp,reward)

        self.exp_value[chosen_arm] = new_exp
        self.var[chosen_arm] = new_var

    def max_mu(self):
        max_arm = np.argmax(self.exp_value) # select the index of the highest estimate
        max_val = self.exp_value[max_arm] # select the highest estimate

        self.curr_max_mu = max_val
        self.curr_estimator_var = self._estimator_variance(max_val)
        
        return max_val

    def _estimator_variance(self, new_max_mu):
        curr_count = self.counts[np.argmax(self.exp_value)]
        curr_var = self.curr_estimator_var
        curr_max_mu = self.curr_max_mu
        if curr_count <= 2:
            return 0.001
        else:
            new_var = _running_variance(curr_count,curr_var,curr_max_mu,new_max_mu)
        
        return new_var
        
    def error(self):
        true_prob = max(self.probs)
        max_mu_est = self.max_mu()
        rmse = np.sqrt((max_mu_est - true_prob)**2 + self.curr_estimator_var)
        bias = max_mu_est -true_prob
        return rmse, bias

    def play(self, chosen_arm):
        prob = self.probs[chosen_arm]
        return np.random.binomial(1,prob)

class DEBandit():
    def __init__(self, arms, probs) -> None:
        self.arms = arms
        self.probs = probs
        self.ptr = 0

        self.curr_estimator_var = 0.001
        self.curr_max_mu = 0

        self.var_A = [0.001] * arms
        self.var_B = [0.001] * arms

        self.exp_value_A = [0.0] * arms
        self.counts_A = [1] * arms

        self.exp_value_B = [0.0] * arms
        self.counts_B = [1] * arms

    def reset(self):
        self.exp_value_A = [0.0] * self.arms
        self.exp_value_B = [0.0] * self.arms
        self.counts_A = [1] * self.arms
        self.counts_B = [1] * self.arms
        self.var_A = [0.001] * self.arms
        self.var_B = [0.001] * self.arms

    def select_arm(self):
        arm_to_play = self.ptr
        self.ptr = (self.ptr + 1) % self.arms
        return arm_to_play

    def update(self,reward, chosen_arm):
        # Update expected values alternatringly

        if random.random() < 0.5:
            self.counts_A[chosen_arm] += 1
            curr_count_A = self.counts_A[chosen_arm]
            curr_exp_A = self.exp_value_A[chosen_arm]
            new_exp_A = _running_mean(curr_count_A,curr_exp_A,reward)

            # Update variance (Welford's online algorithm)
            curr_var_A = self.var_A[chosen_arm]
            if curr_count_A <= 2:
                new_var_A = 0.001
            else:
                new_var_A = _running_variance(curr_count_A,curr_var_A,curr_exp_A,reward)
                
            self.exp_value_A[chosen_arm] = new_exp_A
            self.var_A[chosen_arm] = new_var_A
        else:
            self.counts_B[chosen_arm] += 1
            curr_count_B = self.counts_B[chosen_arm]
            curr_exp_B = self.exp_value_B[chosen_arm]
            new_exp_B = _running_mean(curr_count_B,curr_exp_B,reward)

            # Update expected value
            # Update variance (Welford's online algorithm)
            curr_var_B = self.var_B[chosen_arm]
            if curr_count_B <= 2:
                new_var_B = 0.001
            else:
                new_var_B = _running_variance(curr_count_B,curr_var_B,curr_exp_B,reward)
            self.exp_value_B[chosen_arm] = new_exp_B
            self.var_B[chosen_arm] = new_var_B


    def max_mu(self):
        max_arm_A = np.argmax(self.exp_value_A) # select the index of the highest estimate
        val_B = self.exp_value_B[max_arm_A] # select the highest estimate from the other sample

        self.curr_max_mu = val_B
        self.curr_estimator_var = self._estimator_variance(val_B)

        return val_B

    def _estimator_variance(self, new_max_mu):
        curr_count = self.counts_A[np.argmax(self.exp_value_A)] + self.counts_B[np.argmax(self.exp_value_B)]
        curr_var = self.curr_estimator_var
        curr_max_mu = self.curr_max_mu
        if curr_count <= 2:
            return 0.001
        else:
            new_var = _running_variance(n=curr_count,var=curr_var,
                                        old_mean=curr_max_mu,new_val=new_max_mu)
        
        return new_var

    def error(self):
        true_prob = max(self.probs)
        max_mu_est = self.max_mu()
        rmse = np.sqrt((max_mu_est - true_prob)**2 + self.curr_estimator_var)
        bias = max_mu_est - true_prob
        return rmse, bias

    def play(self, chosen_arm):
        prob = self.probs[chosen_arm]
        return np.random.binomial(1,prob)

def _running_variance(n,var,old_mean,new_val):
    return ((n - 2) / (n - 1)) * var + ((1/n) * (new_val - old_mean)**2)

def _running_mean(n,old_mean,new_val):
    return ((n-1) / n) * old_mean + (1/n) * new_val