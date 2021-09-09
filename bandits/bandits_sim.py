import numpy as np


def simulate(algorithm, iterations, trials):

    trial_list = []
    iter_list = []
    rewards = []
    chosen_arms = []
    cum_rewards = []
    max_mu_estimate = []
    rmse_list = []
    bias_list = []

    for trial in range(trials):

        algorithm.reset()
        #trials.append(trial)

        for iter_no in range(iterations):

            iter_list.append(iter_no)
            trial_list.append(trial +1)
            tot_index = trial * iterations + iter_no

            chosen_arm = algorithm.select_arm() # select which arm to play
            chosen_arms.append(chosen_arm) # Log the chosen arm
            reward = algorithm.play(chosen_arm) # Play the selected arm

            if iter_no == 0:
                cum_rewards.append(reward)
            else:
                cum_rewards.append(cum_rewards[tot_index - 1] + reward)

            algorithm.update(reward,chosen_arm) # Update mean and variance of the arm
            max_mu = algorithm.max_mu() # Calculate estimated max_mu
            rmse, bias = algorithm.error() # MSE and bias to the largest true expected value

            print(f"iter_no: {iter_no} | trial: {trial+1} | max_mu_estimate:{round(max_mu,4)}")
            
        rmse_list.append(rmse)
        bias_list.append(bias)
        rewards.append(reward) # Log the received reward
        max_mu_estimate.append(max_mu)

    #return trial_list, iter_list, chosen_arms, rewards, cum_rewards, max_mu_estimate, mse_list, bias_list
    return np.mean(max_mu_estimate), np.mean(rmse_list), np.mean(bias_list)
