"""
This is the main simulation script for the multi-armed bandits for several different estimators
"""

import pandas as pd
import matplotlib.pyplot as plt

def format_results(results):
   df = pd.DataFrame(data = [results], columns=["max_mu_estimate","mse","bias"])
   return df

from bandits import OSSBandit, MEBandit, DEBandit
from bandits_sim import simulate

if __name__ == "__main__":
    
    ITERATIONS = 100_000
    TRIALS = 10

    n_arms = 10
    alpha = 0.05
    probs = [0.5] *n_arms 

    # Maximum estimator
    me_bandit = MEBandit(n_arms, probs)
    results_me = simulate(me_bandit,iterations=ITERATIONS,trials=TRIALS)
    df_me = format_results(results_me)

    mse_me = df_me["mse"].iloc[-1]
    bias_me = df_me["bias"].iloc[-1]






    # Double estimator
    de_bandit = DEBandit(n_arms, probs)
    results_de = simulate(de_bandit,iterations=ITERATIONS,trials=TRIALS)
    df_de = format_results(results_de)

    mse_de = df_de["mse"].iloc[-1]
    bias_de = df_de["bias"].iloc[-1]


    # OSS Estimator
    oss_bandit = OSSBandit(n_arms, alpha, probs)
    results_oss = simulate(oss_bandit,iterations=ITERATIONS,trials=TRIALS)
    df_oss = format_results(results_oss)

    mse_oss = df_oss["mse"].iloc[-1]
    bias_oss = df_oss["bias"].iloc[-1]

    # Concat to mse and bias vector
    alg_names = ["Maximum Estimator","Double Estimator","OSS-Estimator"]
    mse = [mse_me,mse_de,mse_oss]
    bias = [bias_me,bias_de,bias_oss]

    #Plotting
    fig, ax = plt.subplots(2, 1, figsize=(16, 9))

    fig.suptitle(f"Internet Ads in a MAB Setting | Normally distributed reward")

    ax[0].bar(alg_names,mse)
    ax[0].set_title("Root Mean Square Error")
    ax[1].bar(alg_names,bias)
    ax[1].set_title("Bias")
    plt.show()
