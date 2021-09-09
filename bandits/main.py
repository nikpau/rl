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
    TRIALS = 20

    N_ARMS = [10,100,1000]
    ALPHA = 0.05
    
    mse_list = []
    bias_list = []
    
    
    for arms in N_ARMS:
      probs = [0.5] *arms 

      # Maximum estimator
      me_bandit = MEBandit(arms, probs)
      results_me = simulate(me_bandit,iterations=ITERATIONS,trials=TRIALS)
      df_me = format_results(results_me)

      mse_me = df_me["mse"].iloc[-1]
      bias_me = df_me["bias"].iloc[-1]

      # Double estimator
      de_bandit = DEBandit(arms, probs)
      results_de = simulate(de_bandit,iterations=ITERATIONS,trials=TRIALS)
      df_de = format_results(results_de)

      mse_de = df_de["mse"].iloc[-1]
      bias_de = df_de["bias"].iloc[-1]

      # OSS Estimator
      oss_bandit = OSSBandit(arms, ALPHA, probs)
      results_oss = simulate(oss_bandit,iterations=ITERATIONS,trials=TRIALS)
      df_oss = format_results(results_oss)

      mse_oss = df_oss["mse"].iloc[-1]
      bias_oss = df_oss["bias"].iloc[-1]

      # Concat to mse and bias vector
      alg_names = ["ME","DE","OSS"]
      mse_list.append([mse_me,mse_de,mse_oss])
      bias_list.append([bias_me,bias_de,bias_oss])

    #Plotting
    fig, ax = plt.subplots(2, len(N_ARMS), figsize=(16, 9))

    fig.suptitle(f"Internet Ads in a MAB Setting | {ITERATIONS} iterations | {TRIALS} trials \n \
                 {N_ARMS} arms | OSS alpha: {ALPHA}")

    for i in range(len(N_ARMS)):
       ax[0,i].bar(alg_names,mse_list[i])
       ax[0,i].set_title(f"RMSE {N_ARMS[i]} arms")
       ax[0,i].axhline(y=0.0, color='r', linestyle='-')
       for alg, val in zip(alg_names,mse_list[i]):
          ax[0,i].text(alg, val, str(round(val,4)), color='black', ha="center")

    for i in range(len(N_ARMS)):
       ax[1,i].bar(alg_names,bias_list[i])
       ax[1,i].set_title("Bias")
       ax[1,i].axhline(y=0.0, color='r', linestyle='-')
       for alg, val in zip(alg_names,bias_list[i]):
          ax[1,i].text(alg, val, str(round(val,4)), color='black', ha="center")

   #  ax[0].bar(alg_names,mse)
   #  ax[0].set_title("Root Mean Square Error")
   #  for alg, val in zip(alg_names,mse):
   #     ax[0].text(alg, val, str(round(val,4)), color='black', ha="center")
              
   #  ax[1].bar(alg_names,bias)
   #  ax[1].set_title("Bias")
   #  for alg, bias in zip(alg_names,bias):
   #    ax[1].text(alg, bias, str(round(bias,4)), color='black', ha="center")
    plt.show()
