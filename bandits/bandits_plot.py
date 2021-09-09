import pandas as pd
import numpy as np 
import os 
import math
import shutil

import matplotlib.pyplot as plt

def format_results(results):
   df = pd.DataFrame(list(zip(*results)), columns=["trial","iteration","chosen_arm","reward","cum_reward","max_mu_estimate","mse"])
   return df

def 