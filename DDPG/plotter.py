"""TEST RANGE: This thing is useless at the moment.

But soon this will be a performance plotter for the csv log files
I generate. 
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

class Plotter:
    def __init__(self, csv_log):

        # Import csv file into a pandas data frame
        self.data = pd.read_csv(csv_log)

        # Define column numbers and x axis to be the 0th dim
        self.n_cols = len(self.data.columns)
        self.x = self.data.iloc[:,0]
        
        print(f"Inititalized {self.n_cols - 1} plots...")
        
    def draw(self):
        fig, axes = plt.subplots(1,self.n_cols - 1)
        
        for plot in range(self.n_cols - 1):
            axes[0,plot] = self.data.plot(self.x, self.data.iloc[:, plot + 1])




x = np.array([1,2,3,4,5])
y = np.array([2,3.5,6,8,8.5])

fig, axes = plt.subplots(2,2)

axes[0,0].plot(x,y, "tab:green")
axes[0,0].plot(-x,y, "tab:red")
axes[0,1].plot(-x,y)
axes[1,0].plot(x,-y)
axes[1,1].plot(-x,-y)

plt.savefig("DDPG/test.png")

