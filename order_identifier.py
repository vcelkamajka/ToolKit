import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# to determine the order of a reaction

def n_order(x, m,n,c):
    return m*x**n + c

class ReactionOrder:
    def __init__(self,filename,df=None,x_vals=None,y_vals=None):
        self.df = pd.read_csv(filename)
        self.x_vals = self.df.iloc[:,0]
        self.y_vals = self.df.iloc[:,1]

        print(self.df)

        param, param_cov = curve_fit(n_order, self.x_vals, self.y_vals)
        print(param)
        print(param_cov)

        fitted_y = param[0] * self.x_vals ** param[1] + param[2]

        plt.scatter(self.x_vals,self.y_vals,label='Actual Data')
        plt.scatter(self.x_vals,fitted_y,label='Fitted Data')
        plt.legend()
        plt.xlabel(self.df.columns[0])
        plt.ylabel(self.df.columns[1])
        plt.show()


test = ReactionOrder('drug_release.csv')




