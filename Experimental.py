import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import os
import statsmodels.api as sm

def scatterplot(Market, Stocks, title, xlabel, ylabel, how_to_save, folder_name):
    cwd = os.getcwd()
    
    if not os.path.exists(folder_name):
        folder_name = os.mkdir(cwd + "/" + folder_name)
        
    ylabel = iter(ylabel.columns)
    
    for stock in Stocks:  
        ylabel_i = next(ylabel)  
        plt.figure()
        plt.scatter(Market, stock, label = title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_i)
        plt.savefig(folder_name + '/' + how_to_save + ylabel_i + '.png')
        plt.close()
  
def ols_sum(Market, Stocks):
    X = np.column_stack((np.ones_like(Market), Market))
    Res = []
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        Res.append(Res1.summary())
    return Res

def OLS(Market, Stocks, df_stocks):
    X = np.column_stack((np.ones_like(Market), Market))
    parameters = []
    dict = {}
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        alpha = Res1.params[0]
        beta = Res1.params[1]
        p_value = Res1.pvalues[1]
        r_squared = Res1.rsquared
        for name in df_stocks:
           dict[name] = [alpha, beta, p_value, r_squared]
    return dict



if __name__ == "__main__":
    "__main__"