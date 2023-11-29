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
    dict = {}

    try:
        Stocks.shape[1]
        for stock,name in zip(Stocks,df_stocks):
            Res1 = sm.OLS(stock, X).fit()
            alpha = Res1.params[0]
            beta = Res1.params[1]
            p_value = Res1.pvalues[1]
            r_squared = Res1.rsquared
            dict[name] = [alpha, beta, p_value, r_squared]
             
    except:
        Res1 = sm.OLS(Stocks, X).fit()
        alpha = Res1.params[0]
        beta = Res1.params[1]
        p_value = Res1.pvalues[1]
        r_squared = Res1.rsquared
        dict["Equally Weighted Portfolio"] = [alpha, beta, p_value, r_squared]
        
    df = pd.DataFrame(dict)
    df.index = ['alpha','beta','pvalue','r_squared']
    return df

def fitted_values(Res):
    fit_t = []
    for e in Res:
        fit_t.append(e.fittedvalues)
    return fit_t

def RESET_test(Market, Stocks, df):
    n = len(Stocks)
    Pvals_f = []
    X = np.column_stack((np.ones_like(Market), Market))
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        fit = Res1.fittedvalues
        XR = np.column_stack((np.ones_like(Market), Market, np.power(fit,2),
                            np.power(fit,3)))
        Res2 = sm.OLS(stock, XR).fit()
        
        #Test
        RSSR = Res1.ssr
        RSSU = Res2.ssr
        Fstat=((RSSR-RSSU)/2)/(RSSU/n)
        Pval_f = 1-sp.stats.f.cdf(Fstat,2,n)
        Pvals_f.append(round(Pval_f,4))
    df['RESET_test'] = Pvals_f
    return df

def White_test(Market, Stocks, df):
    X = np.column_stack((np.ones_like(Market), Market))
    pvals_chi = []
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        whitetest = sm.stats.diagnostic.het_white(Res1.resid,X)
        pvals_chi.append(round(whitetest[1],4))
    df['White_test'] = pvals_chi
    return df

def Breusch_Godfrey_test(Market, Stocks, df):
    X = np.column_stack((np.ones_like(Market), Market))
    pvals_chi = []
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        bgtest=sm.stats.diagnostic.acorr_breusch_godfrey(Res1,nlags=3)
        pvals_chi.append(round(bgtest[1],4))
    df['BG_test'] = pvals_chi
    return df

def Durbin_Watson_test(Market, Stocks,df):
    X = np.column_stack((np.ones_like(Market), Market))
    value = []
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        dw = sm.stats.stattools.durbin_watson(Res1.resid)
        value.append(round(dw,4))
    df['DW_Test'] = value
    return df

def Chow_Test(Market, Stocks):
    n=np.size(Market)
    m=60
    X = np.column_stack((np.ones_like(Market), Market))
    for stock in Stocks:
        X1 = np.column_stack((np.ones_like(Market[1:m]), Market[1:m]))
        X2 = np.column_stack((np.ones_like(Market[m+1:n]), Market[m+1:n]))
        Res1 = sm.OLS(stock, X).fit()
        Res2A = sm.OLS(stock[1:m], X1).fit()
        Res2B = sm.OLS(stock[m+1:n], X2).fit()
        RSSU=Res2A.ssr+Res2B.ssr
        RSSR=Res1.ssr
        Fstat=((RSSR-RSSU)/2)/(RSSU/(n-4))
        Pval=1-sp.stats.f.cdf(Fstat,2,n-4)
        return Pval
    
        
        
if __name__ == "__main__":
    "__main__"