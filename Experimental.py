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


def OLS(Market, Stocks, names_stocks, covariance_type='nonrobust'):
    dict = {}
 
    try:
        Stocks.shape[0]
        X = np.column_stack((np.ones_like(Market), Market))
        for stock, name in zip(Stocks, names_stocks):
            Res1 = sm.OLS(stock, X).fit(cov_type=covariance_type)
            print(Res1.summary())
            dict[name] = [Res1.params[0], Res1.params[1], Res1.pvalues[1], Res1.rsquared]
        
    except:
        X = np.column_stack((np.ones_like(Stocks), Stocks))
        Res1 = sm.OLS(Stocks, X).fit()
        dict["Equally Weighted Portfolio"] = [Res1.params[0], Res1.params[1], Res1.pvalues[1], Res1.rsquared]
        
    df = pd.DataFrame(dict, index=['alpha', 'beta', 'pvalue', 'r_squared'])
    return df.T


def fitted_values(Res):
    fit_t = []
    for e in Res:
        fit_t.append(e.fittedvalues)
    return fit_t


def reset_test(Market, Stocks, df):
    n = len(Stocks)
    Pvals_f = []
    X = np.column_stack((np.ones_like(Market), Market))
    for stock in Stocks:
        model = sm.OLS(stock, X).fit()
        fit = model.fittedvalues
        XR = np.column_stack((np.ones_like(Market), Market, np.power(fit,2),np.power(fit,3)))
        model2 = sm.OLS(stock, XR).fit()
        
        RSSR = model.ssr
        RSSU = model2.ssr
        Fstat=((RSSR-RSSU)/2)/(RSSU/n)
        Pval_f = 1-sp.stats.f.cdf(Fstat,2,n)
        Pvals_f.append(round(Pval_f,4))
    df['RESET_test'] = Pvals_f
    return df


def white_test(Market, Stocks, df):
    X = np.column_stack((np.ones_like(Market), Market))
    pvals_chi = []
    for stock in Stocks:
        result = sm.OLS(stock, X).fit()
        whitetest = sm.stats.diagnostic.het_white(result.resid, X)
        pvals_chi.append(round(whitetest[1], 4))
    df['White_test'] = pvals_chi
    return df


def breusch_godfrey_test(Market, Stocks, df):
    X = np.column_stack((np.ones_like(Market), Market))
    pvals_chi = [round(sm.stats.diagnostic.acorr_breusch_godfrey(sm.OLS(stock, X).fit(), nlags=3)[1], 4) for stock in Stocks]
    df['BG_test'] = pvals_chi
    return df


def durbin_watson_test(Market, Stocks, df):
    X = np.column_stack((np.ones_like(Market), Market))
    dw_values = []
    for stock in Stocks:
        model = sm.OLS(stock, X)
        results = model.fit()
        dw = sm.stats.stattools.durbin_watson(results.resid)
        dw_values.append(round(dw, 4))
    df['DW_Test'] = dw_values
    return df


def ols_rob_ste(Market, dict_Stocks, df_tests, folder_name, covariance_type):
    folder_path = os.path.join(os.getcwd(), folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
        
    X = np.column_stack((np.ones_like(Market), Market))
    stocks_e = df_tests.loc[df_tests['White_test'] <= 0.05].index
    
    for name in stocks_e:
        stock = dict_Stocks[name]
        
        result = sm.OLS(stock, X).fit()
        df = pd.DataFrame(result.summary().tables[1].data[1:],
                          columns=["Variable", "Coefficient", "Std. Error", "t-value", "P-value", "Lower CI", "Upper CI"])
        df = df[["Std. Error", "P-value"]]
    
        result_hac = result.get_robustcov_results(cov_type=covariance_type, maxlags=1)
        df_hac = pd.DataFrame(result_hac.summary().tables[1].data[1:],
                              columns=["Variable", "Coefficient", "Std. Error", "t-value", "P-value", "Lower CI", "Upper CI"])
        df_hac = df_hac[["Std. Error", "P-value"]]
        
        file_path = os.path.join(folder_path, f'{name}.xlsx')
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Non Robust')
            df_hac.to_excel(writer, sheet_name=covariance_type)
            

def chow_test(Market, Stocks, name_Stocks, folder_name):
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, folder_name)
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    X = np.column_stack((np.ones_like(Market), Market))
    n = len(Market)
    w = 36
    t = pd.date_range(start = '2015-11-01', end ='2023-10-31', freq ='M')
    i1 = w
    i2 = n-w
    Fstat = np.empty(n-w-w+1, dtype=float)
    Pval = np.empty(n-w-w+1, dtype=float)
    
    for stock, name in zip(Stocks,name_Stocks):
        for ii in range (i1,i2):
            X1 = np.column_stack((np.ones_like(Market[1:ii]), Market[1:ii]))
            X2 = np.column_stack((np.ones_like(Market [ii+1:n]), Market[ii+1:n]))
            
            Res1 = sm.OLS(stock[1:n], X[1:n]).fit()
            Res2a = sm.OLS(stock[1:ii], X1).fit() 
            Res2b = sm.OLS(stock[ii+1:n], X2).fit()
            
            RSSR = Res1.ssr
            RSSU = Res2a.ssr+Res2b.ssr
            Fstat[ii-w+1] = ((RSSR-RSSU)/2)/(RSSU/(n-4))
            Pval[ii-w+1] = 1-sp.stats.f.cdf(Fstat[ii-w+1],2,n-4)
            
            plt.plot(t,Pval,t,0.01*np.ones_like(Pval))
            plt.title(f'Chow test for {name} - moving break date')
            plt.savefig(os.path.join(folder_path, f'CHOWING OF {name}.png'), dpi=300)
            plt.close()

