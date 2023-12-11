import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import os
import statsmodels.api as sm
from xlsxwriter import Workbook

def create_folder(folder_name):
    cwd = os.getcwd()
    folder_path = os.path.join(cwd, folder_name)
    
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return folder_path

def residuals(result):
    residuals = []
    for e in result:
        residuals.append(e.resid)
    return residuals

def fitted_values(result):
    fit_t = []
    for e in result:
        fit_t.append(e.fittedvalues)
    return fit_t

def scatterplot(Market, Stocks, title, xlabel, ylabel, how_to_save, folder_name):
    folder_path = create_folder(folder_name)
        
    ylabel = iter(ylabel.columns)
    
    for stock in Stocks:  
        ylabel_i = next(ylabel)  
        plt.figure()
        plt.scatter(Market, stock, label = title)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel_i)
        plt.savefig(folder_path + '/' + how_to_save + ylabel_i + '.png')
        plt.close()
  
  
def ols_sum(Market, Stocks):
    X = np.column_stack((np.ones_like(Market), Market))
    results = []
    for stock in Stocks:
        result = sm.OLS(stock, X).fit()
        results.append(result)
    return results

    
def ols_rob_ste(Market, dict_Stocks, df_tests, folder_name, covariance_type):
    
    folder_path = create_folder(folder_name)
        
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


def OLS(Market, Stocks, names_stocks, covariance_type='nonrobust'):
    
    dict = {}
 
    try:
        Stocks.shape[0]
        X = np.column_stack((np.ones_like(Market), Market))
        for stock, name in zip(Stocks, names_stocks):
            Res1 = sm.OLS(stock, X).fit(cov_type=covariance_type)
            dict[name] = [round(Res1.params[0],4), round(Res1.params[1],4), 
                          round(Res1.pvalues[1],4), round(Res1.rsquared,4)]
        
    except:
        X = np.column_stack((np.ones_like(Market), Market))
        Res1 = sm.OLS(Stocks, X).fit()
        dict["Equally Weighted Portfolio"] = [Res1.params[0], Res1.params[1], Res1.pvalues[1], Res1.rsquared]
        
    df = pd.DataFrame(dict, index=['alpha', 'beta', 'pvalue', 'r_squared'])
    return df.T


def reset_test(Market, Stocks, df):
    
    n = len(Stocks)
    Pvals_f = []
    X = np.column_stack((np.ones_like(Market), Market))
    
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        fit = Res1.fittedvalues
        XR = np.column_stack((np.ones_like(Market), Market, np.power(fit, 2), np.power(fit, 3)))
        Res2 = sm.OLS(stock, XR).fit()
        
        RSSR = Res1.ssr
        RSSU = Res2.ssr
        Fstat = ((RSSR - RSSU) / 2) / (RSSU / n)
        Pval_f = 1 - sp.stats.f.cdf(Fstat, 2, n)
        Pvals_f.append(round(Pval_f, 4))
    
    df['RESET_test'] = Pvals_f
    return df

def white_test(Market, Stocks, df):
    
    X = np.column_stack((np.ones_like(Market), Market))
    pvals_chi = []
    for stock in Stocks:
        Res1 = sm.OLS(stock, X).fit()
        whitetest = sm.stats.diagnostic.het_white(Res1.resid,X)
        pvals_chi.append(round(whitetest[1],4))
    df['White_test'] = pvals_chi
    return df


def breusch_godfrey_test(Market, Stocks, df):

    X = np.column_stack((np.ones_like(Market), Market))
    pvals_chi = []
    for stock in Stocks:
        model = sm.OLS(stock, X).fit()
        bgtest=sm.stats.diagnostic.acorr_breusch_godfrey(model,nlags=3)
        pvals_chi.append(round(bgtest[1],4))
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
    
    
def chow_test(Market, Stocks, name_Stocks, folder_name):
    folder_path = create_folder(folder_name)
    
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


def multifactor_model_4(Market, Stocks, factor_1, factor_2, factor_3, factor_4, factor_5, name_stocks):
    '''
    Return: df with alphas, betas, r_squared, pvalue
            results of OLS
    '''
    n = len(Market)
    results = []
    dict = {}
    X = np.column_stack((np.ones_like(Market), Market, factor_1, factor_2, factor_3, factor_4, factor_5))
    
    for stock,name in zip(Stocks, name_stocks):
        result = sm.OLS(stock,X).fit()
        results.append(result)
        X = np.column_stack((np.ones_like(Market), Market, factor_3, factor_4, factor_5))
        Res2 = sm.OLS(stock,X).fit()
        dict[name] = [round(Res2.params[0],4), round(Res2.params[1],4), 
                          round(Res2.pvalues[1],4), round(Res2.rsquared,4)]
        RSSU = result.ssr
        RSSR = Res2.ssr
        Fstat = ((RSSR-RSSU)/4)/(RSSU/(n))
        Pval = 1-sp.stats.f.cdf(Fstat,4,n)
        print(Pval)
    df = pd.DataFrame(dict, index=['alpha', 'beta', 'pvalue', 'r_squared'])
    return results, df.T

def subplots(results_ols, folder_name, name_stocks):
    folder_path = create_folder(folder_name)
    fig, ax = plt.subplots(3,1, figsize=(8,12))
    ax[0].plot([result.params[0] for result in results_ols])
    ax[0].set_xticks(range(len(name_stocks)))
    ax[0].set_xticklabels(name_stocks, rotation=45, ha='right')
    ax[0].set_title('Alpha')
    ax[1].plot([result.params[1] for result in results_ols])
    ax[1].set_xticks(range(len(name_stocks)))
    ax[1].set_xticklabels(name_stocks, rotation=45, ha='right')
    ax[1].set_title('Beta')
    ax[2].plot([result.rsquared for result in results_ols])
    ax[2].set_xticks(range(len(name_stocks)))
    ax[2].set_xticklabels(name_stocks, rotation=45, ha='right')
    ax[2].set_title('R-squared')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f'Parameters for each bank.png'), dpi=300)
    plt.close()


def hist_residuals(results_CAPM, results_multifactor, name_stocks, folder_name):
    
    folder_path = create_folder(folder_name)
    
    residuals_CAPM = residuals(results_CAPM)
    residuals_multifactor = residuals(results_multifactor)
    
    for i,name in zip(range(len(residuals_CAPM)),name_stocks):
        
        fig, ax = plt.subplots()
        ax.hist(residuals_CAPM[i], alpha=0.7, bins=25, label='CAPM')
        ax.hist(residuals_multifactor[i], alpha=0.7, bins=25, label='Multifactor')
        ax.set_xlabel('Value of residual')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} - Residuals')
        ax.legend()
        plt.savefig(os.path.join(folder_path, f'{name}.png'), dpi=300) 
        plt.close()
             

def correlation_residuals(results_CAPM, results_multifactor, name_stocks, folder_name):
    folder_path = create_folder(folder_name)
    
    t = pd.date_range(start = '2009-12-01', end ='2023-10-31', freq ='M')
    
    residuals_CAPM = pd.DataFrame(residuals(results_CAPM), index=name_stocks, columns=t)
    residuals_multifactor = pd.DataFrame(residuals(results_multifactor), index=name_stocks, columns=t)

    if len(residuals_CAPM) != len(residuals_multifactor):
        raise ValueError("Lengths of residuals should be equal.")
    
    correlations = []
    
    for i, name in zip(range(len(residuals_CAPM)), name_stocks):
        correlation, _ = sp.stats.pearsonr(residuals_CAPM.iloc[i], residuals_multifactor.iloc[i])
        correlations.append(correlation)
        
        fig, ax = plt.subplots()
        ax.plot(t, residuals_CAPM.loc[name], label='CAPM', color='blue', alpha=0.8)
        ax.plot(t, residuals_multifactor.loc[name], label='Multifactor', color='red', alpha=0.7)
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Value Residual')
        ax.set_title(f'{name} with Market and 4 factor')
        ax.legend()
        plt.savefig(os.path.join(folder_path, f'{name}.png'), dpi=300)
        plt.close()
    
    print(correlations)
    
    
def rolling_capm(Market, Stocks, name_stocks, folder_name):
    
    folder_path = create_folder(folder_name)
        
    window_size = 60
    
    t = pd.date_range(start = '2014-11-01', end ='2023-10-31', freq ='M')
    
    for stock, name in zip(Stocks, name_stocks):
        alphas = []
        betas = []
        r_squareds = []
        conf_alpha_up = []
        conf_alpha_low = []
        conf_beta_up = []
        conf_beta_low = []
        
        for i in range(len(stock) - window_size + 1):
            window_returns_stock = pd.DataFrame(stock[i:i+window_size], columns=['Returns'])
            window_returns_mkt = pd.DataFrame(Market[i:i+window_size], columns=['Market Returns'])
            X = sm.add_constant(window_returns_mkt)
            model = sm.OLS(window_returns_stock, X)
            results_model = model.fit()
            confidence = results_model.conf_int()
            
            alpha = results_model.params.iloc[0]
            beta = results_model.params.iloc[1]
            r_squared = results_model.rsquared

            alphas.append(alpha)
            betas.append(beta)
            r_squareds.append(r_squared)
            conf_alpha_up.append(confidence.iloc[0,1])
            conf_beta_up.append(confidence.iloc[1,1])
            conf_alpha_low.append(confidence.iloc[0,0])
            conf_beta_low.append(confidence.iloc[1,0])
            
        fig, ax = plt.subplots()
        ax.plot(t, conf_alpha_up, label='Upper')
        ax.plot(t, conf_alpha_low, label='Lower')
        ax.plot(t, alphas, label='Alpha')
        ax.set_xlabel('Window Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Alpha - {name}')
        ax.legend()
        plt.savefig(os.path.join(folder_path, f'Alpha - {name}.png'), dpi=300)
        plt.close()
        
        fig1, bx = plt.subplots()
        bx.plot(t, conf_beta_up, label='Upper')
        bx.plot(t, conf_beta_low, label='Lower')
        bx.plot(t, betas, label=f'Beta')
        bx.set_xlabel('Window Index')
        bx.set_ylabel('Value')
        bx.set_title(f'Beta - {name}')
        bx.legend()
        plt.savefig(os.path.join(folder_path, f'Beta - {name}.png'), dpi=300)
        plt.close()
        
        plt.plot(t, r_squareds)
        plt.xlabel('Window Index')
        plt.ylabel('Value')
        plt.title(f'R squared - {name}')
        plt.savefig(os.path.join(folder_path, f'R squared - {name}.png'), dpi=300)
        plt.close()
        