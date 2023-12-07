import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import statsmodels.api as sm
import Functions as fun

# Define date - time index monthly
t=pd.date_range(start ='01-11-2009', end ='31-10-2023', freq ='M')

# Import data from Excel File
StoxxEuro = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="STOXXEURO600")
Euribor_3M = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="EURIBOR3M_DEL")
Constituents = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="Constituents")

# Select data
MKT = StoxxEuro[['STOXX EUROPE 600 E - TOT RETURN IND']]
BANKS = Constituents.iloc[:, 1:]
RFREE = Euribor_3M[['EBF EURIBOR 3M DELAYED - OFFERED RATE']]

# Clean names on Market and Banks
MKT.columns = MKT.columns.str.replace(' E - TOT RETURN IND','')
BANKS.columns = BANKS.columns.str.replace(' - TOT RETURN IND','')

# Compute returns in percentages
rBanks = np.array(100 * (np.log(BANKS) - np.log(BANKS.shift(1)))) 
rMkt = np.array(100 * (np.log(MKT) - np.log(MKT.shift(1))))
rF = np.array(RFREE/12)

# Compute excess returns 
erBanks = np.subtract(rBanks, rF)[1:].T
erMkt = np.subtract(rMkt, rF)[1:]

# Dictionary of Banks: 1 - return in percentages, 2 - excess returns
d_banks_r = {BANKS.columns[i]: rBanks[i+1] for i in range(len(BANKS.columns))}
d_banks_ex = {BANKS.columns[i]: erBanks[i] for i in range(len(BANKS.columns))}

# Create scatterplot
fun.scatterplot(erMkt,erBanks, "Excess Return", "STOXXEURO 600", BANKS,
               'StoxxEuro vs ','Equity vs Mkt')

# DataFrame with all quantities we need (I think)
df_quants = fun.OLS(erMkt, erBanks, BANKS.columns)
print(df_quants)

''' 
    If you want to download .xlsx file, change 40 with this
    df_quants.to_excel('Quants.xlsx')
'''

# Average return accross the equities' excess return - returns of an equally weighted portfolio
average_erBanks = sum(erBanks)/len(erBanks)

# DataFrame with all quantities for equally weighted portfolio
df_av_quants = fun.OLS(erMkt, average_erBanks, BANKS.columns)
print(df_av_quants)

''' 
    If you want to download .xlsx file, change 51 with this
    df_av_quants.to_excel('Quants_Weighted_Portfolio.xlsx')
'''

# Create DataFrame for tests
df_tests = pd.DataFrame(index = BANKS.columns)

# Compute: RESET, WHite, Breusch-GOdfrey and Durbin-Watson tests
# And insert them in previously created DataFrame 
df_tests = fun.reset_test(erMkt, erBanks, df_tests)
df_tests = fun.white_test(erMkt, erBanks, df_tests)                         
df_tests = fun.breusch_godfrey_test(erMkt, erBanks, df_tests)
df_tests = fun.durbin_watson_test(erMkt, erBanks, df_tests)

''' 
    If you want to download .xlsx file, change 51 with this
    df_tests.to_excel('Tests.xlsx')
'''

# OLS with robust standard errors
fun.ols_rob_ste(erMkt, d_banks_ex, df_tests, 'Robust Standar Error', 'HAC')

# Fama-French
FF_data = pd.read_csv('Europe_5_Factors.csv', skiprows=3, nrows=400, index_col=0)
FF_data.index = pd.to_datetime(FF_data.index, format='%Y%m')
FF_data.index = FF_data.index + pd.offsets.MonthEnd()
start_date = pd.to_datetime('2009-12-01')
end_date = pd.to_datetime('2023-10-31')
FF_data = FF_data.loc[start_date:end_date]

Mkt_RF = FF_data['Mkt-RF']
SMB = FF_data['SMB']
HML = FF_data['HML']
RMW = FF_data['RMW']
CMA = FF_data['CMA']

results_multifactor = fun.multifactor_model_4(erMkt, erBanks, Mkt_RF, SMB, HML, RMW, CMA)

# In this case HML, RMW, CMA are relevant variables
# We have all pvalue = 1, except the Market

results_CAPM = fun.ols_sum(erMkt, erBanks)

corr = fun.correlation_residuals(results_CAPM, results_multifactor, BANKS.columns, 
                                'Residuals with Market and 4 factor of Fama-French')
print(corr)

# Compute and create lineplot for Chow Test - moving break dates 
#fun.chow_test(erMkt, erBanks, BANKS.columns, 'Chow_Testing')

# Re-estimate the CAPM model for very window of size 5 years
# by moving the estimation sample by one month at a time
fun.rolling_capm(erMkt, erBanks, BANKS.columns, 'Rolling')

