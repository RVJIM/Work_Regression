import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import statsmodels.api as sm
import Experimental_f as ex

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

'''df_quants = ex.OLS(erMkt, erBanks, BANKS.columns)
print(df_quants)

# Create DataFrame for tests
df_tests = pd.DataFrame(index = BANKS.columns)

# Compute: RESET, White, Breusch-Godfrey and Durbin-Watson tests
# And insert them in previously created DataFrame 
df_tests = ex.reset_test(erMkt, erBanks, df_tests)
df_tests = ex.white_test(erMkt, erBanks, df_tests)                         
df_tests = ex.breusch_godfrey_test(erMkt, erBanks, df_tests)
df_tests = ex.durbin_watson_test(erMkt, erBanks, df_tests)
#df_tests.to_latex('Tests.lex')
#print(df_tests)

#ols with robust standard errors
ex.ols_rob_ste(erMkt, d_banks_ex, df_tests, 'Robust Standar Error', 'HAC')'''

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

results_multifactor = ex.multifactor_model_4(erBanks, Mkt_RF, SMB, HML, RMW, CMA)

results_CAPM = ex.ols_sum(erMkt, erBanks)

corr = ex.correlation_residuals(results_CAPM, results_multifactor, BANKS.columns, 
                                'Residuals with Market and 4 factor of Fama-French')
print(corr)
#ex.chow_test(erMkt, erBanks, BANKS.columns, 'Chow_Testing')
#ex.rolling_capm(erMkt, erBanks, BANKS.columns, 'Rolling')