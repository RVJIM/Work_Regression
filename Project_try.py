import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp
import statsmodels.api as sm
import Experimental as ex

# Define date - time index monthly
t = pd.date_range(start ='01-11-2009', end ='31-10-2023', freq ='M')

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
print(rBanks)
# Compute excess returns 
erBanks = np.subtract(rBanks, rF)[1:].T
erMkt = np.subtract(rMkt, rF)[1:]

''''ex.scatterplot(erMkt,erBanks, "Excess Return", "STOXXEURO 600", BANKS,
               'STOXXEURO vs ','Equity_vs_Mkt')'''

'''df_quants = ex.OLS(erMkt, erBanks, BANKS.columns)
df_quants.to_latex('Quantities.lex')

average_erBanks = (sum(erBanks)/len(erBanks)).T

df_av_quants = ex.OLS(erMkt, average_erBanks, BANKS.columns)
df_av_quants.to_latex('Quantities_Weighted_Portfolio.lex')'''

# Create DataFrame for tests
'''df_tests = pd.DataFrame(index = BANKS.columns)

# Compute: RESET, WHite, Breusch-GOdfrey and Durbin-Watson tests
# And insert them in previously created DataFrame 
df_tests = ex.RESET_test(erMkt, erBanks, df_tests)
df_tests = ex.White_test(erMkt, erBanks, df_tests)                         
df_tests = ex.Breusch_Godfrey_test(erMkt, erBanks, df_tests)
df_tests = ex.Durbin_Watson_test(erMkt, erBanks, df_tests)
#df_tests.to_latex('Tests.lex')
#print(df_tests)'''

#Fama-French
'''FF_data = pd.read_csv('Europe_5_Factors.csv', skiprows=3, nrows=400, index_col=0)
FF_data.index = pd.to_datetime(FF_data.index, format='%Y%m')
FF_data.index = FF_data.index + pd.offsets.MonthEnd()
start_date = pd.to_datetime('2009-11-01')
end_date = pd.to_datetime('2023-10-31')
FF_data = FF_data.loc[start_date:end_date]
print(FF_data)'''


# Explanatory Variables - Unemployment, Real Estate Index, GDP, EUR/USD