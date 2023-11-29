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

# Compute returns in percentages
rBanks = np.array(100 * (np.log(BANKS) - np.log(BANKS.shift(1)))) 
rMkt = np.array(100 * (np.log(MKT) - np.log(MKT.shift(1))))
rF = np.array(RFREE/12)

# Clean names on Market and Banks
MKT.columns = MKT.columns.str.replace(' E - TOT RETURN IND','')
BANKS.columns = BANKS.columns.str.replace(' - TOT RETURN IND','')

# Compute excess returns 
erBanks = np.subtract(rBanks, rF)[1:].T
erMkt = np.subtract(rMkt, rF)[1:]

# Create scatterplot
ex.scatterplot(erMkt,erBanks, "Excess Return", "STOXXEURO 600", BANKS,
               'StoxxEuro vs ','Equity_vs_Mkt')

# DataFrame with all quantities we need (I think)
df_quants = ex.OLS(erMkt, erBanks, BANKS.columns)
print(df_quants)

''' 
    If you want to download .xlsx file, change 40 with this
    df_quants.to_excel('Quants.xlsx')
'''

# Average return accross the equities' excess return - returns of an equally weighted portfolio
average_erBanks = sum(erBanks)/len(erBanks)

# DataFrame with all quantities for equally weighted portfolio
df_av_quants = ex.OLS(erMkt, average_erBanks, BANKS.columns)
print(df_av_quants)

''' 
    If you want to download .xlsx file, change 51 with this
    df_av_quants.to_excel('Quants_Weighted_Portfolio.xlsx')
'''

# Create DataFrame for tests
df_tests = pd.DataFrame(index = BANKS.columns)

# Compute: RESET, WHite, Breusch-GOdfrey and Durbin-Watson tests
# And insert them in previously created DataFrame 
df_tests = ex.RESET_test(erMkt, erBanks, df_tests)
df_tests = ex.White_test(erMkt, erBanks, df_tests)                         
df_tests = ex.Breusch_Godfrey_test(erMkt, erBanks, df_tests)
df_tests = ex.Durbin_Watson_test(erMkt, erBanks, df_tests)

''' 
    If you want to download .xlsx file, change 51 with this
    df_tests.to_excel('Tests.xlsx')
'''

# Compute and create lineplot for Chow Test - moving break dates 
ex.Chow_Test(erMkt, erBanks, BANKS.columns, 'Chow_Testing')

# --- CORRELATION NOT BEING TOO HIGH --- BEFORE NEAR COLLINEARITY
