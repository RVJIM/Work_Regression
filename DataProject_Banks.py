import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp



#Define date - time index monthly
t=pd.date_range(start ='01-11-2009', end ='31-10-2023', freq ='M')

#Import data from Excel File
StoxxEuro = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="STOXXEURO600")
Euribor_3M = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="EURIBOR3M_DEL")
Constituents = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="Constituents")

#select data
MKT = StoxxEuro[['STOXX EUROPE 600 E - TOT RETURN IND']]
BANKS = Constituents.iloc[:, 1:]
RFREE = Euribor_3M[['EBF EURIBOR 3M DELAYED - OFFERED RATE']]

# compute returns in percentages
rBanks = np.array(100 * (np.log(BANKS) - np.log(BANKS.shift(1)))) 
rMkt = np.array(100 * (np.log(MKT) - np.log(MKT.shift(1))))
rF = np.array(RFREE/12)


# compute excess returns 
erBanks = np.subtract(rBanks, rF)
erMkt = np.subtract(rMkt, rF)

n = np.size(BANKS)

# plotting banks vs mkt index
# plt.plot(t ,rMkt[1: n],t , rF[1: n])

for banks in BANKS:
    plt.scatter(erMkt, erBanks[:,banks], label = f'{banks} vs STOXXEURO 600')
    plt.xlabel('STOXXEURO 600- Monthly - 01/11/2009 - 31/10/2023')
    plt.ylabel(f'{banks}')
    plt.savefig('rMKTrF_monthly.png', dpi = 300)
