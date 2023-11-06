import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as sp

#Define date
t=pd.date_range(start ='01-11-2009', end ='31-10-2023', freq ='M')

#Import data from Excel File
StoxxEuro = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="STOXXEURO600")
Euribor_3M = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="EURIBOR3M_DEL")
StoxxEuro = pd.read_excel('Data_Banks_EuroStoxx600_Euribor.xlsx', sheet_name="STOXXEURO600")