import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima_model import ARIMA

#Loading Dataset
dataset = pd.read_csv('AXISBANK.csv')
#Dropping Null values
AXISBANKStockData = dataset.dropna()

#Re-indexing Dataset Table indes to be "Date" Column 
AXISBANKStockData.index = pd.to_datetime(AXISBANKStockData.Date)
# AXISBANKStockData_all= AXISBANKStockData["Prev Close"]



#Selecting "Close" column and Remaove others Columns
AXISBANKStockData_data = AXISBANKStockData["Close"]['2015-01-01':'2016-01-01']
AXISBANKStockData_data.describe()
#Plotting Dataset
plt.figure(figsize=(16,7))
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.set_xlabel('Time Frame')
ax1.set_ylabel('Stock Price for AXISBANK')
ax1.plot(AXISBANKStockData_data)




rolLmean = AXISBANKStockData_data.rolling(12).mean()
rolLstd = AXISBANKStockData_data.rolling(12).std()
plt.figure(figsize=(16,7))
fig = plt.figure(2)
#Plot rolling statistics:
orig = plt.plot(AXISBANKStockData_data, color='blue',label='Original')
mean = plt.plot(rolLmean, color='red', label='Rolling Mean')
std = plt.plot(rolLstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)



#Training ARIMA Model 
model = ARIMA(AXISBANKStockData_data, order=(2,0,2))  
results_ARIMA = model.fit()
print(results_ARIMA.summary())



# Plotting the Results
plt.figure(figsize=(16,8))
plt.plot(AXISBANKStockData_data)
plt.plot(results_ARIMA.fittedvalues, color='red')