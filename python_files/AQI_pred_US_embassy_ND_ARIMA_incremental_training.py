#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing all the required packages
import pandas as pd
import numpy as np
from matplotlib.pyplot import rcParams
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


#Reading the csv file 
df = pd.read_csv('new-delhi us embassy, india-air-quality.csv', parse_dates=True)

#Converting date column from string to datetime type.
df['date'] = pd.to_datetime(df['date'])

#Setting 
df = df.set_index('date')
df.head(2)


# In[2]:


df.isnull().sum()


# In[5]:


# Sort the data according to the date
df = df.sort_values(by=['date'])


# In[7]:


#Type of elements of dataframe
df.dtypes


# In[8]:


df1 = df[' pm25']


# In[9]:


df1


# In[10]:


#Checking the max value before removing the outliers from the given data
max1 = df1.max()
print('Max value before removing outlier is: ',max1)


# In[11]:


# Import libraries
import matplotlib.pyplot as plt
 
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(df1)
 
# show plot
plt.show()


# In[12]:


#Removing the outliers

Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 


# In[13]:


print(Q1,Q3)


# In[14]:


IQR


# In[15]:


filter = (df1 >= Q1 - 1.5 * IQR) & (df1 <= Q3 + 1.5 *IQR)
newdf = df1.loc[filter] 


# In[16]:


newdf


# In[17]:


#Checking the max value before removing the outliers from the given data
newdf_max = newdf.max()
print('Max value before removing outlier is: ',newdf_max)


# In[18]:



fig = plt.figure(figsize =(10, 7))

# Creating plot
plt.boxplot(newdf)

# show plot
plt.show()


# In[19]:


#Replacing all the outliers with maximum value
final=[]
for value in df1:
    if value>newdf_max:
        final.append(newdf_max)
    else:
        final.append(value)


# In[20]:


final_df = pd.DataFrame(final,columns=['PM2.5'])


# In[21]:


final_df


# In[22]:



 
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(final_df)
 
# show plot
plt.show()


# In[23]:


final_df.describe()


# In[24]:


# Write to csv file
# df.to_csv('norm_NDEmbassy.csv')


# In[25]:


final_df.plot(figsize=(10,8))


# In[26]:


final_df


# In[28]:


final_df.isnull().sum()


# In[31]:


#Checking whether the given data is stationary or not.
# If p-value is less than 0.05, then the data is stationary else non-stationary.
# For stationary data, ARIMA model does not depends upon date. It only depends upon previous values.
from statsmodels.tsa.stattools import adfuller
def adf_test(dataset):
     dftest = adfuller(dataset, autolag = 'AIC')
     print("1. ADF : ",dftest[0])
     print("2. P-Value : ", dftest[1])
     print("3. Num Of Lags : ", dftest[2])
     print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
     print("5. Critical Values :")
     for key, val in dftest[4].items():
         print("\t",key, ": ", val)
adf_test(final_df)


# In[32]:


#Plotting autocorrelation plot to check the p value
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
fig = plt.figure(figsize =(10, 7))
autocorrelation_plot(df)
pyplot.show()


# In[29]:


# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
# load dataset

#ARIMA model
X = final_df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(2,0,2))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


# In[33]:


# RMSE
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)


# In[38]:


# plot forecasts against actual outcomes
import matplotlib.pyplot as plt
fig = plt.figure(figsize =(15, 7))
plt.plot(test,label='Test Data')
plt.plot(predictions, color='red', label='Predicted data')
# pyplot.show()
plt.legend()


# In[40]:


#MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, predictions)
print('MAE: %f' % mae)


# In[42]:


#Saving the model for further use
model_fit.save('AQI_pred_US_Embassy_ND_ARIMA_incremental.h5')


# In[ ]:





# In[ ]:




