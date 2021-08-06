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


# In[6]:


df


# In[7]:


#Type of elements of dataframe
df.dtypes


# In[8]:


# Import libraries
import matplotlib.pyplot as plt

 
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(df[' pm25'])
 
# show plot
plt.show()


# In[9]:


Q1 = df[' pm25'].quantile(0.25)
Q3 = df[' pm25'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

# filter = (df[''] >= Q1 - 1.5 * IQR) & (df['AVG'] <= Q3 + 1.5 *IQR)
# df.loc[filter] 


# In[10]:


print(Q1,Q3)


# In[11]:


IQR


# In[12]:


filter = (df[' pm25'] >= Q1 - 1.5 * IQR) & (df[' pm25'] <= Q3 + 1.5 *IQR)
newdf = df.loc[filter] 


# In[13]:



 
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(newdf[' pm25'])
 
# show plot
plt.show()


# In[15]:


newdf.describe()


# In[17]:


newdf[' pm25'].plot(figsize=(10,8))


# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


# Checking the autocorrelation plot for checking of lag value
from pandas.plotting import autocorrelation_plot

rcParams['figure.figsize'] = [13,6]
autocorrelation_plot(newdf[' pm25'])
plt.show()


# # ARIMA

# In[25]:


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
adf_test(newdf[' pm25'])


# In[26]:


#Dividing the data into train and test dataset with 70 to 30 ratio.
X = newdf[' pm25']
train = X[:int(0.7*len(X))]
test = X[int(0.7*len(X)):]


# In[27]:


from pmdarima import auto_arima
stepwise_fit = auto_arima(train, trace=True, suppress_warnings=True)


# In[28]:


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(X,order=(2,0,2))
model=model.fit()
model.summary()


# In[29]:


start=len(train)
end=len(train)+len(test)-1
# start=
pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
plt.plot(pred,color='green',label='Prediction')
plt.plot(test, color='yellow',label='Original Test')
plt.legend()


# In[30]:


pred


# In[31]:


lis = pd.DataFrame(test)
lis['pred']=pred


# In[32]:


lis


# In[33]:


new_pred = model.predict(start=end,end=end+7,typ='levels')


# In[34]:


new_pred


# In[37]:


# evaluate forecasts
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(lis[' pm25'], lis['pred']))
print('Test RMSE: %.3f' % rmse)


# In[38]:


#MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(lis[' pm25'], lis['pred'])
print('MAE: %f' % mae)


# In[40]:


#Saving the model for further use
model.save('AQI_pred_US_Embassy_ND_ARIMA.h5')


# In[ ]:




