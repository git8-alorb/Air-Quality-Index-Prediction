#!/usr/bin/env python
# coding: utf-8

# In[88]:


#Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[89]:


# Ignoring warnings for better visualization
import warnings
warnings.filterwarnings('ignore')


# In[90]:


#Reading datasheet
df = pd.read_excel('dtu.xlsx',parse_dates=True)
df.head()


# In[91]:


# Converting date column from string to datetime format
df['From Date'] = pd.to_datetime(df['From Date'])


# In[92]:


#Printing top 5 rows of dataframe
df.head()


# In[93]:


# Replacing all None elements with "NaN" value
nan_value = float("NaN")
df.replace("None", nan_value, inplace=True)


# In[94]:


# Converting data type of all columns from object type to float64 type.
df['PM2.5'] = df['PM2.5'].apply(pd.to_numeric)
df['Temp'] = df['Temp'].apply(pd.to_numeric)
df['RH'] = df['RH'].apply(pd.to_numeric)
df['WS'] = df['WS'].apply(pd.to_numeric)
df.dtypes


# In[95]:


#Checking all the null values available in the dataset
df.isnull().sum()


# In[96]:


#Taking PM2.5 column for prediction from above dataframe.
newdf = df['PM2.5']
newdf.head()


# In[97]:


#Checking the null value
newdf.isnull().sum()


# In[98]:


# Filling all the null values with the adjacent previous values.
newdf.fillna(method='ffill',inplace=True)


# In[99]:


#Plotting the graph for visualization.
newdf.plot(figsize=(15,9))


# # Outliers removal using Box Plot

# In[100]:


newdf


# In[101]:


#Importing matplotlib and plotting box plot of new dataframe
import matplotlib.pyplot as plt
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(newdf)
 
# show plot
plt.show()


# In[102]:


#Checking the max value before removing the outliers from the given data
newdf_max = newdf.max()
print('Max value before removing outlier is: ',newdf_max)


# In[103]:


#remove outlier using IQR
Q1 = newdf.quantile(0.25)
Q3 = newdf.quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range.

filter = (newdf >= Q1 - 1.5 * IQR) & (newdf <= Q3 + 1.5 *IQR)
newdf_after_removing_outlier = newdf.loc[filter] 


# In[104]:


#New dataframe after removing all the outliers.
newdf_after_removing_outlier


# In[105]:


#Checking the max value after removing the outliers from the data
newdf_max_after_removing_outlier = newdf_after_removing_outlier.max()
print('Max value after removing outlier is: ',newdf_max_after_removing_outlier)


# In[106]:


#Replacing all the outliers with maximum value
final=[]
for value in newdf:
    if value>newdf_max_after_removing_outlier:
        final.append(newdf_max_after_removing_outlier)
    else:
        final.append(value)


# In[107]:


final_df = pd.DataFrame(final,columns=['PM2.5'])


# In[108]:


# Final dataframe after replacing all the outliers with maximum value and removing the null values.
final_df


# In[109]:


#Boxplot of final data
import matplotlib.pyplot as plt
fig = plt.figure(figsize =(10, 7))
 
# Creating plot
plt.boxplot(final_df)
 
# show plot
plt.show()


# # ARIMA Model

# In[110]:


#Plotting the new data
final_df.plot(figsize=(15,10))


# In[111]:


#Stats of final data
final_df.describe()


# In[112]:


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


# In[113]:


#Dividing the data into train and test dataset with 70 to 30 ratio.
X = final_df
train = X[:int(0.7*len(X))]
test = X[int(0.7*len(X)):]


# In[114]:


test


# In[115]:


# Finding optimal value of (p,d,q) for ARIMA model
from pmdarima import auto_arima
stepwise_fit = auto_arima(X, trace=True, suppress_warnings=True)


# In[116]:


#Training the ARIMA model on the final dataset
from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(X,order=(3,0,1))
model=model.fit()
model.summary()


# In[117]:


#Predicting the value for test data and comparing it with original availaible data.
start=len(train)
end=len(train)+len(test)-1
fig = plt.figure(figsize =(17, 8))
pred=model.predict(start=start,end=end,typ='levels')
plt.plot(pred,color='red',label='Prediction')
plt.plot(test, color='blue',label='Original Test')
plt.legend()


# In[119]:


lis['pred']=pred


# In[120]:


# Showing the original test data (PM2.5 column) and predicted data (pred column) for comparision.
lis


# In[121]:


#Prediction for next 7 days
new_pred = model.predict(start=end,end=end+7,typ='levels')


# In[122]:


new_pred


# In[129]:


#Saving the model
model.save('AQI_DTU_ARIMA_model.h5')


# In[85]:


df.tail()


# In[125]:


#MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, pred)
print('MAE: %f' % mae)


# In[128]:


# RMSE
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(test, pred))
print('Test RMSE: %.3f' % rmse)


# In[ ]:




