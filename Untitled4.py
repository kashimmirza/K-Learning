#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd 
from fbprophet import Prophet 
from fbprophet.plot import add_changepoints_to_plot

data = pd.read_csv('https://raw.githubusercontent.com/rahulhegde99/Time-Series-Analysis-and-Forecasting-of-Air-Passengers/master/airpassengers.csv') 
data.head()


# In[2]:



df = pd.DataFrame() 
df['ds'] = pd.to_datetime(data['Month']) 
df['y'] = data['#Passengers'] 
df.head()


# In[3]:



m = Prophet() 
m.fit(df) 

future = m.make_future_dataframe(periods=12 * 5, freq='M') 


# In[5]:



forecast = m.predict(future) 
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'trend_lower', 'trend_upper']].tail()

fig1 = m.plot(forecast) 


# In[6]:


fig2 = m.plot_components(forecast)


# In[7]:



fig = m.plot(forecast) 
a = add_changepoints_to_plot(fig.gca(), m, forecast)


# In[ ]:




