#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install yfinance


# In[31]:


import pandas as pd
import yfinance as yf
import random
import math
import numpy as np


# In[59]:


#Read in the symbols for Russell 1000 Index
df = pd.read_html('https://en.wikipedia.org/wiki/Russell_1000_Index')


# In[60]:


#Read in the symbols for Russell 2000 Index
df2 = pd.read_html('https://bullishbears.com/russell-2000-stocks-list/')


# In[61]:


#Convert them to a list
tickers_russell_1000 = df[2]['Ticker'].to_list()
tickers_russell_2000 = df2[0]['Symbol'].to_list()


# In[1]:


#Method that gets the tickers recognized by yfinance for any Index
def valid_tickers(tickers):
    corrupted_symbols = []
    valid_symbols_old = []
    for x in tickers:
        try:
            df = yf.download(x, start='2013-01-01', end='2023-10-31')
            if df.empty:
                corrupted_symbols.append(x)
            else:
                valid_symbols_old.append(x)
        except KeyError:
            corrupted_symbols.append(x)
            
                  
    valid_symbols = list(set(valid_symbols_old))
    
    return valid_symbols


# In[63]:


#Get the valid tickers for the Russell 1000 Index
valid_ticker_1000 = valid_tickers(tickers_russell_1000)


# In[64]:


#Checks to see if any of the tickers are not of a valid format in yfinance and removes them
#Then gets the valid tickers for the Russell 2000 Index
tickers_russell_2000 = [x for x in tickers_russell_2000 if type(x) == str]
valid_ticker_2000 = valid_tickers(tickers_russell_2000)


# In[65]:


#Downloads the data for the list of tickers for that Index
#Returns open and close lists containing open and closing prices for the index where each element is the open/close price
#for a ticker
def download_data(lis_ticker):
    open_prices = []
    close_prices = []
    for x in lis_ticker:
        df = yf.download(x, start='2013-01-01', end='2023-10-31')
        open_prices.append(df['Open'])
        close_prices.append(df['Close'])
    return open_prices, close_prices


# In[66]:


#Get open/close prices for each ticker in Russell 1000 and 2000 indexes
open_price_1000, close_price_1000 = download_data(valid_ticker_1000)
open_price_2000, close_price_2000 = download_data(valid_ticker_2000)


# In[78]:


#Organize Russell 1000 and 2000 index tickers in: (ticker_name, open_price, close_price)
#Makes it easier to choose random tickers from each index
valid_final_1000 = []
valid_final_2000 = []

for i in range(len(valid_ticker_1000)):
    if len(open_price_1000[i]) == 2726:
        valid_final_1000.append((valid_ticker_1000[i], open_price_1000[i], close_price_1000[i]))

for i in range(len(valid_ticker_2000)):
    if len(open_price_2000[i]) == 2726:
        valid_final_2000.append((valid_ticker_2000[i], open_price_2000[i], close_price_2000[i]))


# In[104]:


#Choose 333 random tickers from Russell 1000 
#Checks to see if those tickers exist in Russell 2000 index and remove them so that we can get 666 unique tickers from
#Russell 2000 index
random.shuffle(valid_final_1000)
sample_russell_1000 = random.sample(valid_final_1000, 333)

for x in sample_russell_1000:
    for y in valid_final_2000:
        if (x[0] == y[0]):
            valid_final_2000.remove(y)

random.shuffle(valid_final_2000)
sample_russell_2000 = random.sample(valid_final_2000, 667)


# In[105]:


#Consolidate all 1000 random tickers in a list
sample_stocks = sample_russell_1000
sample_stocks.extend(sample_russell_2000)


# In[108]:


#Create sep open and close dataframes from the lists. Store the ticker names to change the column names
open_df = [x[1] for x in sample_stocks]
close_df = [x[2] for x in sample_stocks]
valid_symbols = [x[0] for x in sample_stocks]


# In[109]:


#Concat all the Series (each series represents open/close prices for each ticker)
open_df_final = pd.concat(open_df, axis=1)
close_df_final = pd.concat(close_df, axis=1)


# In[110]:


#Ignore Output
open_df_final


# In[111]:


#Rename the columns to the ticker names for easy recognition
open_df_final.columns = valid_symbols
close_df_final.columns = valid_symbols


# In[113]:


#Import the open/close prices dfs to csvs so that we can use these csvs for further analysis
open_df_final.to_csv('open.csv', index=False)
close_df_final.to_csv('close.csv', index=False)


# In[ ]:




