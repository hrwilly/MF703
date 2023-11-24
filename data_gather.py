#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:59:32 2023

@author: liyihan
"""

import yfinance as yf
import pandas as pd
import os
from tqdm import tqdm


def get_tickers(file_path, list_time_max):
    data = pd.read_csv(file_path)
    filtered_data = data[(data['IPO Year'] <= list_time_max) | (data['IPO Year'].isna())]
    print(filtered_data)
    symbol_list = filtered_data['Symbol'].tolist()
    
    return symbol_list
    
folder_path = '/All stocks in US market/'
file_name_list = ['NASDAQ.csv', 'NYSE.csv', 'AMEX.csv']
symbols = []
for file in file_name_list:
    symbol_list = get_tickers(file, 2022)
    print(len(symbol_list))
    symbols += symbol_list
print(len(symbols))


def getHistory(symbols, period, start_date, end_date, save_path=None):
    Close = pd.DataFrame()
    Open = pd.DataFrame()

    for ticker in tqdm(symbols):
        ticker_object = yf.Ticker(ticker)
        ticker_historical = ticker_object.history(period=period, start=start_date, end=end_date)
        Close = pd.concat([Close, ticker_historical['Close']], axis=1)
        Open = pd.concat([Open, ticker_historical['Open']], axis=1)
    
    Close.columns = symbols
    Open.columns = symbols

    if save_path:
        Close.to_csv(os.path.join(save_path, 'Close.csv'))
        Open.to_csv(os.path.join(save_path, 'Open.csv'))
        

    return Close, Open



getHistory(symbols, period = '1d', start_date = '2011-01-01', end_date = '2023-11-22', save_path = '/original data')
Close = pd.read_csv('/original data/Close.csv', index_col=0)
Close.index = pd.to_datetime(Close.index)
close_non_null_counts = Close.count()
close_threshold = 0.75 * len(Close)
close_data = Close.loc[:, close_non_null_counts >= close_threshold]
close_data.index = pd.to_datetime(close_data.index, utc=True).tz_localize(None).date
month_counts = close_data.index.to_period("M").value_counts()
under_15_months = month_counts[month_counts < 15].index
close_data = close_data[~close_data.index.to_period("M").isin(under_15_months)]
close_data.to_csv('/Users/liyihan/Documents/GitHub/MF703/filtered_data/close_data.csv')


Open = pd.read_csv('/original data/Open.csv', index_col=0)
Open.index = pd.to_datetime(Open.index)
open_non_null_counts = Open.count()
open_threshold = 0.75 * len(Open)
open_data = Open.loc[:, open_non_null_counts >= open_threshold]
open_data.index = pd.to_datetime(open_data.index, utc=True).tz_localize(None).date
open_data = open_data[~open_data.index.to_period("M").isin(under_15_months)]
open_data.to_csv('/filtered_data/open_data.csv')









