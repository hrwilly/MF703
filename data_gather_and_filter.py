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
import yahoo_fin.stock_info as si



symbols = si.tickers_sp500()

def getHistory(symbols, period, start_date, end_date, save_path=None):
    Close = pd.DataFrame()
    Open = pd.DataFrame()
    Volume =pd.DataFrame()

    for ticker in tqdm(symbols):
        ticker_object = yf.Ticker(ticker)
        ticker_historical = ticker_object.history(period=period, start=start_date, end=end_date)
        Close = pd.concat([Close, ticker_historical['Close']], axis=1)
        Open = pd.concat([Open, ticker_historical['Open']], axis=1)
        Volume = pd.concat([Volume, ticker_historical['Volume']], axis=1)
    
    Close.columns = symbols
    Open.columns = symbols
    Volume.columns = symbols

    if save_path:
        Close.to_csv(os.path.join(save_path, 'Close_sp500.csv'))
        Open.to_csv(os.path.join(save_path, 'Open_sp500.csv'))
        Volume.to_csv(os.path.join(save_path, 'Volume_sp500.csv'))
        

    return Close, Open



getHistory(symbols, period = '1d', start_date = '2012-12-01', end_date = '2023-11-28', save_path = 'original data')
Close = pd.read_csv('original data/Close_sp500.csv', index_col=0)
Close.index = pd.to_datetime(Close.index)
close_non_null_counts = Close.count()
close_threshold = 0.75 * len(Close)
close_data = Close.loc[:, close_non_null_counts >= close_threshold]
close_data.index = pd.to_datetime(close_data.index, utc=True).tz_localize(None)
close_data.index = close_data.index.to_period("M")
month_counts = close_data.index.value_counts()
under_15_months = month_counts[month_counts < 15].index
close_data = close_data[~close_data.index.isin(under_15_months)]
close_data.to_csv('filtered_data/close_data_sp500.csv')


Open = pd.read_csv('original data/Open_sp500.csv', index_col=0)
Open.index = pd.to_datetime(Open.index)
open_non_null_counts = Open.count()
open_threshold = 0.75 * len(Open)
open_data = Open.loc[:, open_non_null_counts >= open_threshold]
open_data.index = pd.to_datetime(open_data.index, utc=True).tz_localize(None)
open_data.index = open_data.index.to_period("M")
open_data = open_data[~open_data.index.isin(under_15_months)]
open_data.to_csv('filtered_data/open_data_sp500.csv')

Volume = pd.read_csv('original data/Volume_sp500.csv', index_col=0)
Volume.index = pd.to_datetime(Volume.index)
Volume_non_null_counts = Volume.count()
Volume_threshold = 0.75 * len(Volume)
Volume_data = Volume.loc[:, Volume_non_null_counts >= Volume_threshold]
Volume_data.index = pd.to_datetime(Volume_data.index, utc=True).tz_localize(None)
Volume_data.index = Volume_data.index.to_period("M")
Volume_data = Volume_data[~Volume_data.index.isin(under_15_months)]
Volume_data.to_csv('filtered_data/volume_data_sp500.csv')








