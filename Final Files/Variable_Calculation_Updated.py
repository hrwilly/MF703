#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:51:48 2023

@author: aadiljaved
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

close_data = pd.read_csv('filtered_data/close_data_sp500.csv', index_col=0)
close_data.index = pd.to_datetime(close_data.index)
close_data.index = close_data.index.to_period("M")
open_data = pd.read_csv('filtered_data/open_data_sp500.csv', index_col=0)
open_data.index = pd.to_datetime(open_data.index)

open_data.index = open_data.index.to_period("M")


daytime_return = (close_data / open_data) - 1
daytime_return = daytime_return[daytime_return.index >= '2013-01']


daily_return = (close_data / close_data.shift(1) - 1)
daily_return = daily_return[daily_return.index >= '2013-01']


overnight_return = ((1 + daily_return) / (1 + daytime_return)) - 1


monthly_close = close_data.groupby(close_data.index).last()
monthly_open = open_data.groupby(open_data.index).first()
monthly_return = (monthly_close / monthly_open) - 1

shifted_mon = monthly_return.shift(1)
ret_6 = shifted_mon.rolling(6, min_periods=1).sum()

ret_co = overnight_return.groupby(pd.Grouper(freq = 'M')).cumsum()
ret_co = ret_co.sort_index().resample("M").apply(lambda ser: ser.iloc[-1,])

ret_oc = daytime_return.groupby(pd.Grouper(freq = 'M')).cumsum()
ret_oc = ret_oc.sort_index().resample("M").apply(lambda ser: ser.iloc[-1,])

monthly_return.to_csv('variable_results/monthly_returns.csv')
ret_6.to_csv('variable_results/cumsum_6mo.csv')
ret_co.to_csv('variable_results/ret_co_m.csv')
ret_oc.to_csv('variable_results/ret_oc_m.csv')


pos_reversal = ((overnight_return < 0) & (daytime_return > 0)).astype(int)
PR_ratio = pos_reversal.groupby(pos_reversal.index).mean()

neg_reversal = ((overnight_return > 0) & (daytime_return < 0)).astype(int)
NR_ratio = neg_reversal.groupby(neg_reversal.index).mean()


AB_PR = PR_ratio.rolling(window=12).mean()
AB_PR = AB_PR[AB_PR.index >= '2014-01']
PR_ratio = PR_ratio[PR_ratio.index >= '2014-01']
AB_PR = PR_ratio * AB_PR 


AB_NR = NR_ratio.rolling(window=12).mean()
AB_NR = AB_NR[AB_NR.index >= '2014-01']
NR_ratio = NR_ratio[NR_ratio.index >= '2014-01']
AB_NR = NR_ratio * AB_NR


tot_assets = pd.read_csv('Final Variables/annual_tot_asset.csv', index_col = 0)
tot_assets.index = pd.to_datetime(tot_assets.index, format = '%Y')

for ticker in tot_assets:
    for i in range(len(tot_assets[ticker])):
        if tot_assets[ticker][i] == '--':
            tot_assets[ticker][i] = 0

shifted_assets = tot_assets.shift(1)

asset_growth = pd.DataFrame()

for ticker in tot_assets:
    asset = []
    for i in range(1, len(tot_assets[ticker])):
        if tot_assets[ticker][i] != 0:
            asset.append((int(float(shifted_assets[ticker][i])) - int(float(tot_assets[ticker][i]))) / int(float(tot_assets[ticker][i])))
        else:
            asset.append(0)
    asset_growth[ticker] = asset

asset_growth.index = tot_assets.index[1:]

volume = pd.read_csv('filtered_data/volume_data.csv', index_col = 0)
volume.index = pd.to_datetime(volume.index)
volume = volume.sort_index()

returns = daily_return
returns = returns.to_timestamp()
returns = returns.reset_index()
D = returns.groupby(pd.Grouper(key='index', freq='1M')).count()

close_data = pd.read_csv('filtered_data/close_data_sp500.csv', index_col=0)
close_data.index = pd.to_datetime(close_data.index)
 
vold = volume*close_data

div = close_data / vold
div = div.reset_index()
summ = div.groupby(pd.Grouper(key='index', freq='1M')).sum()
illiquidity = summ / D

daytime_return.to_csv('variable_results/sp500/daytime_returns.csv')
daily_return.to_csv('variable_results/sp500/daily_returns.csv')
monthly_return.to_csv('variable_results/sp500/monthly_returns.csv')
overnight_return.to_csv('variable_results/sp500/overnight_returns.csv')
PR_ratio.to_csv('variable_results/sp500/positive_ratio.csv')
NR_ratio.to_csv('variable_results/sp500/negative_ratio.csv')
AB_PR.to_csv('variable_results/sp500/abnormal_positive_ratio.csv')
AB_NR.to_csv('variable_results/sp500/abnormal_negative_ratio.csv')
asset_growth.to_csv('variable_results/sp500/asset_growth')
illiquidity.to_csv('variable_results/sp500/illiquidity.csv')