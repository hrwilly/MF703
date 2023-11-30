#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:37:33 2023

@author: liyihan
"""

import pandas as pd
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


overnight_return = (1 + daily_return) / (1 + daytime_return) - 1


monthly_close = close_data.groupby(close_data.index).last()
monthly_open = open_data.groupby(open_data.index).first()
monthly_return = (monthly_close / monthly_open) - 1


pos_reversal = ((overnight_return > 0) & (daytime_return < 0)).astype(int)
PR_ratio = pos_reversal.groupby(pos_reversal.index).mean()

neg_reversal = ((overnight_return < 0) & (daytime_return > 0)).astype(int)
NR_ratio = neg_reversal.groupby(neg_reversal.index).mean()


AB_PR = PR_ratio.rolling(window=12).mean()
AB_PR = AB_PR[AB_PR.index >= '2014-01']
PR_ratio = PR_ratio[PR_ratio.index >= '2014-01']
AB_PR = AB_PR * PR_ratio


AB_NR = NR_ratio.rolling(window=12).mean()
AB_NR = AB_NR[AB_NR.index >= '2014-01']
NR_ratio = NR_ratio[NR_ratio.index >= '2014-01']
AB_NR = AB_NR * NR_ratio



daytime_return.to_csv('variable_results/sp500/daytime_returns.csv')
daily_return.to_csv('variable_results/sp500/daily_returns.csv')
monthly_return.to_csv('variable_results/sp500/monthly_returns.csv')
overnight_return.to_csv('variable_results/sp500/overnight_returns.csv')
PR_ratio.to_csv('variable_results/sp500/positive_ratio.csv')
NR_ratio.to_csv('variable_results/sp500/negative_ratio.csv')
AB_PR.to_csv('variable_results/sp500/abnormal_positive_ratio.csv')
AB_NR.to_csv('variable_results/sp500/abnormal_negative_ratio.csv')