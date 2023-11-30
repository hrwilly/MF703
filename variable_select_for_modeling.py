#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:35:33 2023

@author: liyihan
"""

import pandas as pd
import os

AB_PR = pd.read_csv('variable_results/sp500/abnormal_positive_ratio.csv', index_col=0)
AB_PR_list = AB_PR.columns

def filter_data(data_path, end_year):
    df = pd.read_excel(data_path, header=0, index_col=0)
    df = df.T
    df = df[df.index <= end_year].dropna()
    ticker_list = df.columns.tolist()
    return df, ticker_list

gross_profit = filter_data('filtered_data/spx gross profit.xlsx', 2022)[0]
gross_profit_list = filter_data('filtered_data/spx gross profit.xlsx', 2022)[1]

PB = filter_data('filtered_data/spx price_to_book.xlsx', 2022)[0]
PB_list = filter_data('filtered_data/spx price_to_book.xlsx', 2022)[1]

turnover = filter_data('filtered_data/spx share turnover.xlsx', 2022)[0]
turnover_list = filter_data('filtered_data/spx share turnover.xlsx', 2022)[1]

size = filter_data('filtered_data/spx size.xlsx', 2022)[0]
size_list = filter_data('filtered_data/spx size.xlsx', 2022)[1]

tot_asset = filter_data('filtered_data/spx total asset.xlsx', 2022)[0]
tot_asset_list = filter_data('filtered_data/spx total asset.xlsx', 2022)[1]

intersection_result = list(set(AB_PR_list).intersection(gross_profit_list, PB_list, turnover_list, size_list, tot_asset_list))

with open('Final Variables/ticker_selected.txt', 'w') as file:
    for element in intersection_result:
        file.write(str(element) + '\n')




folder_path = 'variable_results/sp500'
file_list = os.listdir(folder_path)

all_data = pd.DataFrame()

for file in file_list:
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, index_col=0)
        data.index = pd.to_datetime(data.index)
        data.index = data.index.to_period("M")
        data = data[ (data.index <= '2022-12') & (data.index >= '2014-01')]
        filtered_data = data.filter(intersection_result, axis=1)
        output_filename = f"monthly_{file}.csv"
        save_path = os.path.join('Final Variables', file)
        filtered_data.to_csv(save_path)
        
gross_profit = gross_profit.filter(intersection_result, axis=1)
filtered_data.to_csv('Final Variables/annual_gross_profit.csv')

PB = PB.filter(intersection_result, axis=1)
PB.to_csv('Final Variables/annual_PB.csv')

turnover = turnover.filter(intersection_result, axis=1)
turnover.to_csv('Final Variables/annual_turnover.csv')

size = size.filter(intersection_result, axis=1)
size.to_csv('Final Variables/annual_size.csv')

tot_asset = tot_asset.filter(intersection_result, axis=1)
tot_asset.to_csv('Final Variables/annual_tot_asset.csv')

volume = pd.read_csv('filtered_data/volume_data_sp500.csv', index_col=0)
volume = volume.filter(intersection_result, axis=1)
volume.to_csv('Final Variables/monthly_Volume.csv')
    
cum_sum = pd.read_csv('original data/control variables/cumsum_6mo.csv', index_col=0)
cum_sum.index = pd.to_datetime(cum_sum.index)
cum_sum.index = cum_sum.index.to_period("M")
cum_sum = cum_sum[ (cum_sum.index <= '2022-12') & (cum_sum.index >= '2014-01')]
cum_sum.to_csv('Final Variables/monthly_cum_sum.csv')

illiquidity = pd.read_csv('original data/control variables/illiquidity.csv', index_col=0)
illiquidity.index = pd.to_datetime(illiquidity.index)
illiquidity.index = illiquidity.index.to_period("M")
illiquidity = illiquidity[ (illiquidity.index <= '2022-12') & (illiquidity.index >= '2014-01')]
illiquidity.to_csv('Final Variables/monthly_illiquidity.csv')


