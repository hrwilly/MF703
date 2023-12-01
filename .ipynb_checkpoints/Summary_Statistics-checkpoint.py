#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 20:26:18 2023

@author: liyihan
"""

import pandas as pd

ab_pos = pd.read_csv('Final Variables/abnormal_positive_ratio.csv', index_col=0)
ab_neg = pd.read_csv('Final Variables/abnormal_negative_ratio.csv', index_col=0)
pos = pd.read_csv('Final Variables/positive_ratio.csv', index_col=0)
neg = pd.read_csv('Final Variables/negative_ratio.csv', index_col=0)

variables = ['NR', 'ABNR', 'PR', 'ABPR']
ab_pos_lag = ab_pos[ab_pos.index <= '2022-11']
ab_pos_lead = ab_pos.iloc[1:(len(ab_pos_lag)+1)]

ab_neg_lag = ab_neg[ab_neg.index <= '2022-11']
ab_neg_lead = ab_neg.iloc[1:(len(ab_neg_lag)+1)]

pos_lag = pos[pos.index <= '2022-11']
pos_lead = pos.iloc[1:(len(pos_lag)+1)]

neg_lag = neg[neg.index <= '2022-11']
neg_lead = neg.iloc[1:(len(neg_lag)+1)]

ab_pos_means = ab_pos_lag.mean()
ab_neg_means = ab_neg_lag.mean()
pos_means = pos_lag.mean()
neg_means = neg_lag.mean()

def summary_stats(df):
    
    column_stats = pd.DataFrame([df.mean(), df.std(), df.skew(), df.kurt(), df.min(), df.quantile(0.01), df.quantile(0.05), df.quantile(0.1), df.quantile(0.25), df.quantile(0.5), df.quantile(0.75), df.quantile(0.9), df.quantile(0.95), df.quantile(0.99), df.max()])
    column_stats.index = ['Mean', 'SD', 'Skew', 'Kurt', 'Min', '1%', '5%', '10%', '25%', 'Median', '75%', '90%', '95%', '99%', 'Max']
    
    return column_stats

summary_stats = pd.concat([summary_stats(neg_means), summary_stats(ab_neg_means), summary_stats(pos_means), summary_stats(ab_pos_means)], keys = variables, axis=1).transpose()
summary_stats.index = summary_stats.index.droplevel(1)
summary_stats.to_csv('summary statistic/table1.csv')

ab_pos_means_df = pd.DataFrame(ab_pos_means, columns=['ABPR'])
ab_neg_means_df = pd.DataFrame(ab_neg_means, columns=['ABNR'])
pos_means_df = pd.DataFrame(pos_means, columns=['PR'])
neg_means_df = pd.DataFrame(neg_means, columns=['NR'])

lead_ab_pos_means_df = pd.DataFrame(ab_pos_lead.mean(), columns=['Lead_ABPR'])
lead_ab_neg_means_df = pd.DataFrame(ab_neg_lead.mean(), columns=['Lead_ABNR'])
lead_pos_means_df = pd.DataFrame(pos_lead.mean(), columns=['Lead_R'])
lead_neg_means_df = pd.DataFrame(neg_lead.mean(), columns=['Lead_NR'])

data_df = pd.concat([neg_means_df, ab_neg_means_df, pos_means_df, ab_pos_means_df, lead_neg_means_df, lead_ab_neg_means_df, lead_pos_means_df, lead_ab_pos_means_df], axis=1)
correlation_matrix = data_df.corr()
correlation_matrix.to_csv('summary statistic/table2.csv')