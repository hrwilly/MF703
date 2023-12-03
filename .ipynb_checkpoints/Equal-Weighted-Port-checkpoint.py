#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:20:55 2023

@author: liyihan
"""

import pandas as pd
import statsmodels.api as sm

def load_data(file_path, start_date, end_date):
    df = pd.read_csv(file_path, index_col=0)
    if df.index.dtype == int:
        df.index = pd.to_datetime(df.index, format='%Y%m')
    df.index = pd.to_datetime(df.index).to_period('M')
    df = df.loc[start_date : end_date]
    return df


def get_quantile(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df_ranks = df.apply(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'), axis=1)
    return df_ranks

def get_stocks_in_quantile(df, quantile):
    stocks_result = {}
    df_ranks = get_quantile(df)
    for index, row in df_ranks.iterrows():
        selected_columns = [col for col, value in row.items() if value == quantile]
        stocks_result[index] = ', '.join(selected_columns)
    

    result_df = pd.DataFrame(list(stocks_result.items()), columns=['Date', quantile])
    result_df.set_index('Date', inplace=True)
    return result_df

def cal_portfolio_return(df, return_df, ff3_df, quantile):
    stocks_result = get_stocks_in_quantile(df, quantile)
    portfolio_return = pd.DataFrame(index=return_df.index[1:], columns=['EquallyWeightedReturn'])
    for i in range(len(df)-1):
        stocks_list = stocks_result.iloc[i]
        stocks_list = stocks_list.iloc[0].split(', ')
        
        index = return_df.index[i+1]
        values = return_df.loc[index][return_df.columns.isin(stocks_list)]
        value_equally = 100 * values.mean()
        portfolio_return.at[portfolio_return.index[i], 'EquallyWeightedReturn'] = value_equally
    return portfolio_return

def get_quantile_returns(df, return_df, ff3_df):
    quantile_returns = pd.DataFrame()
    for quantile in range(10):
        quantile_returns[quantile+1] = cal_portfolio_return(df, return_df, ff3_df, quantile)
    
    quantile_returns[11] = quantile_returns[10]-quantile_returns[1]
    return quantile_returns


def cal_CAPM(df, ff3_df):
    result = pd.DataFrame(index = range(1,12), columns= ['Alpha', 'T-Test'])
    for column in range(1,12):
        Y = pd.to_numeric(df[column], errors='coerce').dropna()
        temp = pd.merge(Y, ff3_df['Mkt-RF'], left_index=True, right_index=True)
        temp['Intercept'] = 1.0
        X = temp[['Intercept', 'Mkt-RF']]

        model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        alpha = round(model.params['Intercept'], 2)
        p_value = model.pvalues['Intercept']
        result.at[column, 'Alpha'] = alpha
        if p_value < 0.01:
            result.at[column, 'T-Test'] = "***"
        elif 0.01 < p_value < 0.05:
            result.at[column, 'T-Test'] = "**"
        elif 0.05 < p_value < 0.10:
            result.at[column, 'T-Test'] = "*"
        else:
            result.at[column, 'T-Test'] = " "
            
        result.at[12, 'Alpha'] = result['Alpha'].loc[10] - result['Alpha'].loc[1]
        result.at[12, 'T-Test'] = result['T-Test'].loc[10]

    
    return result

def cal_Fama_French(df, ff3_df):
    result = pd.DataFrame(index = range(1,13), columns= ['Alpha', 'T-Test'])
    for column in range(1,12):
        Y = pd.to_numeric(df[column], errors='coerce').dropna()
        temp = pd.merge(Y, ff3_df, left_index=True, right_index=True)
        temp['Intercept'] = 1.0
        X = temp[['Intercept', 'Mkt-RF', 'SMB', 'HML']]

        model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        alpha = round(model.params['Intercept'], 2)
        p_value = model.pvalues['Intercept']
        result.at[column, 'Alpha'] = alpha
        if p_value < 0.01:
            result.at[column, 'T-Test'] = "***"
        elif 0.01 < p_value < 0.05:
            result.at[column, 'T-Test'] = "**"
        elif 0.05 < p_value < 0.10:
            result.at[column, 'T-Test'] = "*"
        else:
            result.at[column, 'T-Test'] = " "
    result.at[12, 'Alpha'] = result['Alpha'].loc[10] - result['Alpha'].loc[1]
    result.at[12, 'T-Test'] = result['T-Test'].loc[10]
    
    return result

    
def result_report(excess_df, capm_df, ff_df):
    new_row = pd.DataFrame({'Excess': [excess_df['Excess'].iloc[-2] - excess_df['Excess'].iloc[0]]}, index=[12])
    excess_df = pd.concat([excess_df, new_row])
    
    result = pd.merge(excess_df, capm_df, left_index=True, right_index=True)
    result = result.rename(columns={'Alpha': 'CAPM alpha'})
    result = result.rename(columns={'T-Test': 'CAPM T-Test'})
    result = pd.merge(result, ff_df, left_index=True, right_index=True)
    result = result.rename(columns={'Alpha': 'FF-3 alpha'})
    result = result.rename(columns={'T-Test': 'FF-3 T-Test'})
    result = result.rename(index={1: 'L'})
    result = result.rename(index={10: 'H'})
    result = result.rename(index={12: 'H-L'})
    result = result.drop(11)
    
    return result


#   Hedge Part

def cal_hedge_portfolio_return(df, return_df, ff3_df, quantile, other_df):
    stocks_result = get_stocks_in_quantile(df, quantile)
    portfolio_return = pd.DataFrame(index=return_df.index[1:], columns=['EquallyWeightedReturn'])
    for i in range(len(df)-1):
        stocks_list = stocks_result.iloc[i]
        stocks_list = stocks_list.iloc[0].split(', ')
        index = return_df.index[i+1]
        index_other = other_df.index[i]
        
        other_values_series = other_df.loc[index_other][other_df.columns.isin(stocks_list)]
        other_values = other_values_series.to_frame()
        other_values = other_values.T
        stocks_result_low = get_stocks_in_quantile(other_values, 0)
        stocks_list_low = stocks_result_low.iloc[0]
        stocks_list_low = stocks_list_low.iloc[0].split(', ')
        
        stocks_result_high = get_stocks_in_quantile(other_values, 9)
        stocks_list_high = stocks_result_high.iloc[0]
        stocks_list_high = stocks_list_high.iloc[0].split(', ')

            
        values_high = return_df.loc[index][return_df.columns.isin(stocks_list_high)]
        values_low = return_df.loc[index][return_df.columns.isin(stocks_list_low)]

        value_equally_high = 100 * values_high.mean()
        value_equally_low = 100 * values_low.mean()
        
        value_equally = value_equally_high - value_equally_low
        portfolio_return.at[portfolio_return.index[i], 'EquallyWeightedReturn'] = value_equally
        
    return portfolio_return

def get_quantile_hedge_returns(df, return_df, ff3_df, other_df):
    quantile_returns = pd.DataFrame()
    for quantile in range(10):
        quantile_returns[quantile+1] = cal_hedge_portfolio_return(df, return_df, ff3_df, quantile, other_df)
    
    quantile_returns[11] = quantile_returns[10]-quantile_returns[1]
    return quantile_returns


return_df = load_data('Final Variables/monthly_returns.csv', '2014-01', '2022-12')
ff3_df = load_data('filtered_data/ff3.csv', '2014-01', '2022-12')
ab_nr_df = load_data('Final Variables/abnormal_negative_ratio.csv', '2014-01', '2022-12')
ab_pr_df = load_data('Final Variables/abnormal_positive_ratio.csv', '2014-01', '2022-12')

ABNR_quantile_returns = get_quantile_returns(ab_nr_df, return_df, ff3_df)
ABPR_quantile_returns = get_quantile_returns(ab_pr_df, return_df, ff3_df)

ABNR_excess = pd.DataFrame(round(ABNR_quantile_returns.mean(), 2), columns=['Excess'])
ABPR_excess = pd.DataFrame(round(ABPR_quantile_returns.mean(), 2), columns=['Excess'])


ABNR_CAPM = cal_CAPM(ABNR_quantile_returns, ff3_df)
ABNR_FF3 = cal_Fama_French(ABNR_quantile_returns, ff3_df)
ABNR_result = result_report(ABNR_excess, ABNR_CAPM, ABNR_FF3).T
ABNR_result.to_csv('Equally_Weighted_Portfolio/sp500_ABNR_result.csv')

ABPR_CAPM = cal_CAPM(ABPR_quantile_returns, ff3_df)
ABPR_FF3 = cal_Fama_French(ABPR_quantile_returns, ff3_df)
ABPR_result = result_report(ABPR_excess, ABPR_CAPM, ABPR_FF3).T
ABPR_result.to_csv('Equally_Weighted_Portfolio/sp500_ABPR_result.csv')


#   Hedge Part
size = load_data('Final Variables/monthly_size.csv', '2014-01', '2022-12')

hedge_ABNR_quantile_returns = get_quantile_hedge_returns(size, return_df, ff3_df, ab_nr_df)
hedge_ABPR_quantile_returns = get_quantile_hedge_returns(size, return_df, ff3_df, ab_pr_df)

hedge_ABNR_excess = pd.DataFrame(round(hedge_ABNR_quantile_returns.mean(), 2), columns=['Excess'])
hedge_ABPR_excess = pd.DataFrame(round(hedge_ABPR_quantile_returns.mean(), 2), columns=['Excess'])



hedge_ABNR_CAPM = cal_CAPM(hedge_ABNR_quantile_returns, ff3_df)
hedge_ABNR_FF3 = cal_Fama_French(hedge_ABNR_quantile_returns, ff3_df)
hedge_ABNR_result = result_report(hedge_ABNR_excess, hedge_ABNR_CAPM, hedge_ABNR_FF3).iloc[:-1]
hedge_ABNR_result.to_csv('Equally_Weighted_Portfolio/sp500_hedge_ABNR_result.csv')

hedge_ABPR_CAPM = cal_CAPM(hedge_ABPR_quantile_returns, ff3_df)
hedge_ABPR_FF3 = cal_Fama_French(hedge_ABPR_quantile_returns, ff3_df)
hedge_ABPR_result = result_report(hedge_ABPR_excess, hedge_ABPR_CAPM, hedge_ABPR_FF3).iloc[:-1]
hedge_ABPR_result.to_csv('Equally_Weighted_Portfolio/sp500_hedge_ABPR_result.csv')

