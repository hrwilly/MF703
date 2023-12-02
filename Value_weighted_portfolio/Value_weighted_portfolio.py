import pandas as pd
import numpy as np
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

def cal_portfolio_return(df, return_df, size_df, quantile):
    stocks_result = get_stocks_in_quantile(df, quantile)
    portfolio_return = pd.DataFrame(index=return_df.index[1:], columns=['ValueWeightedReturn'])
    for i in range(len(df)-1):
        stocks_list = stocks_result.iloc[i]
        stocks_list = stocks_list.iloc[0].split(', ')
        
        index = return_df.index[i+1]
        returns = return_df.loc[index][return_df.columns.isin(stocks_list)]
        sizes = size_df.loc[index][size_df.columns.isin(stocks_list)]

        weighted_returns = returns.multiply(sizes) / sizes.sum()
        portfolio_return.at[portfolio_return.index[i], 'ValueWeightedReturn'] = 100 * weighted_returns.sum()

    return portfolio_return

def get_quantile_returns(df, return_df, size_df):
    quantile_returns = pd.DataFrame()
    for quantile in range(10):
        quantile_returns[quantile+1] = cal_portfolio_return(df, return_df, size_df, quantile)
    quantile_returns[11] = quantile_returns[10] - quantile_returns[1]
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

def calculate_hedged_portfolio_returns(returns_df, ranks_df, market_cap_df):
    # Calculate market cap deciles
    market_cap_deciles = market_cap_df.rank(axis=1, method='min', pct=True)
    market_cap_deciles = pd.qcut(market_cap_deciles.stack(), 10, labels=False, duplicates='drop').unstack()

    # Dictionary to store hedged returns for each decile
    hedge_returns_dict = {}

    for decile in range(10):
        size_decile_stocks = market_cap_deciles == decile
        size_decile_returns = returns_df[size_decile_stocks]
        
        ab_nr_within_decile = ranks_df[size_decile_stocks].stack().groupby(level=0).rank(pct=True)
        high_ab_nr_stocks = ab_nr_within_decile[ab_nr_within_decile > 0.9].index.get_level_values(1)
        low_ab_nr_stocks = ab_nr_within_decile[ab_nr_within_decile < 0.1].index.get_level_values(1)

        high_weighted_returns = size_decile_returns[high_ab_nr_stocks].mul(market_cap_df[high_ab_nr_stocks]).sum(axis=1) / market_cap_df[high_ab_nr_stocks].sum(axis=1)
        low_weighted_returns = size_decile_returns[low_ab_nr_stocks].mul(market_cap_df[low_ab_nr_stocks]).sum(axis=1) / market_cap_df[low_ab_nr_stocks].sum(axis=1)

        hedged_returns = high_weighted_returns - low_weighted_returns

        # Store the hedged returns
        hedge_returns_dict[f'Decile {decile + 1}'] = hedged_returns
        
    # Return the results as a DataFrame
    return pd.DataFrame(hedge_returns_dict)

def run_regression_analysis(returns, market_factor, smb, hml):
    results = []

    for decile in returns.columns:
        Y = returns[decile].dropna()
        ff_data = pd.concat([market_factor, smb, hml], axis=1).loc[Y.index]

        # CAPM Regression
        X_capm = sm.add_constant(ff_data['Mkt-RF'])
        capm_model = sm.OLS(Y, X_capm).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        capm_alpha = round(capm_model.params['const'], 4)
        capm_p_value = capm_model.pvalues['const']

        # Fama-French Regression
        X_ff = sm.add_constant(ff_data)
        ff_model = sm.OLS(Y, X_ff).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
        ff_alpha = round(ff_model.params['const'], 4)
        ff_p_value = ff_model.pvalues['const']

        results.append({
            'Decile': decile,
            'CAPM Alpha': capm_alpha,
            'CAPM T-Test': "***" if capm_p_value < 0.01 else "**" if capm_p_value < 0.05 else "*" if capm_p_value < 0.1 else '',
            'FF Alpha': ff_alpha,
            'FF T-Test': "***" if ff_p_value < 0.01 else "**" if ff_p_value < 0.05 else "*" if ff_p_value < 0.1 else '',
        })
    
    return pd.DataFrame(results)


def safe_round(series, decimals):
    try:
        return series.round(decimals)
    except TypeError:
        return series

    
return_df = load_data('monthly_returns.csv', '2014-01', '2022-12')
ff3_df = load_data('ff3.csv', '2014-01', '2022-12')
ab_nr_df = load_data('abnormal_negative_ratio.csv', '2014-01', '2022-12')
ab_pr_df = load_data('abnormal_positive_ratio.csv', '2014-01', '2022-12')
size = load_data('reordered_monthly_size.csv', '2014-01', '2022-12')

ABNR_quantile_returns = get_quantile_returns(ab_nr_df, return_df, size)
ABPR_quantile_returns = get_quantile_returns(ab_pr_df, return_df, size)

ABNR_excess = pd.DataFrame({'Excess': safe_round(ABNR_quantile_returns.mean(), 2)})
ABPR_excess = pd.DataFrame({'Excess': safe_round(ABPR_quantile_returns.mean(), 2)})

ABNR_CAPM = cal_CAPM(ABNR_quantile_returns, ff3_df)
ABNR_FF3 = cal_Fama_French(ABNR_quantile_returns, ff3_df)
ABNR_result = result_report(ABNR_excess, ABNR_CAPM, ABNR_FF3).T
ABNR_result.to_csv('sp500_ABNR_result.csv')

ABPR_CAPM = cal_CAPM(ABPR_quantile_returns, ff3_df)
ABPR_FF3 = cal_Fama_French(ABPR_quantile_returns, ff3_df)
ABPR_result = result_report(ABPR_excess, ABPR_CAPM, ABPR_FF3).T
ABPR_result.to_csv('sp500_ABPR_result.csv')

#   Hedge Part
hedge_ABNR_quantile_returns = calculate_hedged_portfolio_returns(return_df, ab_nr_df, size)*100
hedge_ABNR_excess = hedge_ABNR_quantile_returns.mean().to_frame(name='Excess Return')
hedge_ABNR_excess.index = ['Decile ' + str(i) for i in range(1, 11)]
hedge_ABNR_results = run_regression_analysis(hedge_ABNR_quantile_returns, ff3_df['Mkt-RF'], ff3_df['SMB'], ff3_df['HML'])
hedge_ABNR_results.set_index('Decile', inplace=True)
hedge_ABNR = pd.concat([round(hedge_ABNR_excess,4), hedge_ABNR_results], axis=1)
hedge_ABNR.to_csv('ABNR_hedge_results.csv')

hedge_ABPR_quantile_returns = calculate_hedged_portfolio_returns(return_df, ab_pr_df, size)*100
hedge_ABPR_excess = hedge_ABPR_quantile_returns.mean().to_frame(name='Excess Return')
hedge_ABPR_excess.index = ['Decile ' + str(i) for i in range(1, 11)]
hedge_ABPR_results = run_regression_analysis(hedge_ABPR_quantile_returns, ff3_df['Mkt-RF'], ff3_df['SMB'], ff3_df['HML'])
hedge_ABPR_results.set_index('Decile', inplace=True)
hedge_ABPR = pd.concat([round(hedge_ABPR_excess,4), hedge_ABPR_results], axis=1)
hedge_ABPR.to_csv('ABPR_hedge_results.csv')


