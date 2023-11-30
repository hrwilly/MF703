"""
Created on Tue Nov 28 10:00:43 2023

@author: LiangJiali
"""

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

def rank_deciles(df):
    df_ranks = df.apply(lambda x: pd.qcut(x, 10, labels=False, duplicates='drop'), axis=0)
    return df_ranks

def calculate_value_weighted_returns(returns, market_caps, decile_ranks):
    weighted_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for stock in returns.columns:
        for month in returns.index:
            decile = decile_ranks.at[month, stock]
            if pd.notnull(decile):
                weighted_returns.at[month, stock] = returns.at[month, stock] * market_caps.at[month, stock]
                
    
    decile_returns = pd.DataFrame(index=returns.index, columns=np.arange(10))
    for decile in range(10):
        decile_stocks = decile_ranks == decile
        decile_returns[decile] = weighted_returns[decile_stocks].sum(axis=1) / market_caps[decile_stocks].sum(axis=1)
        
    return decile_returns.dropna()

def filter_top_stocks_by_market_cap(market_cap_df, returns_df, percentile=0.99):
    """This function will filter out the top stocks by market cap."""
    threshold = market_cap_df.quantile(percentile, axis=1)
    is_below_threshold = market_cap_df.lt(threshold, axis=0)
    filtered_market_caps = market_cap_df[is_below_threshold]
    filtered_returns = returns_df[is_below_threshold]
    return filtered_market_caps, filtered_returns

def perform_regression(Y, X, regression_type='CAPM'):
    if regression_type == 'CAPM':
        X = sm.add_constant(X['Mkt-RF'])  
    elif regression_type == 'FF3':
        X = sm.add_constant(X[['Mkt-RF', 'SMB', 'HML']])  
    else:
        raise ValueError("Invalid regression type specified.")

    model = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
    return model

def format_significance(p_value):
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    else:
        return ''

def calculate_regression_results(weighted_returns, market_factor, ff_factors):
    results = []
    for decile in range(10):
        decile_returns = weighted_returns.iloc[:, decile]
        
        # Perform CAPM regression
        capm_model = perform_regression(decile_returns, market_factor, 'CAPM')
        capm_alpha = capm_model.params['const']
        capm_t_stat = capm_model.tvalues['const']
        capm_p_value = capm_model.pvalues['const']
        
        # Perform FF3 regression
        ff3_model = perform_regression(decile_returns, ff_factors, 'FF3')
        ff3_alpha = ff3_model.params['const']
        ff3_t_stat = ff3_model.tvalues['const']
        ff3_p_value = ff3_model.pvalues['const']

        # Format results
        results.append({
            'Decile': 'Low' if decile == 0 else 'High' if decile == 9 else decile,
            'Raw Return': decile_returns.mean(),
            't-stat': '({:.2f})'.format(decile_returns.std()),
            'CAPM Alpha': capm_alpha,
            'CAPM t-stat': '({:.2f})'.format(float(capm_t_stat)),
            'CAPM Significance': format_significance(capm_p_value),
            'FF3 Alpha': ff3_alpha,
            'FF3 t-stat': '({:.2f})'.format(ff3_t_stat),
            'FF3 Significance': format_significance(ff3_p_value)
        })

    # Calculate (H-L)
    high_returns = weighted_returns.iloc[:, 9]
    low_returns = weighted_returns.iloc[:, 0]
    hl_returns = high_returns - low_returns
    capm_model_hl = perform_regression(hl_returns, market_factor, 'CAPM')
    ff3_model_hl = perform_regression(hl_returns, ff_factors, 'FF3')

    # Append H-L 
    results.append({
        'Decile': 'H-L',
        'Raw Return': hl_returns.mean(),
        't-stat': '({:.2f})'.format(hl_returns.std()),
        'CAPM Alpha': capm_model_hl.params['const'],
        'CAPM t-stat': '({:.2f})'.format(capm_model_hl.tvalues['const']),
        'CAPM Significance': format_significance(capm_model_hl.pvalues['const']),
        'FF3 Alpha': ff3_model_hl.params['const'],
        'FF3 t-stat': '({:.2f})'.format(ff3_model_hl.tvalues['const']),
        'FF3 Significance': format_significance(ff3_model_hl.pvalues['const'])
    })

    return pd.DataFrame(results)


def calculate_hedged_portfolio_returns(returns_df, ranks_df, market_factor, ff_factors):

    def get_quintiles_within_decile(decile_rank):
        decile_df = returns_df.where(ranks_df == decile_rank)
        quantiles = decile_df.rank(axis=1, method='first', pct=True)
        highest_quantile = decile_df.where(quantiles > 0.8).mean(axis=1)
        lowest_quantile = decile_df.where(quantiles <= 0.2).mean(axis=1)
        return highest_quantile, lowest_quantile

    results = []

    # Iterate through each decile
    for decile_rank in range(10):
        high, low = get_quintiles_within_decile(decile_rank)
        hedged_returns = high - low

        # Calculate t-stat for Raw Return
        raw_return_t_stat = hedged_returns.mean() / hedged_returns.std()

        # Perform CAPM and FF3 regressions on the hedged returns
        capm_model = perform_regression(hedged_returns, market_factor, 'CAPM')
        ff3_model = perform_regression(hedged_returns, ff_factors, 'FF3')

        # Gather results
        results.append({
            'Decile': 'L' if decile_rank == 0 else 'H' if decile_rank == 9 else decile_rank,
            'Raw Return': hedged_returns.mean(),
            'Raw Return t-stat': raw_return_t_stat,
            'CAPM Alpha': capm_model.params['const'],
            'CAPM Significance': format_significance(capm_model.pvalues['const']),
            'FF3 Alpha': ff3_model.params['const'],
            'FF3 Significance': format_significance(ff3_model.pvalues['const'])
        })

    results_df = pd.DataFrame(results)
    return results_df

returns_df = load_data('monthly_returns-2.csv', '2022-01', '2023-10')
ff3_factors = load_data('ff3.csv', '2022-01', '2023-10')
ab_nr_df = load_data('abnormal_negative_ratio.csv', '2022-01', '2023-10')
ab_pr_df = load_data('abnormal_positive_ratio.csv', '2022-01', '2023-10')
market_cap_df = load_data('dummy_market_caps.csv', '2022-01', '2023-10')

market_factor = ff3_factors[['Mkt-RF']]
                            
abnr_ranks = rank_deciles(ab_nr_df)
abpr_ranks = rank_deciles(ab_pr_df)

# value-weighted portfolio
vw_returns_ab_nr = calculate_value_weighted_returns(returns_df, market_cap_df, abnr_ranks)
vw_returns_ab_pr = calculate_value_weighted_returns(returns_df, market_cap_df, abpr_ranks)

regression_results_ab_nr = calculate_regression_results(vw_returns_ab_nr, market_factor, ff3_factors)
regression_results_ab_pr = calculate_regression_results(vw_returns_ab_pr, market_factor, ff3_factors)

# Filter out the top 1% stocks by market cap for each time period
filtered_market_caps, filtered_returns = filter_top_stocks_by_market_cap(market_cap_df, returns_df, 0.99)

filtered_vw_returns_ab_nr = calculate_value_weighted_returns(filtered_returns, filtered_market_caps, abnr_ranks)
filtered_vw_returns_ab_pr = calculate_value_weighted_returns(filtered_returns, filtered_market_caps, abpr_ranks)

filtered_regression_results_ab_nr = calculate_regression_results(filtered_vw_returns_ab_nr, market_factor, ff3_factors)
filtered_regression_results_ab_pr = calculate_regression_results(filtered_vw_returns_ab_pr, market_factor, ff3_factors)

# hedged portfolio
hedged_abnr = calculate_hedged_portfolio_returns(returns_df, abnr_ranks,market_factor, ff3_factors)
hedged_abpr = calculate_hedged_portfolio_returns(returns_df, abpr_ranks,market_factor, ff3_factors)

regression_results_ab_nr.to_csv('value_weighted_ab_nr.csv')
regression_results_ab_pr.to_csv('value_weighted_ab_pr.csv')
filtered_regression_results_ab_nr.to_csv('filtered_ab_nr.csv')
filtered_regression_results_ab_pr.to_csv('filtered_ab_pr.csv')
hedged_abnr.to_csv('hedged_abnr.csv')
hedged_abnr.to_csv('hedged_abpr.csv')
