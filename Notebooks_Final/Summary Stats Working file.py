import pandas as pd
import numpy as np


def load_data(file_path, start_date , end_date):
    df = pd.read_csv(file_path, index_col=0)
    if df.index.dtype == int:
        df.index = pd.to_datetime(df.index, format='%Y%m')
    df.index = pd.to_datetime(df.index).to_period('M')
    df = df.loc[start_date : end_date]
    df.index.name = 'Date'
    #df = df.rename_axis('Date')
    return df

def lag_lead_operations(df, lag_date, lead_date):
    df_lag = df[df.index <= lag_date]
    df_lead = df.iloc[1:(len(df_lag) + 1)]
    return df_lag, df_lead

def summary_stats(df):
    
    column_stats = pd.DataFrame([df.mean(), df.std(), df.min(), df.quantile(0.25), df.quantile(0.5), df.quantile(0.75), df.max()])
    column_stats.index = ['Mean', 'SD', 'Min', '25%', 'Median', '75%', 'Max']
    
    return column_stats

# Specifying Start & End Date:
    
start_date = '2014-01'
end_date = '2022-12'

# Data Loading:
    
ret = pd.read_csv('Final Variables/monthly_returns.csv')
abpr = pd.read_csv('Final Variables/abnormal_positive_ratio.csv')
abnr = pd.read_csv('Final Variables/abnormal_negative_ratio.csv')
ret_co_m = pd.read_csv('variable_results/sp500/filtered_ret_co_m.csv')
ret_oc_m = pd.read_csv('variable_results/sp500/filtered_ret_oc_m.csv')
size = pd.read_csv('Final Variables/monthly_size.csv')
bm = pd.read_csv('variable_results/sp500/log_monthly_bm.csv')
ret_6m = pd.read_csv('Final Variables/monthly_cum_sum.csv')
gpa = pd.read_csv('Final Variables/annual_gross_profit.csv')
atgth = pd.read_csv('variable_results/sp500/filtered_monthly_asset_growth.csv')
turn_m = pd.read_csv('variable_results/sp500/log_monthly_turnover.csv')
illiq_m = pd.read_csv('variable_results/sp500/filtered_illiquidity.csv')

def correct_format(var):
    var['Unnamed: 0'] = pd.to_datetime(pd.to_datetime(var['Unnamed: 0']).dt.strftime('%Y-%m'))
    var = var.set_index('Unnamed: 0')
    var = var.reindex(sorted(var.columns), axis=1)
    return var

ret = correct_format(ret)
abpr = correct_format(abpr)
abnr = correct_format(abnr)
ret_co_m = correct_format(ret_co_m)
ret_oc_m = correct_format(ret_oc_m)
size = correct_format(size)
size = np.log(size)
bm = correct_format(bm)
ret_6m = correct_format(ret_6m)
gpa = correct_format(gpa)
atgth = atgth.rename(columns = {'Date' : 'Unnamed: 0'})
atgth = correct_format(atgth)
turn_m = turn_m.rename(columns = {'Date' : 'Unnamed: 0'})
turn_m = correct_format(turn_m)
illiq_m = illiq_m.rename(columns = {'index' : 'Unnamed: 0'})
illiq_m = correct_format(illiq_m)['2014-01-01' : '2022-12-13']
illiq_m = illiq_m.replace(0.0, np.float64('nan'))
illiq_m = np.log(illiq_m)

fix = ['MTCH', 'WBD']
for ticker in fix:
    for i in range(len(turn_m[ticker])):
        if turn_m[ticker][i] == '--':
            turn_m[ticker][i] = np.float64('nan')
        else:
            turn_m[ticker][i] = np.log(np.float64(turn_m[ticker][i]))
    turn_m[ticker] = turn_m[ticker].astype(float)

# Creating Lag & Lead:


ab_neg_lag, ab_neg_lead = lag_lead_operations(ab_neg, '2021-12-31', None)
ab_pos_lag, ab_pos_lead = lag_lead_operations(ab_pos, '2021-12-31', None)
ret_lag, ret_lead = lag_lead_operations(ret, '2021-12-31', None)
ret_oc_m_lag, ret_oc_m_lead = lag_lead_operations(ret_oc_m, '2021-12-31', None)
ret_co_m_lag, ret_co_m_lead = lag_lead_operations(ret_co_m, '2021-12-31', None)
size_lag, size_lead = lag_lead_operations(size, '2021-12-31', None)
bm_lag, bm_lead = lag_lead_operations(bm, '2021-12-31', None)
ret_6m_lag, ret_6m_lead = lag_lead_operations(ret_6m, '2021-12-31', None)
gpa_lag, gpa_lead = lag_lead_operations(gpa, '2021-12-31', None)
atgth_lag, atgth_lead = lag_lead_operations(atgth, '2021-12-31', None)
turn_m_lag, turn_m_lead = lag_lead_operations(turn_m, '2021-12-31', None)
illiq_m_lag, illiq_m_lead = lag_lead_operations(illiq_m, '2021-12-31', None)


# Calculating Mean:

ab_pos_means = ab_pos_lag.mean()
ab_neg_means = ab_neg_lag.mean()
ret_means = ret_lag.mean()
ret_oc_m_means = ret_oc_m_lag.mean()  
ret_co_m_means = ret_co_m_lag.mean()
size_means = size_lag.mean()
bm_means = bm_lag.mean()
ret_6m_means = ret_6m_lag.mean()
gpa_means = gpa_lag.mean()
atgth_means = atgth_lag.mean()
turn_m_means = turn_m_lag.mean()
illiq_m_means = illiq_m_lag.mean()

# Summary Statistics:
    
variables = ['ABPR', 'ABNR', 'RET', 'RET_OC_M', 'RET_CO_M', 'SIZE', 'BM','RET_6M','GPA','ATGTH','TURN_M','ILLIQ_M']
summary_stats_additional = pd.concat([summary_stats(ab_pos_means), summary_stats(ab_neg_means),summary_stats(ret_means), summary_stats(ret_oc_m_means), summary_stats(ret_co_m_means), summary_stats(size_means),summary_stats(bm_means), summary_stats(ret_6m_means) ,summary_stats(gpa_means), summary_stats(atgth_means),summary_stats(turn_m_means),summary_stats(illiq_m_means)], keys=variables, axis=1).transpose()
summary_stats_additional.index = summary_stats_additional.index.droplevel(1)

summary_stats_additional.to_csv('Table_3_A.csv')


ab_pos_means_df = pd.DataFrame(ab_pos_means, columns=['ABPR'])
ab_neg_means_df = pd.DataFrame(ab_neg_means, columns=['ABNR'])
ret_means_df = pd.DataFrame(ret_means, columns=['RET'])
ret_oc_m_means_df = pd.DataFrame(ret_oc_m_means, columns=['RET_OC_M'])
ret_co_m_means_df = pd.DataFrame(ret_co_m_means, columns=['RET_CO_M'])
size_means_df = pd.DataFrame(size_means, columns=['SIZE'])
bm_means_df = pd.DataFrame(bm_means, columns=['BM'])
ret_6m_means_df = pd.DataFrame(ret_6m_means, columns=['RET_6M'])
gpa_means_df = pd.DataFrame(gpa_means, columns=['GPA'])
atgth_means_df = pd.DataFrame(atgth_means, columns=['ATGTH'])
turn_m_means_df = pd.DataFrame(turn_m_means, columns=['TURN_M'])
illiq_m_means_df = pd.DataFrame(illiq_m_means, columns=['ILLIQ_M'])


lead_ab_pos_means_df = pd.DataFrame(ab_pos_lead.mean(), columns=['Lead_ABPR'])
lead_ab_neg_means_df = pd.DataFrame(ab_neg_lead.mean(), columns=['Lead_ABNR'])
lead_ret_means_df = pd.DataFrame(ret_lead.mean(), columns=['Lead_RET'])
lead_ret_oc_m_means_df = pd.DataFrame(ret_oc_m_lead.mean(), columns=['Lead_RET_OC_M'])
lead_ret_co_m_means_df = pd.DataFrame(ret_co_m_lead.mean(), columns=['Lead_RET_CO_M'])
lead_size_means_df = pd.DataFrame(size_lead.mean(), columns=['Lead_SIZE'])
lead_bm_means_df = pd.DataFrame(bm_lead.mean(), columns=['Lead_BM'])
lead_ret_6m_means_df = pd.DataFrame(ret_6m_lead.mean(), columns=['Lead_RET_6M'])
lead_gpa_means_df = pd.DataFrame(gpa_lead.mean(), columns=['Lead_GPA'])
lead_atgth_means_df = pd.DataFrame(atgth_lead.mean(), columns=['Lead_ATGTH'])
lead_turn_m_means_df = pd.DataFrame(turn_m_lead.mean(), columns=['Lead_TURN_M'])
lead_illiq_m_means_df = pd.DataFrame(illiq_m_lead.mean(), columns=['Lead_ILLIQ_M'])


additional_data_df = pd.concat([ab_pos_means_df, ab_neg_means_df,ret_means_df, ret_oc_m_means_df, ret_co_m_means_df, size_means_df,bm_means_df,ret_6m_means_df ,gpa_means_df, atgth_means_df,turn_m_means_df,illiq_m_means_df,
                                ],axis=1)
additional_correlation_matrix = additional_data_df.corr()

additional_correlation_matrix.to_csv('Table_3_B.csv')

# Printing the Summary Stats:
    
print(summary_stats_additional)
print(additional_correlation_matrix)