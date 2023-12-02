import pandas as pd
import numpy as np
from finance_byu import fama_macbeth
import warnings
warnings.filterwarnings("ignore")

ab_pos = pd.read_csv('Final Variables/abnormal_positive_ratio.csv', index_col=0)
ab_neg = pd.read_csv('Final Variables/abnormal_negative_ratio.csv', index_col=0)

def fama(data, shift_factor, control = []):

    future = data.shift(shift_factor)
    
    comb = []
    tickers = data.columns

    for i in range(len(tickers)):
        data[tickers[i]].index = pd.to_datetime(data[tickers[i]].index)
        combined = pd.concat([data[tickers[i]], future[tickers[i]].rename('future')], axis = 1)
        if control != []:
            combined = pd.concat([combined, control[i]], axis = 1)

        comb.append(combined[:shift_factor])

    for i in range(len(comb)):
        comb[i] = comb[i].reset_index()
        comb[i]['index'] = pd.to_datetime(pd.to_datetime(comb[i]['index']).dt.strftime('%Y-%m'))

    results = []
    tickers = data.columns        

    for i in range(len(comb)):
        xvar = [tickers[i]]
        comb[i] = comb[i].dropna(axis = 1, how = 'all')
        comb[i] = comb[i].dropna(axis = 0, how = 'any')
        if control != []:
            for j in range(3, len(comb[i].columns)):
                xvar.append(comb[i].columns[j])
        comb[i] = comb[i].dropna(axis = 1, how = 'all')
        comb[i] = comb[i].dropna(axis = 0, how = 'any')
        results.append(fama_macbeth.fama_macbeth_master(comb[i], t = 'index', yvar = 'future', xvar = xvar, intercept=True))

    alpha = []
    beta = []
    for i in range(len(results)):
        alpha.append(results[i]['intercept'])
        beta.append(results[i][tickers[i]])

    return alpha, beta, results

ab_neg.index = pd.to_datetime(ab_neg.index)
ab_pos.index = pd.to_datetime(ab_pos.index)
ab_neg = ab_neg.reindex(sorted(ab_neg.columns), axis=1)
ab_pos = ab_pos.reindex(sorted(ab_pos.columns), axis=1)

neg1_alpha, neg1_beta, neg1_results = fama(ab_neg, -1)
neg2_alpha, neg2_beta, neg2_results = fama(ab_neg, -2)
neg3_alpha, neg3_beta, neg3_results = fama(ab_neg, -3)
pos1_alpha, pos1_beta, pos1_results = fama(ab_pos, -1)
pos2_alpha, pos2_beta, pos2_results = fama(ab_pos, -2)
pos3_alpha, pos3_beta, pos3_results = fama(ab_pos, -3)

d_beta = {'ABNR 1mo' : neg1_beta, 'ABNR 2mo' : neg2_beta, 'ABNR 3mo' : neg3_beta, 
          'ABPR 1mo' : pos1_beta, 'ABPR 2mo' : pos2_beta, 'ABPR 3mo' : pos3_beta}
d_alpha = {'ABNR 1mo' : neg1_alpha, 'ABNR 2mo' : neg2_alpha, 'ABNR 3mo' : neg3_alpha, 
          'ABPR 1mo' : pos1_alpha, 'ABPR 2mo' : pos2_alpha, 'ABPR 3mo' : pos3_alpha}

betas = np.mean(pd.DataFrame(data = d_beta))
alphas = np.mean(pd.DataFrame(data = d_alpha))

results_no_control = pd.concat([alphas.rename('Intercept'), betas.rename('Beta')], axis = 1)
results_no_control = results_no_control.style.set_caption("Fama Macbeth without Control Variables")

ret_co_m = pd.read_csv('variable_results/sp500/filtered_ret_co_m.csv')
ret_oc_m = pd.read_csv('variable_results/sp500/filtered_ret_oc_m.csv')

def correct_format(var):
    var['Unnamed: 0'] = pd.to_datetime(pd.to_datetime(var['Unnamed: 0']).dt.strftime('%Y-%m'))
    var = var.set_index('Unnamed: 0')
    var = var.reindex(sorted(var.columns), axis=1)
    return var

ret_co_m = correct_format(ret_co_m)
ret_oc_m = correct_format(ret_oc_m)

controls = []

for ticker in size:
    data = pd.DataFrame()
    data['close to open'] = ret_co_m[ticker]
    data['open to close'] = ret_oc_m[ticker]
    controls.append(data)

neg1_alpha_c, neg1_beta_c, neg1_results_c = fama(ab_neg, -1, controls)
neg2_alpha_c, neg2_beta_c, neg2_results_c = fama(ab_neg, -2, controls)
neg3_alpha_c, neg3_beta_c, neg3_results_c = fama(ab_neg, -3, controls)
pos1_alpha_c, pos1_beta_c, pos1_results_c = fama(ab_pos, -1, controls)
pos2_alpha_c, pos2_beta_c, pos2_results_c = fama(ab_pos, -2, controls)
pos3_alpha_c, pos3_beta_c, pos3_results_c = fama(ab_pos, -3, controls)

d_beta_c = {'ABNR 1mo' : neg1_beta_c, 'ABNR 2mo' : neg2_beta_c, 'ABNR 3mo' : neg3_beta_c, 
          'ABPR 1mo' : pos1_beta_c, 'ABPR 2mo' : pos2_beta_c, 'ABPR 3mo' : pos3_beta_c}
d_alpha_c = {'ABNR 1mo' : neg1_alpha_c, 'ABNR 2mo' : neg2_alpha_c, 'ABNR 3mo' : neg3_alpha_c, 
          'ABPR 1mo' : pos1_alpha_c, 'ABPR 2mo' : pos2_alpha_c, 'ABPR 3mo' : pos3_alpha_c}

betas_c = np.mean(pd.DataFrame(data = d_beta_c))
alphas_c = np.mean(pd.DataFrame(data = d_alpha_c))

results_w_control = pd.concat([alphas_c.rename('Intercept'), betas_c.rename('Beta')], axis = 1)
results_w_control.style.set_caption("Fama Macbeth with Control Variables")

results_no_control.to_csv('summary statistic/reg_no_control.csv')
results_w_control.to_csv('summary statistic/reg_w_control.csv')