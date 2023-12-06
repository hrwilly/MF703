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
    sum = []
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
        reg = fama_macbeth.fama_macbeth(comb[i], t = 'index', yvar = 'future', xvar = xvar, intercept = True)
        sum.append(fama_macbeth.fm_summary(reg, pvalues = True))
        results.append(fama_macbeth.fama_macbeth_master(comb[i], t = 'index', yvar = 'future', xvar = xvar, intercept = True))

    beta = []
    for i in range(len(results)):
        beta.append(results[i][tickers[i]])

    return beta, sum

ab_neg.index = pd.to_datetime(ab_neg.index)
ab_pos.index = pd.to_datetime(ab_pos.index)
ab_neg = ab_neg.reindex(sorted(ab_neg.columns), axis=1)
ab_pos = ab_pos.reindex(sorted(ab_pos.columns), axis=1)

neg1b, neg1s = fama(ab_neg, -1)
neg2b, neg2s = fama(ab_neg, -2)
neg3b, neg3s = fama(ab_neg, -3)
pos1b, pos1s = fama(ab_pos, -1)
pos2b, pos2s = fama(ab_pos, -2)
pos3b, pos3s = fama(ab_pos, -3)

def find_vals(data):
    int = []
    pval = []
    for i in range(len(data)):
        int.append(data[i]['mean'][0])
        pval.append(data[i]['pval'][1])
    return int, pval

neg1a, neg1_pval = find_vals(neg1s)
neg2a, neg2_pval = find_vals(neg2s)
neg3a, neg3_pval = find_vals(neg3s)
pos1a, pos1_pval = find_vals(pos1s)
pos2a, pos2_pval = find_vals(pos2s)
pos3a, pos3_pval = find_vals(pos3s)

d_alpha = {'ABNR 1mo' : neg1a, 'ABNR 2mo' : neg2a, 'ABNR 3mo' : neg3a,
           'ABPR 1mo' : pos1a, 'ABPR 2mo' : pos2a, 'ABPR 3mo' : pos3a}
d_beta = {'ABNR 1mo' : neg1b, 'ABNR 2mo' : neg2b, 'ABNR 3mo' : neg3b,
          'ABPR 1mo' : pos1b, 'ABPR 2mo' : pos2b, 'ABPR 3mo' : pos3b}
d_pval = {'ABNR 1mo' : neg1_pval, 'ABNR 2mo' : neg2_pval, 'ABNR 3mo' : neg3_pval,
          'ABPR 1mo' : pos1_pval, 'ABPR 2mo' : pos2_pval, 'ABPR 3mo' : pos3_pval}

alphas = np.mean(pd.DataFrame(data = d_alpha))
betas = np.mean(pd.DataFrame(data = d_beta))
pvals = np.mean(pd.DataFrame(data = d_pval))

res_n_c = pd.concat([alphas.rename('Intercept'), betas.rename('Beta'), pvals.rename('p')], axis = 1)
res_n_c.style.set_caption("Fama Macbeth without Control Variables")

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

for ticker in ret_co_m:
    data = pd.DataFrame()
    data['close to open'] = ret_co_m[ticker]
    data['open to close'] = ret_oc_m[ticker]
    controls.append(data)

neg1b_c, neg1s_c = fama(ab_neg, -1, controls)
neg2b_c, neg2s_c = fama(ab_neg, -2, controls)
neg3b_c, neg3s_c = fama(ab_neg, -3, controls)
pos1b_c, pos1s_c = fama(ab_pos, -1, controls)
pos2b_c, pos2s_c = fama(ab_pos, -2, controls)
pos3b_c, pos3s_c = fama(ab_pos, -3, controls)

neg1a_c, neg1_pval_c = find_vals(neg1s_c)
neg2a_c, neg2_pval_c = find_vals(neg2s_c)
neg3a_c, neg3_pval_c = find_vals(neg3s_c)
pos1a_c, pos1_pval_c = find_vals(pos1s_c)
pos2a_c, pos2_pval_c = find_vals(pos2s_c)
pos3a_c, pos3_pval_c = find_vals(pos3s_c)

d_alpha_c = {'ABNR 1mo' : neg1a_c, 'ABNR 2mo' : neg2a_c, 'ABNR 3mo' : neg3a_c,
             'ABPR 1mo' : pos1a_c, 'ABPR 2mo' : pos2a_c, 'ABPR 3mo' : pos3a_c}
d_beta_c = {'ABNR 1mo' : neg1b_c, 'ABNR 2mo' : neg2b_c, 'ABNR 3mo' : neg3b_c,
            'ABPR 1mo' : pos1b_c, 'ABPR 2mo' : pos2b_c, 'ABPR 3mo' : pos3b_c}
d_pval_c = {'ABNR 1mo' : neg1_pval_c, 'ABNR 2mo' : neg2_pval_c, 'ABNR 3mo' : neg3_pval_c,
            'ABPR 1mo' : pos1_pval_c, 'ABPR 2mo' : pos2_pval_c, 'ABPR 3mo' : pos3_pval_c}

alphas_c = np.mean(pd.DataFrame(data = d_alpha_c))
betas_c = np.mean(pd.DataFrame(data = d_beta_c))
pvals_c = np.mean(pd.DataFrame(data = d_pval_c))

res_w_c = pd.concat([alphas_c.rename('Intercept'), betas_c.rename('Beta'), pvals_c.rename('p')], axis = 1)
res_w_c.style.set_caption("Fama Macbeth with Control Variables")

res_n_c.to_csv('summary statistic/reg_no_control.csv')
res_w_c.to_csv('summary statistic/reg_w_control.csv')