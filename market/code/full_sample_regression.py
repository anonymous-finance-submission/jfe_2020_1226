from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from scipy.stats import t

# paths
PATH = PATH_TO_REPLICATION_PACKAGE + 'market/'
CODE_DIR         = PATH + 'code/'
INPUT_DIR        = PATH + 'input/'
INTERMEDIATE_DIR = PATH + 'intermediate/'   
TABLES_DIR       = PATH + 'tables/'         
FIGURES_DIR      = PATH + 'figures/'        

# Input data that is used in this script
DS_DAILY              = INTERMEDIATE_DIR + 'ds_mkt_daily.csv'              
DS_MONTHLY            = INTERMEDIATE_DIR + 'ds_mkt_monthly.csv'            
BS_RES                = INTERMEDIATE_DIR + 'bootstrap_fs_regs.csv'

# Output of the script 
TABLE_MKT_ON_BND      = TABLES_DIR + 'fs_reg_mkt_bnd_ah_'     # suffix for bound identifier added below
TABLE_MKT_ON_BND_GW   = TABLES_DIR + 'fs_reg_mkt_bnd_gw_ah_'  # suffix for bound identifier added below

#%% ===========================================================================
# Function for cleaning up tex output
# =============================================================================

# output and clean-up tex tables
def tex_wrapper(table, file_name, header_list):
    table.to_latex(file_name +'_pre.tex', escape=False, header = header_list)
    
    # make latex file a fragment (comment begin/end tabular)
    file_pre = open(file_name +'_pre.tex', 'r')
    file_out = open(file_name +'.tex', 'w')
    checkWords = ('\\begin', '\end','label', '\\toprule', 'Estimate', 'Std Error')
    repWords = ('%\\begin', '%\end','%label', '%\\toprule', '', '')
    for line in file_pre:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        file_out.write(line)
    file_pre.close()
    file_out.close()    
    os.remove(file_name +'_pre.tex')

#%% ===========================================================================
# Univariate regressions: excess returns on constant + bounds (daily frequency) 
# Estimate SEs using Hansen-Hodrick and then apply Amihud-Hurvich correction
# =============================================================================

# load the daily bounds and forward returns data
ds = pd.read_csv(DS_DAILY)

# bs_pvals = pd.read_csv(BS_PVALS)
# bs_pvals = bs_pvals.set_index(['bound','horizon','frequency'])
sim_res = pd.read_csv(BS_RES, dtype={'horizon':int, 'sim_num':int})
numSims = sim_res['sim_num'].max() - sim_res['sim_num'].min() + 1
sim_res = sim_res.set_index(['bound', 'horizon', 'frequency', 'sim_num'])

# Run AR(1) regressions for the bound
def vc_create(ts):   
    X = sm.add_constant(ts.shift(1))
    y = ts
    tsreg = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

    rho     = tsreg.params.loc[ts.name]
    numobs  = tsreg.nobs
    rhoc    = rho + (1+3*rho)/numobs + 3*(1 + 3*rho)/(numobs **2)
    
    vc = ts - (tsreg.params.loc['const'] + rhoc * ts.shift(1))
    
    var_rho = tsreg.bse.loc[ts.name]**2  
    var_rhoc= ((1+3/numobs + 9/(numobs **2))**2)*var_rho
    
    # Demean vc (should be close to zero already)
    return vc, var_rhoc

for b in ['m','cylr']:
    for h in [1,3,6,12]:
        bnd='lb_' + b + '_' + str(h)
        ds['vc_'+bnd], ds['var_rhoc_' +bnd] = vc_create(ds[bnd])

# Return on Bound - predictive OLS with augmented residuals from AR regressions
for b in ['m','cylr']:
    Xvars  = ['Constant', 'Bound']
    indx   = pd.MultiIndex.from_product([Xvars,['Estimate','Std Error']])
    colindx= ['1','3','6','12']
    Table  = pd.DataFrame(dtype=float,index=indx,columns=colindx)
    indx_foot = pd.MultiIndex.from_tuples([('N',''),('Adj. $R^2$',''), ('Bootstrap $p$-value','')])
    Table  = Table.append(pd.DataFrame(dtype=float,index=indx_foot,columns=colindx))   #numeric version
    Table2 = pd.DataFrame(dtype=str,index=Table.index,columns=Table.columns)           #string, formatted version
    for h in [1,3,6,12]:
        bnd   ='lb_' + b + '_' + str(h)
        fret  ='f_mktrf' + str(h)        
        xlist = [bnd]
        vclist= ['vc_'+j for j in xlist]
        xlist = xlist + vclist
        
        # Run augmented regression
        X = ds[xlist]
        X = sm.add_constant(X)
        results = sm.OLS(ds[fret],X, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':21*h, 'kernel': 'uniform'})

        # Adjust SEs following Amihud/Hurvich
        numparms = len(vclist)
        se_df = pd.DataFrame(dtype=float,index=[bnd],columns=['phic', 'var_rhoc', 'se_betac', 'adj_se_betac', 'betac', 'adj_t', 'unadj_t', 'adj_p', 'unadj_p'])
        se_df['phic'] = results.params[-numparms:].values
        se_df['var_rhoc'] = ds[['var_rhoc_' + i for i in [bnd]]].mean().values
        se_df['se_betac'] = results.bse[1:numparms+1].values
        se_df['adj_se_betac'] = np.sqrt((se_df.phic**2) * se_df.var_rhoc + (se_df.se_betac**2))
        se_df['betac']   = results.params[1:numparms+1].values
        se_df['adj_t']   = se_df.betac.divide(se_df.adj_se_betac)
        se_df['unadj_t'] = results.tvalues[1:numparms+1].values
        se_df['adj_p']   = se_df.adj_t.apply(lambda x: 2*(1-t.cdf(np.abs(x),results.df_resid))).values
        se_df['unadj_p'] = results.pvalues[1:numparms+1].values

        # add results to table
        tuples = [(a, 'Estimate') for a in Xvars]
        Table[str(h)].loc[tuples] = results.params[0:len(Xvars)].values
        
        Table[str(h)].loc[('Constant', 'Std Error')] = results.bse[0]
        tuples = [(a, 'Std Error') for a in Xvars[1:]]
        Table[str(h)].loc[tuples] = se_df.adj_se_betac.values
        Table[str(h)].loc[('N','')]         = results.nobs
        Table[str(h)].loc[('Adj. $R^2$','')]= results.rsquared
        
        # calculate p-values from simulation results; we need to run the full
        # sample regression without correction terms
        y = ds[fret]
        x = sm.add_constant(ds[bnd])
        results_no_ah = sm.OLS(y, x, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':21*h, 'kernel': 'uniform'})
        data = sim_res.loc[('lb_{}'.format(b), h,'daily'), 't(b)']
        pval = (data >= results_no_ah.tvalues['lb_{}_{}'.format(b,h)]).sum()/numSims*100
        Table[str(h)].loc[('Bootstrap $p$-value','')] = pval
        
        # add results to table in strings with formatting/stars
        coef_string = str(np.round(Table[str(h)].loc[('Constant', 'Estimate')],2))
        Table2[str(h)].loc[('Constant', 'Estimate')] = np.where(results.pvalues[0] <= 0.01, coef_string + '$^{***}$', 
                                np.where(results.pvalues[0] <= 0.05, coef_string + '$^{**}$',  
                                np.where(results.pvalues[0] <= 0.1,  coef_string + '$^{*}$', coef_string)))        
        
        # Other regressors with adjusted SEs/pvals
        tuples = [(x, 'Estimate') for x in Xvars[1:]]
        coef_string = Table[str(h)].loc[tuples].apply(lambda x: f'{x:.2f}').values      
        Table2[str(h)].loc[tuples] = np.where(se_df.adj_p <= 0.01, coef_string + '$^{***}$', 
                                np.where(se_df.adj_p <= 0.05, coef_string + '$^{**}$',  
                                np.where(se_df.adj_p <= 0.1,  coef_string + '$^{*}$', coef_string)))
        tuples = [(x, 'Std Error') for x in Xvars]
        Table2[str(h)].loc[tuples] = Table[str(h)].loc[tuples].apply(lambda x: f'({x:.2f})').values


    Table2.loc[('N','')]           = Table.loc[('N','')].apply(lambda x: str(int(x))).values
    Table2.loc[('Adj. $R^2$','')]  = Table.loc[('Adj. $R^2$','')].apply(lambda x: f'{x:.4f}').values
    Table2.loc[('Bootstrap $p$-value','')]  = Table.loc[('Bootstrap $p$-value','')].apply(lambda x: f'{x:.2f}').values
    print(Table2 , '\n')
        
    my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in Table2.columns]
    tex_wrapper(table=Table2, file_name=TABLE_MKT_ON_BND + b, header_list=my_header)  
    
#%% ===========================================================================
# Returns on Bound + GW (monthly frequency)
# Estimate SEs using Hansen-Hodrick and then apply Amihud-Hurvich correction
# =============================================================================

# load the monthly bounds and forward returns data
ds = pd.read_csv(DS_MONTHLY)

# Standardize regressors
gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']
for i in gw_vars:
    ds[i+'_sd'] = (ds[i] - ds[i].mean())/ ds[i].std()

for b in ['m','cylr']:
    for h in [1,3,6,12]:
        bnd='lb_' + b + '_' + str(h)
        ds[bnd+'_sd'] = (ds[bnd] - ds[bnd].mean())/ ds[bnd].std()
     
# Run AR(1) regressions for each GW predictor and bound (after standardizing at monthly frequency)
def vc_create(ts):   
    X = sm.add_constant(ts.shift(1))
    y = ts
    tsreg = sm.OLS(y, X, missing='drop').fit(cov_type='HC1')

    rho     = tsreg.params.loc[ts.name]
    numobs  = tsreg.nobs
    rhoc    = rho + (1+3*rho)/numobs + 3*(1 + 3*rho)/(numobs **2)
    
    vc = ts - (tsreg.params.loc['const'] + rhoc * ts.shift(1))
    
    var_rho = tsreg.bse.loc[ts.name]**2  
    var_rhoc= ((1+3/numobs + 9/(numobs **2))**2)*var_rho
    
    #Demean vc (should be close to zero already)
    return vc - vc.mean(), var_rhoc

for i in gw_vars:
    ds['vc_'+i+ '_sd'], ds['var_rhoc_' +i] = vc_create(ds[i+'_sd'])
    
for b in ['m','cylr']:
    for h in [1,3,6,12]:
        bnd='lb_' + b + '_' + str(h)
        ds['vc_'+bnd+ '_sd'], ds['var_rhoc_' +bnd] = vc_create(ds[bnd+'_sd'])

# Regression on GW - predictive OLS with augmented residuals from AR regressions
for b in ['m','cylr']:
    Xvars  = ['Constant', 'Bound', 'Div Price Ratio' , 'Earnings Price Ratio',  'Book-to-Market Ratio', 'T-bill Rate', 'Term Spread',  'Credit Spread', 'Stock Variance', 'Net Eq Issuance', 'Inflation']
    indx   = pd.MultiIndex.from_product([Xvars,['Estimate','Std Error']])
    colindx= ['1','3','6','12']
    Table  = pd.DataFrame(dtype=float,index=indx,columns=colindx)
    indx_foot = pd.MultiIndex.from_tuples([('N',''),('Adj. $R^2$',''),('Bootstrap $p$-value','')])
    Table  = Table.append(pd.DataFrame(dtype=float,index=indx_foot,columns=colindx))   #numeric version
    Table2 = pd.DataFrame(dtype=str,index=Table.index,columns=Table.columns)           #string, formatted version
    for h in [1,3,6,12]:
        bnd   ='lb_' + b + '_' + str(h)
        fret  ='f_mktrf' + str(h)        
        xlist = [j +'_sd' for j in [bnd]+gw_vars]
        vclist= ['vc_'+j for j in xlist]
        xlist = xlist + vclist
        
        # Run augmented regression
        X = ds[xlist]
        X = sm.add_constant(X)
        results = sm.OLS(ds[fret], X, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':h, 'kernel': 'uniform'})

        #Adjust SEs following Amihud/Hurvich
        numparms = len(vclist)
        se_df = pd.DataFrame(dtype=float,index=[bnd]+gw_vars,columns=['phic', 'var_rhoc', 'se_betac', 'adj_se_betac', 'betac', 'adj_t', 'unadj_t', 'adj_p', 'unadj_p'])
        se_df['phic'] = results.params[-numparms:].values
        se_df['var_rhoc'] = ds[['var_rhoc_' + i for i in [bnd]+gw_vars]].mean().values
        se_df['se_betac'] = results.bse[1:numparms+1].values
        se_df['adj_se_betac'] = np.sqrt((se_df.phic**2) * se_df.var_rhoc + (se_df.se_betac**2))
        se_df['betac']   = results.params[1:numparms+1].values
        se_df['adj_t']   = se_df.betac.divide(se_df.adj_se_betac)
        se_df['unadj_t'] = results.tvalues[1:numparms+1].values
        se_df['adj_p']   = se_df.adj_t.apply(lambda x: 2*(1-t.cdf(np.abs(x),results.df_resid))).values
        se_df['unadj_p'] = results.pvalues[1:numparms+1].values

        # add results to table
        tuples = [(a, 'Estimate') for a in Xvars]
        Table[str(h)].loc[tuples] = results.params[0:len(Xvars)].values
        
        Table[str(h)].loc[('Constant', 'Std Error')] = results.bse[0]
        tuples = [(a, 'Std Error') for a in Xvars[1:]]
        Table[str(h)].loc[tuples] = se_df.adj_se_betac.values
        Table[str(h)].loc[('N','')]         = results.nobs
        Table[str(h)].loc[('Adj. $R^2$','')]= results.rsquared
        
        # calculate p-values from simulation results
        y = ds[fret]
        x = sm.add_constant(ds[[j for j in [bnd]+gw_vars]])
        results_no_ah = sm.OLS(y, x, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':h, 'kernel': 'uniform'})
        data = sim_res.loc[('lb_{}'.format(b), h,'monthly'), 't(b)']
        pval = (data >= results_no_ah.tvalues['lb_{}_{}'.format(b,h)]).sum()/numSims*100
        Table[str(h)].loc[('Bootstrap $p$-value','')] = pval    
        
        # add results to table in strings with formatting/stars
        # Constant
        coef_string = str(np.round(Table[str(h)].loc[('Constant', 'Estimate')],1))
        Table2[str(h)].loc[('Constant', 'Estimate')] = np.where(results.pvalues[0] <= 0.01, coef_string + '$^{***}$', 
                                np.where(results.pvalues[0] <= 0.05, coef_string + '$^{**}$',  
                                np.where(results.pvalues[0] <= 0.1,  coef_string + '$^{*}$', coef_string)))        
        
        # Other regressors with adjusted SEs/pvals
        tuples = [(x, 'Estimate') for x in Xvars[1:]]
        coef_string = Table[str(h)].loc[tuples].apply(lambda x: f'{x:.1f}').values      
        Table2[str(h)].loc[tuples] = np.where(se_df.adj_p <= 0.01, coef_string + '$^{***}$', 
                                np.where(se_df.adj_p <= 0.05, coef_string + '$^{**}$',  
                                np.where(se_df.adj_p <= 0.1,  coef_string + '$^{*}$', coef_string)))
        tuples = [(x, 'Std Error') for x in Xvars]
        Table2[str(h)].loc[tuples] = Table[str(h)].loc[tuples].apply(lambda x: f'({x:.1f})').values


    Table2.loc[('N','')]           = Table.loc[('N','')].apply(lambda x: str(int(x))).values
    Table2.loc[('Adj. $R^2$','')]  = Table.loc[('Adj. $R^2$','')].apply(lambda x: f'{x:.2f}').values
    Table2.loc[('Bootstrap $p$-value','')] = Table.loc[('Bootstrap $p$-value','')].apply(lambda x: f'{x:.2f}').values
    print(Table2 , '\n')
     
    my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in Table2.columns]
    tex_wrapper(table=Table2, file_name=TABLE_MKT_ON_BND_GW + b, header_list=my_header)
    