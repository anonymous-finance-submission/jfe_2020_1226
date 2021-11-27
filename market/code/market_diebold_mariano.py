"""
This file runs the Diebold-Mariano tests using market-level forecasts.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import seaborn as sns
sns.set_style('whitegrid')
pd.set_option('display.max_rows', 45)


PATH = PATH_TO_REPLICATION_PACKAGE + 'market/'

# output of the script 
INPUTPATH         = PATH + 'intermediate/'
TABLEPATH         = PATH + 'tables/'
FIGUREPATH        = PATH + 'figures/'
TABLE_PREFIX      = TABLEPATH + 'oos_tests_DM_'                 #suffix added below

# input data that is used in this script
INPUTFILE         = INPUTPATH + "market_forecasts.csv"


#%% ===========================================================================
# define functions
# =============================================================================

def diebold_mariano(d,outcome,fcst,benchmrk,lags) :
    ''' Test for H0: benchmrk sqd err <= fcst sqd err
        returns tstat, pvalue, and R-squared relative to the benchmark
        d = name of dataframe
        outcome = target variable
        fcst = forecast to be tested
        benchmrk = benchmark forecast
        lags = number of lags for Hansen-Hodrick std error
    '''
    df = d[[outcome,fcst,benchmrk]].dropna()
    u0 = df[outcome] - df[benchmrk]
    u1 = df[outcome] - df[fcst]
    
    ser = u0**2 - u1**2
    
    result = sm.OLS(ser,np.ones(len(ser))).fit(cov_type='HAC',cov_kwds={'maxlags':lags,'kernel':'uniform'})
    
    t = result.tvalues.item()
    p = result.pvalues.item()

    # If HH SEs are undefined:  
    if np.isnan(t):
        result = sm.OLS(ser,np.ones(len(ser))).fit(cov_type='HAC',cov_kwds={'maxlags':lags,'kernel':'bartlett'})
        t = result.tvalues.item()
        p = result.pvalues.item()
        
    p = p/2 if t>0 else 1-p/2
    R2 = 1 - (u1**2).sum() / (u0**2).sum()
    
    return R2, p, t


#%% ===========================================================================
# prepare the data
# =============================================================================
# read in the data
df_Fcsts = pd.read_csv(INPUTFILE).set_index(['bound', 'truncation', 'horizon', 'date'])

# minimum window size
min_window = 60

# model names and labels
fcst_models = ['tight', 'smean','rbound','rcombo_gw', 'rcombo_gwbound']
table_categories = fcst_models + ['(4) vs. (3)']
fcst_labels = ['(0) Tight Bound', '(1) Bound + Avg Slackness', '(2) OLS on Bound',
               '(3) Combination (GW)', '(4) Combination (GW + Bound)', '(5) (4) vs. (3)']
fcst_labels_LB=['(0) Tight Bound', '(1) Bound + Avg Slackness', '(2) max(OLS on Bound,Bound)', 
               '(3) max(Combination (GW),Zero)', '(4) max(Combination (GW + Bound),Bound)', '(5) (4) vs. (3)']

# empty dataframe, columns are bounds/horizons, rows are model/benchmark and stats
indx = pd.MultiIndex.from_product([table_categories,['R2','p-value','t-stat']])
colindx = pd.MultiIndex.from_product([['lb_m','lb_cylr'],['1','3','6','12']])
Table_NONE = pd.DataFrame(dtype=float,index=indx,columns=colindx)
Table_LB   = pd.DataFrame(dtype=float,index=indx,columns=colindx)


#%% ===========================================================================
# run the Diebold-Mariano tests
# =============================================================================

# loop over bounds and horizons to run the Diebold-Mariano tests
for b in ['lb_cylr','lb_m'] :
    for h in [1,3,6,12] :
        print('Bound is: ' + b+ ' and horizon is: ' + str(h))
        
        df_none = df_Fcsts.xs((b,'not_truncated',h), level=(0,1,2))
        df_lb = df_Fcsts.xs((b,'truncated',h), level=(0,1,2))
        
        bnd = b + '_' + str(h)
        ret = 'f_mktrf' + str(h)
        
        # number of lags for HH std errs
        lags = h
        
        # Diebold-Mariano comparing each model to expanding window market mean
        for fcst in fcst_models:
            Table_NONE[(b,str(h))].loc[fcst] = diebold_mariano(df_none,'outcome',fcst,'mkt',lags)
            Table_LB[(b,str(h))].loc[fcst]   = diebold_mariano(df_lb,'outcome',fcst,'mkt',lags)

        # Diebold-Mariano comparing (4) to (3)
        Table_NONE[(b,str(h))].loc['(4) vs. (3)'] = diebold_mariano(df_none,'outcome','rcombo_gwbound','rcombo_gw',lags)
        Table_LB[(b,str(h))].loc['(4) vs. (3)'] = diebold_mariano(df_lb,'outcome','rcombo_gwbound','rcombo_gw',lags)
        
                                          

#%% ===========================================================================
# Output formatted tables
# =============================================================================
  
def tex_wrapper(table, file_name, header_list):
    
    table.to_latex(file_name +'_pre.tex', escape=False, header = header_list)
    
    # make latex file a fragment (comment begin/end tabular)
    file_pre = open(file_name +'_pre.tex', 'r')
    file_out = open(file_name +'.tex', 'w')
    
    checkWords = ('\\begin', '\end','label', '\\toprule')
    repWords = ('%\\begin', '%\end','%label', '%\\toprule')
    
    for line in file_pre:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        file_out.write(line)
    
    file_pre.close()
    file_out.close()    
    
    os.remove(file_name +'_pre.tex') 
       
# Ouput RMSE tables to latex (R^2 only with stars for significance)
for b in ['lb_cylr','lb_m'] :
    for i in ['NONE','LB']:
        if i=='NONE':
            suffix = str(min_window) + '_' + b
            Table=Table_NONE.copy()
            indx = pd.MultiIndex.from_product([fcst_labels,['$R^2$']])  
        elif i=='LB':
            suffix = str(min_window) + '_' + b + '_LB'
            Table=Table_LB.copy()
            indx = pd.MultiIndex.from_product([fcst_labels_LB,['$R^2$']])  
        else:
            pass

        # convert to string table to facilitate latex formatting
        st = Table[Table.index.isin(['R2','p-value'], level=1)][[(b,'1'),(b,'3'),(b,'6'),(b,'12')]]
        st.columns = st.columns.droplevel(0)      
        
        # formatted string
        st2 = pd.DataFrame(dtype=str,index=st.index,columns=st.columns)           
        
        # add results to table in strings with formatting/stars
        tuples_r2 = [(x, 'R2') for x in table_categories]
        tuples_p  = [(x, 'p-value') for x in table_categories]
        for h in [1,3,6,12]:
            coef_string = st[str(h)].loc[tuples_r2].apply(lambda x: f'{x:.3f}').values      
            pvals = st[str(h)].loc[tuples_p]
            st2[str(h)].loc[tuples_r2]=np.where(pvals<=0.01, coef_string + '$^{***}$', 
                                       np.where(pvals<=0.05, coef_string + '$^{**}$',
                                       np.where(pvals <=0.1, coef_string + '$^{*}$', coef_string)))
            
            r2_num = st[str(h)].loc[tuples_r2]
            r2_str = st2[str(h)].loc[tuples_r2]
            st2[str(h)].loc[tuples_r2]= np.where(r2_num > 0.0    ,  r2_str + '\cellcolor{lightgray}', r2_str)
        
        st2 = st2.loc[tuples_r2]
        st2.index=indx
        st2=st2.droplevel(level=1)
        my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in st2.columns]
        tex_wrapper(table=st2, file_name=TABLE_PREFIX + suffix, header_list=my_header)      
        
        
        tab_p = st.loc[tuples_p]*100
        tab_p.index=indx
        tab_p=tab_p.droplevel(level=1)
        st_p=tab_p.round(1)
        tex_wrapper(table=st_p, file_name=TABLE_PREFIX + 'pvals_' + suffix, header_list=my_header)      

