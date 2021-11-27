"""
This file generates market-level forecasts using various models.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import wrds
import statsmodels.api as sm
import seaborn as sns
sns.set_style('whitegrid')
pd.set_option('display.max_rows', 45)

# input data that is used in this script
PATH         = PATH_TO_REPLICATION_PACKAGE + 'market/'
DS_MONTHLY   = PATH + 'intermediate/ds_mkt_monthly.csv'

# output of the script 
OUTPUTFILE   = PATH + 'intermediate/market_forecasts.csv'

#%% 1 =========================================================================
# Prepare monthly dataset with benchmark average historical excess return
# =============================================================================

# load the monthly bounds and forward returns data
df = pd.read_csv(DS_MONTHLY, index_col='date', parse_dates=['date'])

# load monthly fama-french data from WRDS
conn=wrds.Connection()
ff_monthly_wrds = conn.raw_sql("select date, mktrf, dateff from ff.factors_monthly") 
conn.close()

ff_monthly_wrds['date'] = pd.to_datetime(ff_monthly_wrds.date)
ff_monthly_wrds = ff_monthly_wrds.set_index('date')  

# generate expanding mean of market excess return, convert to percent, annualize
ff_monthly_wrds['roll_avg_mktrf'] = ff_monthly_wrds.mktrf.expanding().mean()*1200

# Merge to monthly panel
df['date'] = df.index.copy()
df = pd.merge(df, ff_monthly_wrds.roll_avg_mktrf, how='inner', left_on=[df.index.year, df.index.month], right_on=[ff_monthly_wrds.index.year, ff_monthly_wrds.index.month])
df.set_index('date', inplace=True)
df = df.drop(columns=['key_0', 'key_1'])
df['mkt_mean'] = df.roll_avg_mktrf
gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']


#%% 2 =========================================================================
# Define forecasting functions
# =============================================================================
def expanding_ols_fcsts(d,y,x,min_window,h) :
    ''' d = dataframe containing variables 
        y = name of dependent variable 
        x = name of independent variable or list of independent variables
        min_obs = minimum window size 
        h = forward return horizon
        returns a dataframe of regression coefficients with the same index as d and columns=['const']+x
    '''
    x = x if isinstance(x,list) else [x]
    df = d[[y]+x].dropna()
    Y = df[y]
    X = sm.add_constant(df[x])
    fcsts = pd.Series(dtype=float,index=df.index)
    for i in range(min_window,d.shape[0]-h+1) :    
        
        # eliminate data following estimation window to ensure true OOS
        params = sm.OLS(Y.iloc[:i],X.iloc[:i],missing='drop').fit().params
        fcsts.iloc[i+h-1] = (X.iloc[i+h-1]*params).sum()
    return fcsts

#%% 3 =========================================================================
# Calculate OOS Forecasts
# =============================================================================

''' Expanding-window forecasts and Diebold-Mariano tests 
    (No truncation)
    (0) tight:          bound is the forecast (assume zero slackness)
    (1) smean:          bound + average past slackness
    (2)	rbound:	        OLS Ret on Bound
    
    (3) rcombo_gw	    Avg of OLS Ret on Univariate GW
    (4) rcombo_gwbound  Avg of (3) and (1)
    (5) (4) vs. (3)
    
    compare each to benchmark
    where benchmark is expanding window market mean
'''

# define minimum window size
min_window = 60

fcst_models = ['tight','smean','rbound','rcombo_gw','rcombo_gwbound']

# loop over bounds and horizons to calculate the forecasts
bounds = ['lb_cylr','lb_m']
truncations = ['not_truncated', 'truncated']
horizons = [1,3,6,12]
dates = df.index
idx = pd.MultiIndex.from_product((bounds, truncations, horizons, dates))
df_Fcsts = pd.DataFrame(index=idx, dtype=float)
df_Fcsts.index.names = ['bound', 'truncation', 'horizon', 'date']

for b in ['lb_cylr','lb_m'] :
    for h in [1,3,6,12] :
        print('Bound is: ' + b+ ' and horizon is: ' + str(h))
      
        bnd = b + '_' + str(h)
        ret = 'f_mktrf' + str(h)
        
        # keep forward return, bound, expanding-window market mean, and GW variables
        d = df[[ret,bnd,'mkt_mean']+gw_vars].copy()
        d.dropna(inplace=True)

        # define slackness
        d['slack'] = d[ret] - d[bnd]
        
        # compute forecast based on past average slackness or tight forecast
        Fcsts = d['slack'].expanding(min_periods=min_window).mean().shift(h)   #shift h is to ensure that this is truly out-of-sample
        Fcsts = pd.DataFrame(Fcsts,index=d.index)
        Fcsts= Fcsts.rename(columns={'slack':'smean'})
        Fcsts['smean'] = Fcsts['smean'] + d[bnd] 
        Fcsts['tight'] = np.where(~np.isnan(Fcsts.smean), 0.0, np.nan )  + d[bnd]     #tight bound -> slackness of zero

      
        # add OLS forecasts based on return prediction
        Fcsts['rbound']      = expanding_ols_fcsts(d,ret,bnd,min_window,h)
        Fcsts['rgw']         = expanding_ols_fcsts(d,ret,gw_vars,min_window,h)
        Fcsts['rgwbound']    = expanding_ols_fcsts(d,ret,gw_vars+[bnd],min_window,h)
        
        # Combination forecasts - use univariate forecasts of return
        varlist=[]
        for v in gw_vars:
            Fcsts['runivariate_'+v] = expanding_ols_fcsts(d,ret,v,min_window,h)
            varlist.append('runivariate_'+v)
        Fcsts['rcombo_gw'] = Fcsts[varlist].mean(axis=1)
        Fcsts.drop(columns = varlist, inplace = True)
        
        # combo of GW vars + bound
        Fcsts['rcombo_gwbound'] = 0.5*Fcsts['smean'] + 0.5*Fcsts['rcombo_gw']
    
        # add forward return and expanding-window market mean
        Fcsts['outcome'] = d[ret]
        Fcsts['mkt'] = d['mkt_mean']        
 
        # Truncate below at LB
        Fcsts['zero'] = np.where(~np.isnan(Fcsts.smean), 0.0, np.nan )
        Fcsts_LB = Fcsts.copy()
        for fcst in fcst_models:
            if fcst in ['rgw', 'rcombo_gw']:
                print('Fcst '+fcst +' truncated at zero')
                Fcsts_LB[fcst] = np.where(Fcsts_LB[fcst]<Fcsts_LB['zero'],Fcsts_LB['zero'],Fcsts_LB[fcst])
            else:
                print('Fcst '+fcst +' truncated at bound')
                Fcsts_LB[fcst] = np.where(Fcsts_LB[fcst]<d[bnd],d[bnd],Fcsts_LB[fcst])

        for col in Fcsts.columns:
            
            # save the forecasts
            temp = np.zeros(df.shape[0])
            temp[:Fcsts.shape[0]] = Fcsts[col]
            temp[Fcsts.shape[0]:] = np.nan
            df_Fcsts.loc[(b,'not_truncated',h), col] = temp
            
            temp = np.zeros(df.shape[0])
            temp[:Fcsts_LB.shape[0]] = Fcsts_LB[col]
            temp[Fcsts_LB.shape[0]:] = np.nan
            df_Fcsts.loc[(b,'truncated',h), col] = temp
 
df_Fcsts.to_csv(OUTPUTFILE)
