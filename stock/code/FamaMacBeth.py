from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os


path       = PATH_TO_REPLICATION_PACKAGE + 'stock/'
INPUT1     = path + 'intermediate/ds_stock_monthly.csv'
INPUT2     = path + 'input/rpsdata_rfs.csv'
OUTPUT1    = path + 'tables/FM_uni.tex'
OUTPUT2    = path + 'tables/FM_multi.tex'

os.chdir(path + 'code/')
import sigtable

def reg(d,y,x) :  
        X = sm.add_constant(d[x])
        res = sm.OLS(d[y],X).fit()
        return res.params

#%% ===========================================================================
# read bound and return data
# =============================================================================

df = pd.read_csv(INPUT1)
for col in ['permno','date'] :
    df[col] = df[col].astype(int)
df.date = pd.to_datetime(df.date.astype(str),format='%Y%m').dt.to_period('M')
df.set_index(['date','permno'],inplace=True)

cols = [x + str(h) for x in ['lb_kt_','lb_mw_','f_xret'] for h in [1,3,6,12]]
df = df[cols+['delta']]


#%% ===========================================================================
# add winsorized and standardized characteristics
# =============================================================================

CHARS = ['mve','bm','agr','operprof','mom12m']      
df2 = pd.read_csv(INPUT2, dtype={'permno':'int', 'date':'period[M]'})
df2.set_index(['date','permno'],inplace=True)


def stdize(d) : 
    d2 = d.clip(d.quantile(0.01),d.quantile(0.99),axis=1)
    return (d2 - d2.mean()) / d2.std()

df = df.join(df2)
df[CHARS] = df[CHARS].groupby('date').apply(stdize)

#%% ===========================================================================
# run univariate FM regressions, only keep slope coefficients
# =============================================================================

idx = pd.MultiIndex.from_product((['Martin-Wagner (All)','Martin-Wagner (Conservative)'],['slope','stderr','pvalue'])) 
table = pd.DataFrame(dtype=float,index=idx,columns=['1','3','6','12'])

# All stocks
for h in ['1','3','6','12'] :
    y = 'f_xret'+h
    x = 'lb_mw_'+h
    d = df[[y,x]].dropna()
    coefs = d.groupby('date').apply(lambda d: reg(d,y,x))
    table.loc[('Martin-Wagner (All)','slope'),h] = coefs[x].mean()
    X = sm.add_constant(coefs[x])
    X = X['const']
    result = sm.OLS(coefs[x],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc[('Martin-Wagner (All)','stderr'),h] = np.sqrt(result.cov_params()['const']['const'])
    table.loc[('Martin-Wagner (All)','pvalue'),h] = result.pvalues.item()

# Conservative stocks
df2 = df[df.delta<=3]
for h in ['1','3','6','12'] :
    y = 'f_xret'+h
    x = 'lb_mw_'+h
    d = df2[[y,x]].dropna()
    coefs = d.groupby('date').apply(lambda d: reg(d,y,x))
    table.loc[('Martin-Wagner (Conservative)','slope'),h] = coefs[x].mean()
    X = sm.add_constant(coefs[x])
    X = X['const']
    result = sm.OLS(coefs[x],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc[('Martin-Wagner (Conservative)','stderr'),h] = np.sqrt(result.cov_params()['const']['const'])
    table.loc[('Martin-Wagner (Conservative)','pvalue'),h] = result.pvalues.item()
    
table = table.to_latex()
table = sigtable.sigtable(table,'slope','stderr','pvalue')
with open(OUTPUT1, 'w') as f: 
    f.write(table)


#%% ===========================================================================
# run multivariate FM regressions, only keep slope coefficients
# =============================================================================
   
idx = pd.MultiIndex.from_product((['Martin-Wagner (All)','Martin-Wagner (Conservative)'],['slope','stderr','pvalue'])) 
table = pd.DataFrame(dtype=float,index=idx,columns=['1','3','6','12'])

# All stocks
for h in ['1','3','6','12'] :
    y = 'f_xret'+h
    x = 'lb_mw_'+h
    d = df[[y,x]+CHARS].dropna()
    coefs = d.groupby('date').apply(lambda d: reg(d,y,[x]+CHARS))
    table.loc[('Martin-Wagner (All)','slope'),h] = coefs[x].mean()
    X = sm.add_constant(coefs[x])
    X = X['const']
    result = sm.OLS(coefs[x],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc[('Martin-Wagner (All)','stderr'),h] = np.sqrt(result.cov_params()['const']['const'])
    table.loc[('Martin-Wagner (All)','pvalue'),h] = result.pvalues.item()

# Conservative stocks
df2 = df[df.delta<=3]
for h in ['1','3','6','12'] :
    y = 'f_xret'+h
    x = 'lb_mw_'+h
    d = df2[[y,x]+CHARS].dropna()
    coefs = d.groupby('date').apply(lambda d: reg(d,y,[x]+CHARS))
    table.loc[('Martin-Wagner (Conservative)','slope'),h] = coefs[x].mean()
    X = sm.add_constant(coefs[x])
    X = X['const']
    result = sm.OLS(coefs[x],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc[('Martin-Wagner (Conservative)','stderr'),h] = np.sqrt(result.cov_params()['const']['const'])
    table.loc[('Martin-Wagner (Conservative)','pvalue'),h] = result.pvalues.item()

table = table.to_latex()
table = sigtable.sigtable(table,'slope','stderr','pvalue')
with open(OUTPUT2, 'w') as f: f.write(table)



