from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from arch.bootstrap import CircularBlockBootstrap
from numpy.random import RandomState

path    = PATH_TO_REPLICATION_PACKAGE + 'stock/'
INPUT1  = path + 'intermediate/ds_stock_monthly.csv'
INPUT2  = path + 'input/rpsdata_rfs.csv'

OUTPUT1 = path + 'tables/panel_nofixed_uni.tex'
OUTPUT2 = path + 'tables/panel_fixed_uni.tex'
OUTPUT3 = path + 'tables/panel_nofixed_multi.tex'
OUTPUT4 = path + 'tables/panel_fixed_multi.tex'

import os
os.chdir(path + 'code/')
import sigtable

# models & bounds
idx = pd.MultiIndex.from_product((['NoUni','FixUni','NoMulti','FixMulti'],['mw','kt']))

# number of bootstrapped samples to draw
num_samples = 1000     

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


#%%============================================================================
# add winsorized and standardized characteristics
# =============================================================================

chars = ['mve','bm','agr','operprof','mom12m']      
df2 = pd.read_csv(INPUT2, dtype={'permno':'int', 'date':'period[M]'})
df2.set_index(['date','permno'],inplace=True)

def stdize(d) : 
    d2 = d.clip(d.quantile(0.01),d.quantile(0.99),axis=1)
    return (d2 - d2.mean()) / d2.std()

df = df.join(df2)
df[chars] = df[chars].groupby('date').apply(stdize)


#%% ===========================================================================
# function to compute slope coefficients for models/bounds from a 
# bootstrapped sample or the original data
# =============================================================================

def slopes(d,h) :
    
    out = pd.Series(dtype=float,index=idx)
    ret = 'f_xret'+h
    
    # Martin-Wagner slopes
    bd = 'lb_mw_'+h
    
    #      no fixed effects, univariate
    d1 = d.dropna(subset=[ret,bd]).copy()
    X = sm.add_constant(d1[bd])
    out.loc['NoUni','mw'] = sm.OLS(d1[ret],X).fit().params[bd]
    
    #      fixed effects, univariate
    d1[bd] = d1[bd] - d1[bd].mean()
    d1[ret] = d1.groupby('permno')[ret].apply(lambda x: x-x.mean())
    out.loc['FixUni','mw'] = sm.OLS(d1[ret],d1[bd]).fit().params[bd]
    
    #      no fixed effects, multivariate
    d1 = d.dropna(subset=[ret,bd]+chars).copy()
    X = sm.add_constant(d1[[bd]+chars])
    out.loc['NoMulti','mw'] = sm.OLS(d1[ret],X).fit().params[bd]
    
    #      fixed effects, multivariate
    for x in [bd]+chars : d1[x] = d1[x] - d1[x].mean()
    d1[ret] = d1.groupby('permno')[ret].apply(lambda x: x-x.mean())
    out.loc['FixMulti','mw'] = sm.OLS(d1[ret],d1[[bd]+chars]).fit().params[bd]
    
    # Kadan-Tang slopes
    bd = 'lb_kt_'+h
    d2 = d[d.delta<=3].copy()     # only Conservative group
    
    #      no fixed effects, univariate
    d1 = d2.dropna(subset=[ret,bd]).copy()
    X = sm.add_constant(d1[bd])
    out.loc['NoUni','kt'] = sm.OLS(d1[ret],X).fit().params[bd]
    
    #      fixed effects, univariate
    d1[bd] = d1[bd] - d1[bd].mean()
    d1[ret] = d1.groupby('permno')[ret].apply(lambda x: x-x.mean())
    out.loc['FixUni','kt'] = sm.OLS(d1[ret],d1[bd]).fit().params[bd]
    
    #      no fixed effects, multivariate
    d1 = d2.dropna(subset=[ret,bd]+chars).copy()
    X = sm.add_constant(d1[[bd]+chars])
    out.loc['NoMulti','kt'] = sm.OLS(d1[ret],X).fit().params[bd]
    
    #      fixed effects, multivariate
    for x in [bd]+chars : d1[x] = d1[x] - d1[x].mean()
    d1[ret] = d1.groupby('permno')[ret].apply(lambda x: x-x.mean())
    out.loc['FixMulti','kt'] = sm.OLS(d1[ret],d1[[bd]+chars]).fit().params[bd]
    
    return out


# =============================================================================
# function to compute slope coefficients for models/bounds for 
# n bootstrapped samples
# =============================================================================

def bootstrap(d,h,n,rs) :
    print('h = ' + h)
    out = pd.DataFrame(dtype=float,index=range(n),columns=idx)
    dates = d.date.unique()              
    dates = np.sort(dates)                              
    bs = CircularBlockBootstrap(int(h),dates, random_state=rs)                
    for i, date_list in enumerate(bs.bootstrap(n)) : 
        if i % 100 == 0 : print(i)
        date_list = pd.DataFrame(date_list[0][0],columns=['date'])
        sample = date_list.merge(d,on='date')
        out.loc[i] = slopes(sample,h)
    return out


#%% ===========================================================================
#  slope coefficients for models/bounds from the original data
# =============================================================================

coefs = pd.DataFrame(dtype=float,index=idx,columns=['1','3','6','12'])
for h in ['1','3','6','12'] :
    coefs[h] = slopes(df,h)
    
    
#%% ===========================================================================
#  bootstrapped std errs for models/bounds
# =============================================================================

stderrs = pd.DataFrame(dtype=float,index=idx,columns=['1','3','6','12'])
for h in ['1','3','6','12'] :
    rs = RandomState(int(h)+1000)
    sims = bootstrap(df.reset_index(),h,num_samples,rs)
    stderrs[h] = sims.std()
          
        
#%% ===========================================================================
# p values for models/bounds
# =============================================================================

pvals = pd.DataFrame(dtype=float,index=idx,columns=['1','3','6','12'])
for row in pvals.index :
    for col in pvals.columns :
        q = norm.cdf(coefs.loc[row,col]/stderrs.loc[row,col])
        pvals.loc[row,col] = 2*np.min((q,1-q))
        
        
#%% ===========================================================================
#  create latex tables - one for each model, because each model will be a panel
# =============================================================================

indx = pd.MultiIndex.from_product((['Martin-Wagner (All)','Kadan-Tang (Conservative)'],['coef','stderr','pvalue']))
table = pd.DataFrame(dtype=float,index=indx,columns=['1','3','6','12'])
for h in ['1','3','6','12'] :
    table.loc[('Martin-Wagner (All)','coef'),h] = coefs.loc[('NoUni','mw'),h]
    table.loc[('Martin-Wagner (All)','stderr'),h] = stderrs.loc[('NoUni','mw'),h]
    table.loc[('Martin-Wagner (All)','pvalue'),h] = pvals.loc[('NoUni','mw'),h]
    table.loc[('Kadan-Tang (Conservative)','coef'),h] = coefs.loc[('NoUni','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','stderr'),h] = stderrs.loc[('NoUni','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','pvalue'),h] = pvals.loc[('NoUni','kt'),h]
table1 = table.to_latex()
    
table = pd.DataFrame(dtype=float,index=indx,columns=['1','3','6','12'])
for h in ['1','3','6','12'] :
    table.loc[('Martin-Wagner (All)','coef'),h] = coefs.loc[('FixUni','mw'),h]
    table.loc[('Martin-Wagner (All)','stderr'),h] = stderrs.loc[('FixUni','mw'),h]
    table.loc[('Martin-Wagner (All)','pvalue'),h] = pvals.loc[('FixUni','mw'),h]
    table.loc[('Kadan-Tang (Conservative)','coef'),h] = coefs.loc[('FixUni','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','stderr'),h] = stderrs.loc[('FixUni','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','pvalue'),h] = pvals.loc[('FixUni','kt'),h]
table2 = table.to_latex()
    
table = pd.DataFrame(dtype=float,index=indx,columns=['1','3','6','12'])
for h in ['1','3','6','12'] :
    table.loc[('Martin-Wagner (All)','coef'),h] = coefs.loc[('NoMulti','mw'),h]
    table.loc[('Martin-Wagner (All)','stderr'),h] = stderrs.loc[('NoMulti','mw'),h]
    table.loc[('Martin-Wagner (All)','pvalue'),h] = pvals.loc[('NoMulti','mw'),h]
    table.loc[('Kadan-Tang (Conservative)','coef'),h] = coefs.loc[('NoMulti','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','stderr'),h] = stderrs.loc[('NoMulti','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','pvalue'),h] = pvals.loc[('NoMulti','kt'),h]
table3 = table.to_latex()
    
table = pd.DataFrame(dtype=float,index=indx,columns=['1','3','6','12'])
for h in ['1','3','6','12'] :
    table.loc[('Martin-Wagner (All)','coef'),h] = coefs.loc[('FixMulti','mw'),h]
    table.loc[('Martin-Wagner (All)','stderr'),h] = stderrs.loc[('FixMulti','mw'),h]
    table.loc[('Martin-Wagner (All)','pvalue'),h] = pvals.loc[('FixMulti','mw'),h]
    table.loc[('Kadan-Tang (Conservative)','coef'),h] = coefs.loc[('FixMulti','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','stderr'),h] = stderrs.loc[('FixMulti','kt'),h]
    table.loc[('Kadan-Tang (Conservative)','pvalue'),h] = pvals.loc[('FixMulti','kt'),h]
table4 = table.to_latex()
    
    
#%% ===========================================================================
# add significance stars and output latex tables
# =============================================================================

for tab, out in zip([table1,table2,table3,table4],[OUTPUT1,OUTPUT2,OUTPUT3,OUTPUT4]) :
    table = sigtable.sigtable(tab,'coef','stderr','pvalue')
    with open(out, 'w') as f: 
        f.write(table)