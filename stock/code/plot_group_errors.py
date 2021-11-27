from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm
import pandas_datareader as pdr

INPUT   = PATH_TO_REPLICATION_PACKAGE + 'stock/intermediate/ds_stock_monthly.csv'
OUTPUT1 = PATH_TO_REPLICATION_PACKAGE + 'stock/figures/cum_sse_conservative.pdf'
OUTPUT2 = PATH_TO_REPLICATION_PACKAGE + 'stock/figures/cum_sse_all.pdf'
horizons = ['1','3','6','12']

#%% ===========================================================================
# read and prepare the data
# =============================================================================

df = pd.read_csv(INPUT)
for col in ['permno','date'] :
    df[col] = df[col].astype(int)
df.date = pd.to_datetime(df.date.astype(str),format='%Y%m').dt.to_period('M')
df.set_index(['date','permno'],inplace=True)

df = df.rename(columns=dict(zip(['lb_kt_' + repr(h) for h in [1,3,6,12]],['kt'+repr(h) for h in [1,3,6,12]])))
df = df.rename(columns=dict(zip(['lb_mw_' + repr(h) for h in [1,3,6,12]],['mw'+repr(h) for h in [1,3,6,12]])))
# df = df.drop(columns=['ret'+h for h in horizons])
df = df.rename(columns=dict(zip(['f_xret' + repr(h) for h in [1,3,6,12]],['ret'+repr(h) for h in [1,3,6,12]])))

# define groups
groups = np.where(df.delta.isnull(),'Missing','Conservative')
groups = np.where(df.delta>3,'Liberal',groups)
groups = np.where(df.delta>7,'Other',groups)
df['group'] = groups

# add market benchmark
factors = pdr.DataReader('F-F_Research_Data_Factors','famafrench','1926-01-01')[0]
mktrf = 12*factors['Mkt-RF'].expanding().mean()
mktrf.index.name = 'date'
df = df.reset_index().merge(mktrf,on='date',how='left')
df.set_index(['date','permno','group'],inplace=True)

bnds = [b+h for b in ['kt','mw'] for h in horizons]
rets = ['ret'+h for h in horizons]
berrs = df[rets].apply(lambda x: x-df['Mkt-RF']) 
berrs = ( berrs / 100 )**2
berrs = berrs.rename(columns = dict(zip(rets,horizons)))
ferrs = pd.DataFrame(dtype=float,index=df.index,columns=bnds)
for b in ['kt','mw'] :
    for h in horizons :
        ferrs[b+h] = ( (df['ret'+h]-df[b+h])/100 )**2
diff = pd.DataFrame(dtype=float,index=df.index,columns=bnds)
for b in ['kt','mw'] :
    for h in horizons :
        diff[b+h] = berrs[h] - ferrs[b+h]
        
errsGroup = diff.unstack().groupby('date').mean().cumsum()
errsGroup.index = errsGroup.index.to_timestamp()
errsAll = diff.groupby('date').mean().cumsum()
errsAll.index = errsAll.index.to_timestamp()

#%% ===========================================================================
# plot figure 14
# =============================================================================
fig, axes = plt.subplots(2,2,figsize=(8,4),sharex=True)
group = 'Conservative'

h = '1'
ax = axes[0,0]
ax.plot(errsGroup['kt'+h,group],label='Kadan-Tang')
ax.plot(errsGroup['mw'+h,group],label='Martin-Wagner')
ax.set_title('a) 1 Month',fontsize=14)

h = '3'
ax = axes[0,1]
ax.plot(errsGroup['kt'+h,group],label='Kadan-Tang')
ax.plot(errsGroup['mw'+h,group],label='Martin-Wagner')
ax.set_title('b) 3 Month',fontsize=14)

h = '6'
ax = axes[1,0]
ax.plot(errsGroup['kt'+h,group],label='Kadan-Tang')
ax.plot(errsGroup['mw'+h,group],label='Martin-Wagner')
ax.set_title('c) 6 Month',fontsize=14)

h = '12'
ax = axes[1,1]
ax.plot(errsGroup['kt'+h,group],label='Kadan-Tang')
ax.plot(errsGroup['mw'+h,group],label='Martin-Wagner')
ax.set_title('d) 12 Month',fontsize=14)

fig.autofmt_xdate()
fig.tight_layout()
leg = axes[0,0].legend(loc='upper center', bbox_to_anchor=(1,-1.7),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT1,additional_artists=[leg],bbox_inches='tight')

#%% ===========================================================================
# plot figure 15
# =============================================================================

fig, axes = plt.subplots(2,2,figsize=(8,4),sharex=True)
group = 'Conservative'

h = '1'
ax = axes[0,0]
ax.plot(errsAll['kt'+h],label='Kadan-Tang')
ax.plot(errsAll['mw'+h],label='Martin-Wagner')
ax.set_title('a) 1 Month',fontsize=14)

h = '3'
ax = axes[0,1]
ax.plot(errsAll['kt'+h],label='Kadan-Tang')
ax.plot(errsAll['mw'+h],label='Martin-Wagner')
ax.set_title('b) 3 Month',fontsize=14)

h = '6'
ax = axes[1,0]
ax.plot(errsAll['kt'+h],label='Kadan-Tang')
ax.plot(errsAll['mw'+h],label='Martin-Wagner')
ax.set_title('c) 6 Month',fontsize=14)

h = '12'
ax = axes[1,1]
ax.plot(errsAll['kt'+h],label='Kadan-Tang')
ax.plot(errsAll['mw'+h],label='Martin-Wagner')
ax.set_title('d) 12 Month',fontsize=14)

fig.autofmt_xdate()
fig.tight_layout()
leg = axes[0,0].legend(loc='upper center', bbox_to_anchor=(1,-1.7),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT2,additional_artists=[leg],bbox_inches='tight')