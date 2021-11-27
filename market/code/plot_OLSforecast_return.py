from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

PATH  = PATH_TO_REPLICATION_PACKAGE + 'market/'
INPUT = PATH + 'intermediate/market_forecasts.csv'
OUTPUT= PATH + 'figures/market_OLSforecast_return.pdf'
n = 50

df = pd.read_csv(INPUT)
df = df.rename(columns=dict(zip(df.columns[:3],['bnd','type','horizon'])))
df = df[df.type=='not_truncated']


#%%
fig, axes = plt.subplots(1,4,figsize=(8,3),sharey=True)

h = '1'
df2 = df[df.horizon==int(h)]

ax = axes[0]
groups = pd.qcut(df2['rbound'],n,labels=False)
bd = df2.groupby(groups)['rbound'].mean()
ret = df2.groupby(groups)['outcome'].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('a) 1 Month',fontsize=14)
ax.set_ylabel('Excess Return (%)',fontsize=12)
ax.set_xlabel('Forecast (%)',fontsize=12)

h = '3'
df2 = df[df.horizon==int(h)]

ax = axes[1]
groups = pd.qcut(df2['rbound'],n,labels=False)
bd = df2.groupby(groups)['rbound'].mean()
ret = df2.groupby(groups)['outcome'].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('b) 3 Month',fontsize=14)
ax.set_xlabel('Forecast (%)',fontsize=12)
ax.set_ylabel('')

h = '6'
df2 = df[df.horizon==int(h)]

ax = axes[2]
groups = pd.qcut(df2['rbound'],n,labels=False)
bd = df2.groupby(groups)['rbound'].mean()
ret = df2.groupby(groups)['outcome'].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('c) 6 Month',fontsize=14)
ax.set_xlabel('Forecast (%)',fontsize=12)
ax.set_ylabel('')

h = '12'
df2 = df[df.horizon==int(h)]

ax = axes[3]
groups = pd.qcut(df2['rbound'],n,labels=False)
bd = df2.groupby(groups)['rbound'].mean()
ret = df2.groupby(groups)['outcome'].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1,'label':'regression line'},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--',label='$45^\circ$ line')
ax.set_title('d) 12 Month',fontsize=14)
ax.set_xlabel('Forecast (%)',fontsize=12)
ax.set_ylabel('')

fig.tight_layout()
leg = axes[3].legend(loc='upper center', bbox_to_anchor=(-1.15,-0.25),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT,additional_artists=[leg],bbox_inches='tight')
