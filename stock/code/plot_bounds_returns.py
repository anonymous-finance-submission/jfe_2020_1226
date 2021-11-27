from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

path  = PATH_TO_REPLICATION_PACKAGE + 'stock/'
INPUT = path + 'intermediate/ds_stock_monthly.csv'
OUTPUTMW  = path + 'figures/MW_bound_return.pdf'
OUTPUTKT1 = path + 'figures/KT_conservative_bound_return.pdf'
OUTPUTKT2 = path + 'figures/KT_liberal_bound_return.pdf'

#%% ===========================================================================
# read and prepare the data
# =============================================================================
df = pd.read_csv(INPUT)
for col in ['permno','date'] :
    df[col] = df[col].astype(int)
df.date = pd.to_datetime(df.date.astype(str),format='%Y%m')
df.set_index(['date','permno'],inplace=True)
   
# keep only bounds and forward returns
cols = [x + repr(h) for x in ['lb_kt_','lb_mw_','f_xret'] for h in [1,3,6,12]]
df = df[cols+['delta']]

df = df.rename(columns=dict(zip(['lb_kt_' + repr(h) for h in [1,3,6,12]],['kt'+repr(h) for h in [1,3,6,12]])))
df = df.rename(columns=dict(zip(['lb_mw_' + repr(h) for h in [1,3,6,12]],['mw'+repr(h) for h in [1,3,6,12]])))
df = df.rename(columns=dict(zip(['f_xret' + repr(h) for h in [1,3,6,12]],['ret'+repr(h) for h in [1,3,6,12]])))

#%% ===========================================================================
# plot figure 6
# =============================================================================
fig, axes = plt.subplots(1,4,figsize=(8,3),sharey=True)

h = '1'
type = 'mw'
ax = axes[0]
groups = pd.qcut(df[type+h],100,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('a) 1 Month',fontsize=14)
ax.set_ylabel('Excess Return (%)',fontsize=12)
ax.set_xlabel('Bound (%)',fontsize=12)

h = '3'
type = 'mw'
ax = axes[1]
groups = pd.qcut(df[type+h],100,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('b) 3 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

h = '6'
type = 'mw'
ax = axes[2]
groups = pd.qcut(df[type+h],100,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('c) 6 Month',fontsize=14)
ax.set_ylabel('',fontsize=12)
ax.set_xlabel('Bound (%)',fontsize=12)

h = '12'
type = 'mw'
ax = axes[3]
groups = pd.qcut(df[type+h],100,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1,'label':'regression line'},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--',label='$45^\circ$ line')
ax.set_title('d) 12 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

fig.tight_layout()
# leg = axes[1,1].legend(loc='upper center', bbox_to_anchor=(0,-0.3),fontsize=12,
leg = axes[3].legend(loc='upper center', bbox_to_anchor=(-1.15,-0.25),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUTMW,additional_artists=[leg],bbox_inches='tight')


#%% ===========================================================================
# plot figure 7
# =============================================================================
df2 = df[df.delta<=3]

fig, axes = plt.subplots(1,4,figsize=(8,3),sharey=True)

h = '1'
type = 'kt'
ax = axes[0]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('a) 1 Month',fontsize=14)
ax.set_ylabel('Excess Return (%)',fontsize=12)
ax.set_xlabel('Bound (%)',fontsize=12)

h = '3'
type = 'kt'
ax = axes[1]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('b) 3 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

h = '6'
type = 'kt'
ax = axes[2]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('c) 6 Month',fontsize=14)
ax.set_ylabel('Excess Return (%)',fontsize=12)
ax.set_xlabel('Bound (%)',fontsize=12)

h = '12'
type = 'kt'
ax = axes[3]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1,'label':'regression line'},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--',label='$45^\circ$ line')
ax.set_title('d) 12 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

fig.tight_layout()
leg = axes[3].legend(loc='upper center', bbox_to_anchor=(-1.4,-0.25),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUTKT1,additional_artists=[leg],bbox_inches='tight')

#%% ===========================================================================
# plot figure 8
# =============================================================================

df2 = df[(df.delta>3)&(df.delta<=7)]

fig, axes = plt.subplots(1,4,figsize=(8,3),sharey=True)

h = '1'
type = 'kt'
ax = axes[0]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('a) 1 Month',fontsize=14)
ax.set_ylabel('Excess Return (%)',fontsize=12)
ax.set_xlabel('Bound (%)',fontsize=12)

h = '3'
type = 'kt'
ax = axes[1]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('b) 3 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

h = '6'
type = 'kt'
ax = axes[2]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('c) 6 Month',fontsize=14)
ax.set_ylabel('')
ax.set_xlabel('Bound (%)',fontsize=12)

h = '12'
type = 'kt'
ax = axes[3]
groups = pd.qcut(df2[type+h],100,labels=False)
bd = df2.groupby(groups)[type+h].mean()
ret = df2.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1,'label':'regression line'},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--',label='$45^\circ$ line')
ax.set_title('d) 12 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

fig.tight_layout()
leg = axes[3].legend(loc='upper center', bbox_to_anchor=(-1.2,-0.25),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUTKT2,additional_artists=[leg],bbox_inches='tight')