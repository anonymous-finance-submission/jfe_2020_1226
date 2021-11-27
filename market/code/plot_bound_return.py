from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# paths
PATH = PATH_TO_REPLICATION_PACKAGE + 'market/'
CODE_DIR         = PATH + 'code/'
INPUT_DIR        = PATH + 'input/'
INTERMEDIATE_DIR = PATH + 'intermediate/'   
TABLES_DIR       = PATH + 'tables/'         
FIGURES_DIR      = PATH + 'figures/'       

# input data that is used in this script
INPUT  = INTERMEDIATE_DIR + 'ds_mkt_monthly.csv'

# output of the script 
OUTPUT_M   = FIGURES_DIR + 'market_bound_return_monthly_m.pdf'
OUTPUT_CYL = FIGURES_DIR + 'market_bound_return_monthly_cyl.pdf'

#%% ===========================================================================
# Import the dataset
# =============================================================================
df = pd.read_csv(INPUT)

df['date'] = pd.to_datetime(df.date)
df.set_index('date',inplace=True)

horizons = ['1','3','6','12']
for h in horizons : 
    df['ret'+h]    = df['f_mktrf'+h]
    df['martin'+h] = df['lb_m_'+h]
    df['cyl'+h]    = df['lb_cylr_'+h]
    df['slack_martin'+h] = df['ret'+h] - df['martin'+h]
    df['slack_cyl'+h]    = df['ret'+h] - df['cyl'+h]

cols = []
for h in horizons :
    cols = cols + ['ret'+h,'martin'+h,'cyl'+h,'slack_martin'+h,'slack_cyl'+h]
df = df[cols]

#%% ===========================================================================
# Martin Bound Plot
# =============================================================================

fig, axes = plt.subplots(1,4,figsize=(8,3),sharey=True)
n = 50

h = '1'
type = 'martin'
ax = axes[0]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('a) 1 Month',fontsize=14)
ax.set_ylabel('Excess Return (%)',fontsize=12)
ax.set_xlabel('Bound (%)')

h = '3'
type = 'martin'
ax = axes[1]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('b) 3 Month',fontsize=14)
ax.set_ylabel('')
ax.set_xlabel('Bound (%)',fontsize=12)

h = '6'
type = 'martin'
ax = axes[2]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('c) 6 Month',fontsize=14)
ax.set_ylabel('')
ax.set_xlabel('Bound (%)',fontsize=12)

h = '12'
type = 'martin'
ax = axes[3]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1,'label':'regression line'},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--',label='$45^\circ$ line')
ax.set_title('d) 12 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

fig.tight_layout()
leg = axes[3].legend(loc='upper center', bbox_to_anchor=(-1.2,-0.25),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT_M,additional_artists=[leg],bbox_inches='tight')

#%% ===========================================================================
# Chabi-Yo/Loudis Bound Plot
# =============================================================================

fig, axes = plt.subplots(2,2,figsize=(6,6),sharey=True)
n = 50

h = '1'
type = 'cyl'
ax = axes[0,0]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('a) 1 Month',fontsize=14)
ax.set_ylabel('Excess Return (%)',fontsize=12)
ax.set_xlabel('Bound (%)')

h = '3'
type = 'cyl'
ax = axes[0,1]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('b) 3 Month',fontsize=14)
ax.set_ylabel('')
ax.set_xlabel('Bound (%)',fontsize=12)

h = '6'
type = 'cyl'
ax = axes[1,0]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--')
ax.set_title('c) 6 Month',fontsize=14)
ax.set_ylabel('')
ax.set_xlabel('Bound (%)',fontsize=12)

h = '12'
type = 'cyl'
ax = axes[1,1]
groups = pd.qcut(df[type+h],n,labels=False)
bd = df.groupby(groups)[type+h].mean()
ret = df.groupby(groups)['ret'+h].mean()
sns.regplot(x=bd,y=ret,ax=ax,ci=None,line_kws={'color':'red','lw':1,'label':'regression line'},scatter_kws={'s':25,'alpha':0.5})
ax.plot(bd,bd,color='green',ls='--',label='$45^\circ$ line')
ax.set_title('d) 12 Month',fontsize=14)
ax.set_xlabel('Bound (%)',fontsize=12)
ax.set_ylabel('')

fig.tight_layout()
leg = axes[1,1].legend(loc='upper center', bbox_to_anchor=(0,-0.3),fontsize=12,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT_CYL,additional_artists=[leg],bbox_inches='tight')