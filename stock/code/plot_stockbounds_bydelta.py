from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

path  = PATH_TO_REPLICATION_PACKAGE + 'stock/'
INPUT = path + 'intermediate/ds_stock_monthly.csv'
OUTPUT= path + 'figures/stockbounds_bydelta.pdf'

#%% ===========================================================================
# read and prepare the data
# =============================================================================

df = pd.read_csv(INPUT)
for col in ['permno','date'] :
    df[col] = df[col].astype(int)
df.date = pd.to_datetime(df.date.astype(str),format='%Y%m')
df.set_index(['date','permno'],inplace=True)

# keep only bounds, forward returns, and delta
cols = [x + str(h) for x in ['lb_kt_','lb_mw_'] for h in [1,3,6,12]]
df = df[cols+['delta']]

df = df.rename(columns=dict(zip(['lb_kt_'+str(h) for h in [1,3,6,12]],['kt'+repr(h) for h in [1,3,6,12]])))
df = df.rename(columns=dict(zip(['lb_mw_'+str(h) for h in [1,3,6,12]],['mw'+repr(h) for h in [1,3,6,12]])))

#%% ===========================================================================
# plot figure 4
# =============================================================================

fig, axes = plt.subplots(2,2,figsize=(8,4),sharex=True)
h = '12'

ax = axes[0,0]
df2 = df[df.delta<=3]
b1 = df2.groupby('date')['kt'+h].median()
b2 = df2.groupby('date')['mw'+h].median()
ax.plot(b1,label='Kadan-Tang')
ax.plot(b2,label='Martin-Wagner')
ax.set_title('a) Conservative',fontsize=14)

ax = axes[0,1]
df2 = df[(df.delta>3)&(df.delta<=7)]
b1 = df2.groupby('date')['kt'+h].median()
b2 = df2.groupby('date')['mw'+h].median()
ax.plot(b1)
ax.plot(b2)
ax.set_title('b) Liberal',fontsize=14)

ax = axes[1,0]
df2 = df[df.delta>7]
b1 = df2.groupby('date')['kt'+h].median()
b2 = df2.groupby('date')['mw'+h].median()
ax.plot(b1)
ax.plot(b2)
ax.set_title('c) Other',fontsize=14)

ax = axes[1,1]
b1 = df.groupby('date')['kt'+h].median()
b2 = df.groupby('date')['mw'+h].median()
ax.plot(b1)
ax.plot(b2)
ax.set_title('d) All',fontsize=14)

fig.autofmt_xdate()
fig.tight_layout()
leg = axes[0,0].legend(loc='upper center', bbox_to_anchor=(1,-1.6),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT,additional_artists=[leg],bbox_inches='tight')
