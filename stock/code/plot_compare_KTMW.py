from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

INPUT  = PATH_TO_REPLICATION_PACKAGE + 'stock/intermediate/ds_stock_monthly.csv'
OUTPUT = PATH_TO_REPLICATION_PACKAGE + 'stock/figures/compare_KTMW.pdf'

#%% ===========================================================================
# read and prepare the data
# =============================================================================

df = pd.read_csv(INPUT)
for col in ['permno','date'] :
    df[col] = df[col].astype(int)
df.date = pd.to_datetime(df.date.astype(str),format='%Y%m')
df.set_index(['date','permno'],inplace=True)

# keep only bounds and forward returns
cols = [x + str(h) for x in ['lb_kt_','lb_mw_'] for h in [1,3,6,12]]
df = df[cols]

df = df.rename(columns=dict(zip(['lb_kt_'+str(h) for h in [1,3,6,12]],['kt'+repr(h) for h in [1,3,6,12]])))
df = df.rename(columns=dict(zip(['lb_mw_'+str(h) for h in [1,3,6,12]],['mw'+repr(h) for h in [1,3,6,12]])))

#%% ===========================================================================
# plot figure 5
# =============================================================================

fig, axes = plt.subplots(2,2,figsize=(8,4),sharex=True)

ax = axes[0,0]
h = '1'
diff = (df['kt'+h] -2*df['mw'+h])
diff = diff.groupby('date').median()
ax.plot(diff)
ax.set_title('a) 1 Month',fontsize=14)

ax = axes[0,1]
h = '3'
diff = (df['kt'+h] -2*df['mw'+h])
diff = diff.groupby('date').median()
ax.plot(diff)
ax.set_title('b) 3 Month',fontsize=14)

ax = axes[1,0]
h = '6'
diff = (df['kt'+h] -2*df['mw'+h])
diff = diff.groupby('date').median()
ax.plot(diff)
ax.set_title('c) 6 Month',fontsize=14)

ax = axes[1,1]
h = '12'
diff = (df['kt'+h] -2*df['mw'+h])
diff = diff.groupby('date').median()
ax.plot(diff)
ax.set_title('d) 12 Month',fontsize=14)

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTPUT)