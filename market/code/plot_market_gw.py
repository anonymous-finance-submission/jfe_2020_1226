from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

PATH  = PATH_TO_REPLICATION_PACKAGE + 'market/'
INPUT = PATH + 'intermediate/market_forecasts.csv'
OUTPUT= PATH + 'figures/market_cum_sse.pdf'

df = pd.read_csv(INPUT)
df = df[(df.bound=='lb_m')&(df.truncation=='not_truncated')]
df.date = pd.to_datetime(df.date)

#%%
fig, axes = plt.subplots(2,2,figsize=(8,4),sharex=True)
fcst = 'smean'

color1 = 'tab:blue'
color2 = 'tab:red'
color3 = 'tab:brown'

h = 3

ax = axes[0,0]
df2 = df[df.horizon==h].copy()
df2.set_index('date',inplace=True)
df2 = df2[[fcst,'mkt','outcome']].dropna() / 100
ax.plot(df2[fcst],color=color1)
ax.plot(df2.mkt,color=color3)
ax.set_title('a) 3 Month Fcst & Benchmark',fontsize=14)

ax = axes[1,0]
f = ((df2.outcome-df2[fcst])**2).cumsum()
mkt = ((df2.outcome-df2.mkt)**2).cumsum()
ax.plot(mkt-f,color=color2)
ax.set_title('b) 3 Month '+r'$\Delta$ Cum Squared Error',fontsize=14)

h = 12

ax = axes[0,1]
df2 = df[df.horizon==h].copy()
df2.set_index('date',inplace=True)
df2 = df2[[fcst,'mkt','outcome']].dropna() / 100
ax.plot(df2[fcst],color=color1,label='forecast')
ax.plot(df2.mkt,color=color3,label='benchmark')
ax.set_title('c) 12 Month Fcst & Benchmark',fontsize=14)

ax = axes[1,1]
f = ((df2.outcome-df2[fcst])**2).cumsum()
mkt = ((df2.outcome-df2.mkt)**2).cumsum()
ax.plot(mkt-f,color=color2)
ax.set_title('d) 12 Month '+r'$\Delta$ Cum Squared Error',fontsize=14)


plt.tight_layout() # pad=0.8, w_pad=0.5, h_pad=2.0
leg = axes[0,1].legend(loc='upper center', bbox_to_anchor=(0,-1.6),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT,additional_artists=[leg],bbox_inches='tight')