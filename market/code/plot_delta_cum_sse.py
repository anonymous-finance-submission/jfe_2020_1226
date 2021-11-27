from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

PATH  = PATH_TO_REPLICATION_PACKAGE + 'market/'
INPUT = PATH + 'intermediate/market_forecasts.csv'
OUTPUT= PATH + 'figures/cum_sse_market.pdf'


df = pd.read_csv(INPUT)
df1 = df[(df.bound=='lb_m')&(df.truncation=='not_truncated')]
df1.date = pd.to_datetime(df1.date)

df2 = df[(df.bound=='lb_cylr')&(df.truncation=='not_truncated')]
df2.date = pd.to_datetime(df2.date)


#%% ===========================================================================
# generate figure 11
# =============================================================================

fcst = 'tight'

fig, axes = plt.subplots(2,2,figsize=(8,4),sharex=True)


h = 1
ax = axes[0,0]
df3 = df1[df1.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Martin')
df3 = df2[df2.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Chabi-Yo/Loudis')                
ax.set_title('a) 1 Month',fontsize=14)


h = 3
ax = axes[0,1]
df3 = df1[df1.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Martin')
df3 = df2[df2.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Chabi-Yo/Loudis')                
ax.set_title('a) 3 Month',fontsize=14)


h = 6
ax = axes[1,0]
df3 = df1[df1.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Martin')
df3 = df2[df2.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Chabi-Yo/Loudis')                
ax.set_title('c) 6 Month',fontsize=14)



h = 12
ax = axes[1,1]
df3 = df1[df1.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Martin')
df3 = df2[df2.horizon==h].copy()
df3.set_index('date',inplace=True)
df3 = df3[[fcst,'mkt','outcome']].dropna() / 100
f   = ((df3.outcome-df3[fcst])**2).cumsum()
mkt = ((df3.outcome-df3.mkt)**2).cumsum()
ax.plot(mkt-f,label='Chabi-Yo/Loudis')                
ax.set_title('d) 12 Month',fontsize=14)



fig.autofmt_xdate()
fig.tight_layout()
leg = axes[0,0].legend(loc='upper center', bbox_to_anchor=(1,-1.7),fontsize=10,
          ncol=2, fancybox=True, shadow=True)
fig.savefig(OUTPUT,additional_artists=[leg],bbox_inches='tight')