from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm
import pandas_datareader as pdr
ff = pdr.DataReader('F-F_Research_Data_5_Factors_2x3','famafrench',start='1990-01')[0] 

path     = PATH_TO_REPLICATION_PACKAGE + 'stock/'
INPUT    = path + 'intermediate/ds_stock_monthly.csv'
OUTPUTMW = path + 'tables/MW_decile_returns.tex'
OUTPUTKT = path + 'tables/KT_conservative_decile_returns.tex'

# Number of bins (quintiles)
n = 5

#%% ===========================================================================
# define functions
# =============================================================================
# this functions adds placeholders for significance stars
def add_stars(s, p):
    if p > 0.10:
        return str(s)
    elif p > 0.05:
        return str(s) + 'ONESTAR'
    elif p > 0.01:
        return str(s) + 'TWOSTARS'
    else:
        return str(s) + 'THREESTARS'
    
#%% ===========================================================================
# read and prepare the data
# =============================================================================
df = pd.read_csv(INPUT)
for col in ['permno','date'] :
    df[col] = df[col].astype(int)
df.date = pd.to_datetime(df.date.astype(str),format='%Y%m').dt.to_period('M')
df.set_index(['date','permno'],inplace=True)
   
# keep only bounds and forward returns
cols = [x + repr(h) for x in ['lb_kt_','lb_mw_','f_xret'] for h in [1,3,6,12]]
df = df[cols+['delta']]

#%% ===========================================================================
# table 10 for MW
# =============================================================================
table = pd.DataFrame(dtype=float,columns=['1','3','6','12'],index=[str(i) for i in range(1,n+1)]+['5-1:']+[x+y for x in ['mean','CAPM','FF'] for y in ['','_se']])
bnd = 'lb_mw_'
for h in ['1','3','6','12'] :
    bd = bnd+h
    ret = 'f_xret'+h
    df['decile'] = df.groupby('date')[bd].apply(lambda x: pd.qcut(x,n,labels=range(1,n+1)))
    rets = df.groupby(['date','decile'])[ret].mean().unstack()

    table[h].iloc[:n] = ['{:.2f}'.format(x) for x in rets.mean().round(2)]
    hml = (rets[n] - rets[1]).dropna()
    X = sm.add_constant(hml)
    X = X['const']
    result = sm.OLS(hml,X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc['mean',h] = add_stars(np.round(hml.mean(), 2) , result.pvalues.item())
    table.loc['mean_se',h] = '({:.2f})'.format(np.sqrt(result.cov_params()['const']['const']))

    ff['ret'] = hml
    data = ff.dropna()
    X = sm.add_constant(data['Mkt-RF'])
    result = sm.OLS(data['ret'],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc['CAPM',h] = add_stars(np.round(result.params['const'], 2), result.pvalues['const'])
    table.loc['CAPM_se',h] = '({:.2f})'.format(np.sqrt(result.cov_params()['const']['const']))

    X = sm.add_constant(data[['Mkt-RF','SMB','HML','RMW','CMA']])
    result = sm.OLS(data['ret'],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc['FF',h] = add_stars(np.round(result.params['const'], 2), result.pvalues['const'])
    table.loc['FF_se',h] = '({:.2f})'.format(np.sqrt(result.cov_params()['const']['const']))

# format the table and convert to latex
table.index = [str(i) for i in range(1,n+1)]+['PLACEHOLDER1 5-1:']+ \
            ['PLACEHOLDER2 Mean', '','PLACEHOLDER2 CAPM PLACEHOLDER3', '', 'PLACEHOLDER2 Fama-French PLACEHOLDER3', '']
table = table.to_latex(na_rep='')
table = table.replace('PLACEHOLDER1', '\\midrule')
table = table.replace('PLACEHOLDER2', '\\hspace{2em}')
table = table.replace('PLACEHOLDER3', '$\\alpha$')
table = table.replace('ONESTAR', '$^{*}$')
table = table.replace('TWOSTARS', '$^{**}$')
table = table.replace('THREESTARS', '$^{***}$')
table = table.replace('\\begin{tabular}{lllll}','%\\begin{tabular}{lllll}')
table = table.replace('\\toprule', '%\\toprule')
table = table.replace('\\end{tabular}', '%\\end{tabular}')

with open(OUTPUTMW, 'w') as f: 
    f.write(table)

#%% ===========================================================================
# table 10 for KT
# =============================================================================
df2 = df[df.delta<=3].copy()
table = pd.DataFrame(dtype=float,columns=['1','3','6','12'],index=[str(i) for i in range(1,n+1)]+['5-1:']+[x+y for x in ['mean','CAPM','FF'] for y in ['','_se']])
bnd = 'lb_kt_'
for h in ['1','3','6','12'] :
    bd = bnd+h
    ret = 'f_xret'+h
    df2['decile'] = df2.groupby('date')[bd].apply(lambda x: pd.qcut(x,n,labels=range(1,n+1)))
    rets = df2.groupby(['date','decile'])[ret].mean().unstack()

    table[h].iloc[:n] = ['{:.2f}'.format(x) for x in rets.mean().round(2)]
    hml = (rets[n] - rets[1]).dropna()
    X = sm.add_constant(hml)
    X = X['const']
    result = sm.OLS(hml,X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc['mean',h] = add_stars(np.round(hml.mean(), 2) , result.pvalues.item())
    table.loc['mean_se',h] = '({:.2f})'.format(np.sqrt(result.cov_params()['const']['const']))

    ff['ret'] = hml
    data = ff.dropna()
    X = sm.add_constant(data['Mkt-RF'])
    result = sm.OLS(data['ret'],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc['CAPM',h] = add_stars(np.round(result.params['const'], 2), result.pvalues['const'])
    table.loc['CAPM_se',h] = '({:.2f})'.format(np.sqrt(result.cov_params()['const']['const']))

    X = sm.add_constant(data[['Mkt-RF','SMB','HML','RMW','CMA']])
    result = sm.OLS(data['ret'],X).fit(cov_type='HAC',cov_kwds={'maxlags':int(h),'kernel':'uniform'})
    table.loc['FF',h] = add_stars(np.round(result.params['const'], 2), result.pvalues['const'])
    table.loc['FF_se',h] = '({:.2f})'.format(np.sqrt(result.cov_params()['const']['const']))
      
# format the table and convert to latex
table.index = [str(i) for i in range(1,n+1)]+['PLACEHOLDER1 5-1:']+ \
            ['PLACEHOLDER2 Mean', '','PLACEHOLDER2 CAPM PLACEHOLDER3', '', 'PLACEHOLDER2 Fama-French PLACEHOLDER3', '']
table = table.to_latex(na_rep='')
table = table.replace('PLACEHOLDER1', '\\midrule')
table = table.replace('PLACEHOLDER2', '\\hspace{2em}')
table = table.replace('PLACEHOLDER3', '$\\alpha$')
table = table.replace('ONESTAR', '$^{*}$')
table = table.replace('TWOSTARS', '$^{**}$')
table = table.replace('THREESTARS', '$^{***}$')
table = table.replace('\\begin{tabular}{lllll}','%\\begin{tabular}{lllll}')
table = table.replace('\\toprule', '%\\toprule')
table = table.replace('\\end{tabular}', '%\\end{tabular}')

with open(OUTPUTKT, 'w') as f:
    f.write(table)


