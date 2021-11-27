from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import os

# input data that is used in this script
path = PATH_TO_REPLICATION_PACKAGE + 'stock/'
DS_MONTHLY = path + 'intermediate/ds_stock_monthly.csv'

# output of the script 
SUMSTATS_OUTPUT  = path + 'tables/panel_summary_stats_'   # suffix for time horizon and bound identifiers added below

# read the data
ds = pd.read_csv(DS_MONTHLY)

#%% ===========================================================================
# Summary statistics of stock-level panel
# =============================================================================

# Define filter masks for various delta bins
delta_all = (ds['delta'].notnull())
delta_lt3 = (ds['delta'] <3)
delta_3_7 = ((ds['delta'] >=3) & (ds['delta'] <=7 ))
delta_gt7 = (ds['delta'] >7)

ds = ds[delta_all].copy()
  

# formats for table output
def f2(x):
    return '%5.2f' % x
def f0(x):
    return '%6.0f' % x

# make summary stats tables for each horizon
for h in [1,3,6,12]:
    # variables of interest
    retvar = 'f_xret' + str(h)
    mwvar   = 'lb_mw_' + str(h)
    ktvar   = 'lb_kt_' + str(h)   
    filter_nonmissing = (ds[mwvar].isnull()==0) & (ds.dp.isnull()==0) & (ds[retvar].isnull()==0)
    
    # empty dataframe, columns are stats, rows are returns & bounds
    indx    = ['Excess Return: All', 'Excess Return: $\delta <3$', 'Excess Return: $\delta \in$ [3,7]', 'Excess Return: $\delta >7$',
               'MW Bound: All', 'MW Bound: $\delta <3$','MW Bound: $\delta \in$ [3,7]', 'MW Bound: $\delta >7$',
               'KT Bound: All', 'KT Bound: $\delta <3$','KT Bound: $\delta \in$ [3,7]', 'KT Bound: $\delta >7$']
    indx = ['\\underline{Excess Return} \\\\ All', 'Conservative', 'Liberal', 'Other',
               '\\underline{Martin-Wagner Bound} \\\\  All', 'Conservative', 'Liberal', 'Other',
               '\\underline{Kadan-Tang Bound} \\\  All', 'Conservative', 'Liberal', 'Other']
    colindx = ['count', 'mean', 'std', 'min', '10%', '25%', '50%', '75%', '90%', 'max']
    Table = pd.DataFrame(dtype=float,index=indx,columns=colindx)
    
    for i, j in enumerate([delta_all, delta_lt3, delta_3_7, delta_gt7]):
        Table.iloc[i] = ds[retvar][filter_nonmissing & j].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])        
    for i, j in enumerate([delta_all, delta_lt3, delta_3_7, delta_gt7]):
        Table.iloc[4+i] = ds[mwvar][filter_nonmissing & j].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])        
    for i, j in enumerate([delta_all, delta_lt3, delta_3_7, delta_gt7]):        
        Table.iloc[8+i] = ds[ktvar][filter_nonmissing & j].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])        

    Table = Table[['mean','std','10%','25%','50%','75%','90%','count']].round(decimals=2)
    
    print(Table)

    # output to latex
    Table.to_latex(SUMSTATS_OUTPUT + str(h)+'_pre.tex', 
               formatters = [f2, f2, f2, f2, f2, f2, f2, f0],
               header = ['Mean', 'SD','P10','P25','P50','P75','P90','N'], escape=False
               )   
    # make latex file a fragment (comment begin/end tabular)
    file_pre = open(SUMSTATS_OUTPUT + str(h)+'_pre.tex', 'r')
    file_out = open(SUMSTATS_OUTPUT + str(h)+'.tex', 'w')
    checkWords = ('\\begin', '\end', '\\toprule')
    repWords = ('%\\begin', '%\end', '%\\toprule')
    for line in file_pre:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        file_out.write(line)
    file_pre.close()
    file_out.close()    
    
    os.remove(SUMSTATS_OUTPUT + str(h)+'_pre.tex')
