from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# paths
PATH = PATH_TO_REPLICATION_PACKAGE + 'market/'
CODE_DIR         = PATH + 'code/'
INPUT_DIR        = PATH + 'input/'
INTERMEDIATE_DIR = PATH + 'intermediate/'   
TABLES_DIR       = PATH + 'tables/'         
FIGURES_DIR      = PATH + 'figures/'       

# input data that is used in this script
DS_DAILY  = INTERMEDIATE_DIR + 'ds_mkt_daily.csv'

# output of the script 
SUMSTATS_OUTPUT  = TABLES_DIR + 'summary_stats_'     # suffix for time horizon and bound identifiers added below

# Import the dataset
ds = pd.read_csv(DS_DAILY)

#%%########################################################################
# Summary statistics of the bounds and excess returns
###########################################################################
# formats for table
def f2(x):
    return '%5.2f' % x
def f0(x):
    return '%6.0f' % x

def sumstats_table(retvar, mvar, cyvar, suffix):
    ss1=ds[retvar][(ds[mvar].isnull()==0) & (ds.dp.isnull()==0) & (ds[retvar].isnull()==0)].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    ss2=ds[mvar][(ds[mvar].isnull()==0) & (ds.dp.isnull()==0) & (ds[retvar].isnull()==0)].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    ss3=ds[cyvar][(ds[mvar].isnull()==0) & (ds.dp.isnull()==0 )& (ds[retvar].isnull()==0)].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    
    ssdf = pd.merge(ss1,ss2, left_index=True,right_index=True)
    ssdf = pd.merge(ssdf,ss3, left_index=True,right_index=True)
    ssdf = ssdf.transpose()
    ssdf = ssdf[['mean','std','10%','25%','50%','75%','90%','count']].round(decimals=2)
    ssdf.index = ['Market Excess Return', 'Martin Bound', 'CYL Bound']
    
    # output to latex
    ssdf.to_latex(SUMSTATS_OUTPUT + suffix+'_pre.tex', 
               formatters = [f2, f2, f2, f2, f2, f2, f2, f0],
               header = ['Mean', 'SD','P10','P25','P50','P75','P90','N']
               )   
    # make latex file a fragment (comment begin/end tabular)
    file_pre = open(SUMSTATS_OUTPUT + suffix+'_pre.tex', 'r')
    file_out = open(SUMSTATS_OUTPUT + suffix+'.tex', 'w')
    checkWords = ('\\begin', '\end', '\\toprule')
    repWords = ('%\\begin', '%\end', '%\\toprule')
    for line in file_pre:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        file_out.write(line)
    file_pre.close()
    file_out.close()    
    
    os.remove(SUMSTATS_OUTPUT + suffix+'_pre.tex')
    return ssdf 

ssdf1 = sumstats_table(retvar='f_mktrf1', mvar='lb_m_1', cyvar = 'lb_cylr_1' , suffix = 'extended1')
ssdf3 = sumstats_table(retvar='f_mktrf3', mvar='lb_m_3', cyvar = 'lb_cylr_3' , suffix = 'extended3')
ssdf6 = sumstats_table(retvar='f_mktrf6', mvar='lb_m_6', cyvar = 'lb_cylr_6' , suffix = 'extended6')
ssdf12= sumstats_table(retvar='f_mktrf12',mvar='lb_m_12',cyvar = 'lb_cylr_12', suffix = 'extended12')

