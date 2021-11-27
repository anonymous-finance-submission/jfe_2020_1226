from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import os
import time
from numpy.random import RandomState
from arch.bootstrap import CircularBlockBootstrap
import shutil

# input data that is used in this script
path = PATH_TO_REPLICATION_PACKAGE + 'stock/'
DS_MONTHLY= path + 'intermediate/ds_stock_monthly.csv'        # both as .csv and .pkl

# load KPClass.py from code folder
os.chdir(path +'code/')
from KPClass_stocklevel import KP

# output of the script 
BS_OUTPUT   = path + 'intermediate/bootstrap/Sigma_avgbound_'            # suffix added below
TABLE_UNFORMATTED_OUTPUT   = path + 'intermediate/validity_tightness/kp_tests_avgbound_'   # suffix added below
TABLE_OUTPUT   = path + 'tables/kp_tests_avgbound_'                     # suffix added below

# read data
ds = pd.read_csv(DS_MONTHLY)
gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']


# delta filter masks
delta_lt3 = (ds['delta'] <3)
delta_3_7 = ((ds['delta'] >=3) & (ds['delta'] <=7 ))
delta_gt7 = ((ds['delta'] >7))


# Replace bounds with sample averages for MW, KT conservative, KT liberal, and KT Other
for i, h in enumerate([1,3,6,12]): 
    # Replace MW with its sample average
    avgbnd = ds['lb_mw_'+str(h)].mean()
    ds['lb_mw_'+str(h)] = avgbnd
    
    # Make new bound column just for KT-Conservative 
    filter_expr = delta_lt3
    avgbnd = ds[filter_expr]['lb_kt_'+str(h)].mean()
    ds['lb_kt_lt3_'+ str(h)] = np.where(filter_expr, avgbnd, np.nan)
    
    # Make new bound column just for KT-Liberal     
    filter_expr = delta_3_7
    avgbnd = ds[filter_expr]['lb_kt_'+str(h)].mean()
    ds['lb_kt_3_7_'+ str(h)] = np.where(filter_expr, avgbnd, np.nan)

    # Make new bound column just for KT-Other
    filter_expr = delta_gt7
    avgbnd = ds[filter_expr]['lb_kt_'+str(h)].mean()
    ds['lb_kt_gt7_'+ str(h)] = np.where(filter_expr, avgbnd, np.nan)





#%%########################################################################
# Draw block bootstrap samples for serial correlation adjustment
###########################################################################
# Create time variable
ds['ddate'] = pd.to_datetime(ds['date'], format='%Y%m')
ds['mdate'] = pd.PeriodIndex(year=ds.ddate.dt.year, month=ds.ddate.dt.month, freq='M')

# Keep firm-months with non-missing data in preparation for bootstrapping
cols = ['permno','date','mdate','delta'] + ['f_xret'+str(h) for h in [1,3,6,12]] + ['lb_mw_'+str(h) for h in [1,3,6,12]]  +  ['lb_kt_lt3_'+str(h) for h in [1,3,6,12]] + ['lb_kt_3_7_'+str(h) for h in [1,3,6,12]] + ['lb_kt_gt7_'+str(h) for h in [1,3,6,12]] + gw_vars
filter_nonmissing = (ds['lb_mw_12'].isnull()==0) & (ds['f_xret12'].isnull()==0)
bs_df = ds[cols][filter_nonmissing].copy()

time_values = pd.Series(ds['f_xret12'][filter_nonmissing].groupby(ds.mdate).first().index)   #Using pandas period[M] values


# number of bootstraps
B = 1000  
for h in [1,3,6,12]:
    print('******************************* Bootstrap for horizon: ', h)
    tic = time.perf_counter()
    # Create horizon-specific slackness with conditioning interaction for each bound
    for m in ['mw', 'kt_lt3', 'kt_3_7', 'kt_gt7']:
        bnd = 'lb_' + m + '_' + str(h)
        ret = 'f_xret' + str(h)
        bs_df['interact_' + m + '0'] = bs_df[ret] - bs_df[bnd]
        for g, gw in enumerate(gw_vars):
            bs_df['interact_' + m + str(g+1)] = (bs_df[ret] - bs_df[bnd])*bs_df[gw]
    
    # Create set of bootstrapped time indicators
    rs = RandomState(int(h)+5000)
    bs = CircularBlockBootstrap(h, time_values, random_state=rs)            #first parm is block size, 2nd parm is dataframe to draw from, 3rd parm is seed for replicability
    
    
    # Empty dataframes to hold moments for each bootstrap sample
    moments_mw     = pd.DataFrame(dtype=float,index=range(B),columns=['moment' + str(j) for j in range(len(gw_vars)+1)])    
        
    moments_kt_lt3 = pd.DataFrame(dtype=float,index=range(B),columns=['moment' + str(j) for j in range(len(gw_vars)+1)])
    moments_kt_3_7 = pd.DataFrame(dtype=float,index=range(B),columns=['moment' + str(j) for j in range(len(gw_vars)+1)])
    moments_kt_gt7 = pd.DataFrame(dtype=float,index=range(B),columns=['moment' + str(j) for j in range(len(gw_vars)+1)])

    
    
    for i, data in enumerate(bs.bootstrap(B)):             #number of bootstraps
        if np.mod(i,100)==0:
            print('BS run # ', i)
        # the bootstrapped dataframe is accessed via: data[0][0]
        bs_data = data[0][0].reset_index()
        obs_num = pd.DataFrame(range(len(bs_data)), index=bs_data.index, columns=['obs_num'])
        bs_data = pd.merge(bs_data.mdate, obs_num, how='left',left_index=True, right_index=True)
        
        # Pull full cross-section for each sample time period in block bootstrap sample
        bs_panel= pd.merge(bs_data, bs_df, how='left', left_on='mdate', right_on ='mdate')
  
        # Calculate the moments for the bootstrap sample for Martin/Wagner
        conditioned_vars   = ['interact_mw' + str(j) for j in range(len(gw_vars)+1)]
        moments_mw.iloc[i] = bs_panel[conditioned_vars].mean().values.T
        
        # Calculate the moments for the bootstrap sample for each Kadan/Tang bin
        conditioned_vars = ['interact_kt_lt3' + str(j) for j in range(len(gw_vars)+1)]
        moments_kt_lt3.iloc[i] = bs_panel[conditioned_vars].mean().values.T

        conditioned_vars = ['interact_kt_3_7' + str(j) for j in range(len(gw_vars)+1)]
        moments_kt_3_7.iloc[i] = bs_panel[conditioned_vars].mean().values.T
        
        conditioned_vars = ['interact_kt_gt7' + str(j) for j in range(len(gw_vars)+1)]
        moments_kt_gt7.iloc[i] = bs_panel[conditioned_vars].mean().values.T        
        
        
    moments_mw.cov().to_csv(BS_OUTPUT + 'mw_h' + str(h)+'.csv')    
    
    moments_kt_lt3.cov().to_csv(BS_OUTPUT + 'kt_lt3_h' + str(h)+'.csv')
    moments_kt_3_7.cov().to_csv(BS_OUTPUT + 'kt_3_7_h' + str(h)+'.csv')
    moments_kt_gt7.cov().to_csv(BS_OUTPUT + 'kt_gt7_h' + str(h)+'.csv')

    toc = time.perf_counter()   
    print(f"Calculations took {toc - tic:0.4f} seconds \r\n")        

#%%########################################################################
# Run Kodde-Palm tests for each bound & delta bin
###########################################################################

models = ['mw', 'kt_lt3', 'kt_3_7', 'kt_gt7']
for m in models:
    print('********** Model: ', m)
    
    # empty dataframe, rows are stats, columns are horizons
    indx    = ['Wald ($D_0$)', 'p-value (%)', 'Validity ($D_1$)', 'p-value (%)', 'Tightness ($D_2$)', 'p-value (%)']
    colindx = ['1', '3', '6', '12']
    Table   = pd.DataFrame(dtype=float,index=indx,columns=colindx)
    Table_UB= pd.DataFrame(dtype=float,index=indx,columns=colindx)
    
    
    tic = time.perf_counter()
    for h in [1,3,6,12]:
        print('***** Horizon: ', h)
        # Subset data
        d = ds.copy()
        bnd = 'lb_'+ m + '_' + str(h)
        ret = 'f_xret' + str(h)
        cols = ['permno','date','mdate','delta'] + [bnd, ret] + gw_vars
        d = d[cols].dropna()

        # Define filter masks for various delta bins
        delta_all = (d['delta'].notnull())
        delta_lt3 = (d['delta'] <3)
        delta_3_7 = ((d['delta'] >=3) & (d['delta'] <=7 ))
        delta_gt7 = (d['delta'] >7)
        filtermap = {'mw':delta_all, 'kt_lt3':delta_lt3, 'kt_3_7':delta_3_7, 'kt_gt7': delta_gt7}
        delta_filter = filtermap[m]
        d = d[delta_filter]
                       
        # Create horizon-specific slackness with conditioning interaction for each bound        
        d['interact0'] = d[ret] - d[bnd]
        for g, gw in enumerate(gw_vars):
            d['interact' + str(g+1)] = (d[ret] - d[bnd])*d[gw]
    
        # Calculate the moments for the actual sample for Martin/Wagner
        conditioned_vars = ['interact' + str(j) for j in range(len(gw_vars)+1)]
        moments = d[conditioned_vars].mean()
    
        # Load bootstrapped covariance matrix
        SIG = pd.read_csv(BS_OUTPUT + m + '_h' + str(h)+'.csv', header=0, index_col=0)
        # print(SIG)
        kp = KP(moments, SIG.to_numpy())                                          #NOTE: these input parameters are unused later - only necessary to call the class
        gamma_hat, wald, validity, tightness, exit_flag = kp.kp_output()
        pval_wald, pval_validity, pval_tightness  = kp.p_values(wald,validity,tightness,SIG.to_numpy(),num_sim = 10000)
        Table[str(h)]=[wald, pval_wald, validity, pval_validity, tightness, pval_tightness]

        # Also run upper bound tests
        kp = KP(-moments, SIG.to_numpy())                                          #NOTE: these input parameters are unused later - only necessary to call the class
        gamma_hat, wald, validity, tightness, exit_flag = kp.kp_output()
        pval_wald, pval_validity, pval_tightness  = kp.p_values(wald,validity,tightness,SIG.to_numpy(),num_sim = 10000)
        Table_UB[str(h)]=[wald, pval_wald, validity, pval_validity, tightness, pval_tightness]       
        
        
    Table.to_csv(TABLE_UNFORMATTED_OUTPUT + m +'.csv')
    Table_UB.to_csv(TABLE_UNFORMATTED_OUTPUT + m +'_UB.csv')
    toc = time.perf_counter()
    print(f"Calculations took {toc - tic:0.4f} seconds \r\n") 
    
    
    
#%%########################################################################
# Tabulate the results
###########################################################################  
# output and clean-up tex tables
def tex_wrapper(table, file_name):
    table.to_latex(file_name +'_pre.tex', escape=False, header = my_header)
    # make latex file a fragment (comment begin/end tabular)
    file_pre = open(file_name +'_pre.tex', 'r')
    file_out = open(file_name +'.tex', 'w')
    checkWords = ('\\begin', '\end', '\\toprule')
    repWords = ('%\\begin', '%\end', '%\\toprule')
    for line in file_pre:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        file_out.write(line)
    file_pre.close()
    file_out.close()    
    os.remove(file_name +'_pre.tex')




# Lower bound - validity
indx    = ['MW Bound: All', 'KT Bound: $\delta <3$','KT Bound: $\delta \in$ [3,7]', 'KT Bound: $\delta >7$']
indx    = ['Martin-Wagner (All)', 'Kadan-Tang \\\\ \\hspace{2em} Conservative','\\hspace{2em} Liberal', '\\hspace{2em} Other']
colindx = ['1','3','6','12']
Table = pd.DataFrame(dtype=str,index=indx,columns=colindx)
models = ['mw', 'kt_lt3', 'kt_3_7', 'kt_gt7']
for i,m in enumerate(models):
    table = pd.read_csv(TABLE_UNFORMATTED_OUTPUT + m +'.csv', index_col=0)
    # Validity p-values are in index 3
    Table.iloc[i]=table.iloc[3].apply(lambda x: f'{x:.1f}')
print('Lower Bound Validity Tests')
print(Table)
my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in Table.columns]
tex_wrapper(table=Table, file_name=TABLE_OUTPUT + 'lb_valid')



# Upper bound - validity
indx    = ['MW Bound: All',  'KT Bound: $\delta <3$','KT Bound: $\delta \in$ [3,7]', 'KT Bound: $\delta >7$']
indx    = ['Martin-Wagner (All)', 'Kadan-Tang \\\\ \\hspace{2em} Conservative','\\hspace{2em} Liberal', '\\hspace{2em} Other']
colindx = ['1','3','6','12']
Table = pd.DataFrame(dtype=str,index=indx,columns=colindx)
models = ['mw', 'kt_lt3', 'kt_3_7', 'kt_gt7']
for i,m in enumerate(models):
    table = pd.read_csv(TABLE_UNFORMATTED_OUTPUT + m +'_UB.csv', index_col=0)
    # Validity p-values are in index 3
    Table.iloc[i]=table.iloc[3].apply(lambda x: f'{x:.1f}')
print('Upper Bound Validity Tests')
print(Table)
my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in Table.columns]
tex_wrapper(table=Table, file_name=TABLE_OUTPUT + 'ub_valid')



# Lower bound - tightness
indx    = ['MW Bound: All', 'KT Bound: $\delta <3$']
indx    = ['Martin-Wagner (All)', 'Kadan-Tang (Conservative)']
colindx = ['1','3','6','12']
Table = pd.DataFrame(dtype=str,index=indx,columns=colindx)
models = ['mw', 'kt_lt3']
for i,m in enumerate(models):
    table = pd.read_csv(TABLE_UNFORMATTED_OUTPUT + m +'.csv', index_col=0)
    # Tightness p-values are in index 5
    Table.iloc[i]=table.iloc[5].apply(lambda x: f'{x:.1f}')
print('Lower Bound Tightness Tests')
print(Table)
my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in Table.columns]
tex_wrapper(table=Table, file_name=TABLE_OUTPUT + 'lb_tight')



# Upper bound - tightness
indx    = ['KT Bound: $\delta \in$ [3,7]', 'KT Bound: $\delta >7$']
indx    = ['Kadan-Tang (Liberal)', 'Kadan-Tang (Other)']
colindx = ['1','3','6','12']
Table = pd.DataFrame(dtype=str,index=indx,columns=colindx)
models = ['kt_3_7', 'kt_gt7']
for i,m in enumerate(models):
    table = pd.read_csv(TABLE_UNFORMATTED_OUTPUT + m +'_UB.csv', index_col=0)
    # Tightness p-values are in index 5
    Table.iloc[i]=table.iloc[5].apply(lambda x: f'{x:.1f}')
print('Upper Bound Tightness Tests')
print(Table)
my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in Table.columns]
tex_wrapper(table=Table, file_name=TABLE_OUTPUT + 'ub_tight')

# remove the unnecessary files and folders
shutil.rmtree(path + 'intermediate/bootstrap')
shutil.rmtree(path + 'intermediate/validity_tightness')