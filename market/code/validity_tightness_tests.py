"""
This code runs the Kodde/Palm tests on the market bounds.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import os
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
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
SIM_DATA  = INTERMEDIATE_DIR + 'bootstrap_kp_results.csv'

# output of the script 
OUTPUT= TABLES_DIR + 'valid_tight_tests_'             # suffix for bound identifier added below

# load KPClass.py
os.chdir(CODE_DIR)
from KPClass import KP


#%% ===========================================================================
# Read data
# =============================================================================
# import the dataset of bounds, returns, and conditioning variables
ds = pd.read_csv(DS_DAILY)

# import simulated finite-sample distribution of test stats
sim_res = pd.read_csv(SIM_DATA, dtype = {'horizon':str}).rename(columns={'B - E[R] = 0.0%':0})
sim_res = sim_res.set_index(['bound', 'horizon', 'stats', 'sim_num'])

#%% ===========================================================================
#  Simulated finite-sample distribution of test stats
# =============================================================================
def MonteCarloPValue(D,
                     sim_res,
                     stat_type = 'validity',
                     bound = 'lb_m', 
                     horizon = '1',
                     kvals = np.linspace(-5,5,21)/100,
                     tol = 1.0e-8):
    """
    
    Parameters
    ----------
    D : The statistic for which p-Value is to be calculated.
    sim_res : A dictionary containing the simulation results .
    stat_type : optional
        The type of the statistic; could be 'validity', 'tightness', or 'wald.
        The default is 'validity'.
    bound : optional
        Name of the bound in the original data. The default is 'lb_m'.
    horizon: optional
        Horizon of the bound (default is '1')
    kvals : optional
        Vlaues of B - E[R] for which simulations have been run.
        The default is np.linspace(-5,5,21)/100.

    Returns
    -------
    pval : p-Value of D.

    """
    # Note: sim_res[0.0] gives the tight bound simulation
    stat_vector = sim_res.loc[(bound, horizon, stat_type), 0]
    s = np.where(stat_vector < tol, 0, stat_vector)    
    numSims = len(s)
    numSims_GE_D = (s >= D).sum() 
    pval = 100*numSims_GE_D/numSims
    return pval

    

#%% ===========================================================================
# Run the tests
# =============================================================================
for bound in ['m','cylr','m_orig','cylr_orig']:         
    
    print(bound)
    
    # empty dataframe, columns are bounds/horizons, rows are quintiles Estimates and Std Errors
    indx   = ['p(a=0, b=1)-F', 'Validity (D_1)', 'p(D_1) - Asymptotic', 'p(D_1) - Finite Sample', 'Tightness (D_2)', 'p(D_2) - Asymptotic','p(D_2) - Finite Sample','N1', 'N2']
    colindx= ['1','3','6','12']
    Table  = pd.DataFrame(dtype=float,index=indx,columns=colindx)
    Table2 = pd.DataFrame(dtype=str,index=Table.index,columns=Table.columns)           #string, formatted version    

    for i, h in enumerate([1,3,6,12]):    
        print(str(h))
               
        ##### KP Test
        gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']
        
        # Hansen-Hodrick Sigma:
        m = KP(ds, h, np.ceil(h*21).astype(int), 'hh', 'lb_'+bound, gw_vars)
        Sigma_hat = m.Sigma()
        
        # If not positive semi-definite, use Newey-West Sigma:
        if np.all(np.linalg.eigvals(m.Sigma()) >= 0)==0:
            print('Hansen-Hodrick Error - Using Newey-West')
            m = KP(ds, h, np.ceil(h*21*1.5).astype(int), 'nw', 'lb_'+bound, gw_vars)
            Sigma_hat = m.Sigma() 
        gamma_hat, wald, validity, tightness, exit_flag = m.kp_output()
       
        print('Wald test for horizon ' + str(h) + ':\t' + f'{wald : 5.2f}')
        print('Validity test for horizon ' + str(h) + ':\t' + f'{validity : 5.2f}')
        print('Tightness test for horizon ' + str(h) + ':\t' + f'{tightness : 5.2f}  \r\n')        
        pval_wald, pval_validity, pval_tightness  = m.p_values(wald,validity,tightness,Sigma_hat,num_sim = 10000)           

        # add results to table
        Table[str(h)].loc['Validity (D_1)']  = validity
        Table[str(h)].loc['Tightness (D_2)'] = tightness
        Table[str(h)].loc['N2'] = m.T
        Table[str(h)].loc['p(D_1) - Asymptotic'] = pval_validity
        Table[str(h)].loc['p(D_2) - Asymptotic'] = pval_tightness   
        if bound =='m' or bound=='m_orig':
            bndlookup='lb_m'
        else:
            bndlookup='lb_cylr'
        
        Table[str(h)].loc['p(D_1) - Finite Sample'] = MonteCarloPValue(validity,  sim_res,'validity', bndlookup, str(h), tol=1.0e-8 )
        Table[str(h)].loc['p(D_2) - Finite Sample'] = MonteCarloPValue(tightness, sim_res,'tightness',bndlookup, str(h), tol=1.0e-8 )
        
        
        ##### F-test
        bnd = 'lb_{}_{}'.format(bound, h)
        fret='f_mktrf{}'.format(h)
        
        # To get the same observations as KP tests, drop observations with missing
        # GW variables
        I = ds['dp'].isnull()
        
        # Run regression
        x = sm.add_constant(ds.loc[~I, bnd])
        y = ds.loc[~I, fret]
        results = sm.OLS(y, x, missing='drop').fit(cov_type='HAC',cov_kwds={'maxlags':h*21, 'kernel': 'uniform'})

        # Test null of tight bounds
        hypotheses = '(const=0), ({}=1)'.format(bnd)
        test = results.f_test(hypotheses)

        # add results to table
        Table[str(h)].loc['p(a=0, b=1)-F'] = test.pvalue*100
        Table[str(h)].loc['N'] = results.nobs     

        
    print(Table.round(2))

    if bound=='m':
        Table_m = Table.copy()
    elif bound=='cylr':
        Table_cylr = Table.copy()
    elif bound=='m_orig':
        Table_m_orig = Table.copy()
    elif bound=='cylr_orig':
        Table_cylr_orig = Table.copy()
        
        
#%% ===========================================================================
# Function for cleaning up tex output
# =============================================================================
def tex_wrapper(table, file_name, header_list):
    table.to_latex(file_name +'_pre.tex', escape=False, header = header_list)
    
    # make latex file a fragment (comment begin/end tabular)
    file_pre = open(file_name +'_pre.tex', 'r')
    file_out = open(file_name +'.tex', 'w')
    checkWords = ('\\begin', '\end','label', '\\toprule')
    repWords = ('%\\begin', '%\end','%label', '%\\toprule')
    for line in file_pre:
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        file_out.write(line)
    file_pre.close()
    file_out.close()    
    os.remove(file_name +'_pre.tex')
    
    
    
#%% ===========================================================================
# output and clean-up tex tables    
# =============================================================================

tables = [Table_m, Table_cylr, Table_m_orig, Table_cylr_orig]
b_dict = {0: 'm', 1:'cylr', 2:'m_orig', 3:'cylr_orig'}
for j,t in enumerate(tables):
    tab = t.copy()
    print(tab.round(4))
    
    indx   = ['p($F$ test of $\\alpha=0, \\beta=1$)', '$D_1$ (Validity)', 'p($D_1$) - Asymptotic', 'p($D_1$) - Finite Sample', '$D_2$ (Tightness)', 'p($D_2$) - Asymptotic', 'p($D_2$) - Finite Sample']
    # indx   = ['p($t$ test of $E[r-b]=0$)', 'p($F$ test of $\\alpha=0, \\beta=1$)', '$D_1$ (Validity)', 'p($D_1$) - Asymptotic', 'p($D_1$) - Finite Sample', '$D_2$ (Tightness)', 'p($D_2$) - Asymptotic', 'p($D_2$) - Finite Sample']
    colindx= ['1','3','6','12']
    Table2 = pd.DataFrame(dtype=str,index=indx,columns=colindx)           #string, formatted version  
    
    for i in np.arange(len(indx)):
        if i ==1 or i==4:
            Table2.iloc[i]   = tab.iloc[i].apply(lambda x: f'{x:.2f}').values
        else:
            Table2.iloc[i]   = tab.iloc[i].apply(lambda x: f'{x:.1f}').values

    print(Table2 , '\n')   
    my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in Table2.columns]
    
    tex_wrapper(table=Table2, file_name=OUTPUT + b_dict[j], header_list=my_header)     
    


