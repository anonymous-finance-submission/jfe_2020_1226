from __main__ import PATH_TO_REPLICATION_PACKAGE, NUMBER_OF_CORES
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import numpy as np
import multiprocessing as mp
sns.set()

# input data that is used in this script
PATH                   = PATH_TO_REPLICATION_PACKAGE + 'market/'
INTERMEDIATE_DIR       = PATH + 'intermediate/'
FIGURES_DIR            = PATH + 'figures/'
INPUT                  = INTERMEDIATE_DIR + 'ds_mkt_daily.csv'
SIM_DATA               = INTERMEDIATE_DIR + 'bootstrap_kp_results.csv'

# import simulated finite-sample distribution of test stats
os.chdir(INTERMEDIATE_DIR)
sim_res = pd.read_csv(SIM_DATA, dtype = {'horizon':str}).rename(columns={'B - E[R] = 0.0%':0})
sim_res = sim_res.set_index(['bound', 'horizon', 'stats', 'sim_num'])

# output of the script 
FIG_VALIDITYP_M        = FIGURES_DIR + 'fig_bound_plus_constant_validity_pval_lb_m.pdf'
FIG_VALIDITYP_CYL      = FIGURES_DIR + 'fig_bound_plus_constant_validity_pval_lb_cyl.pdf'


# import the dataset
ds = pd.read_csv(INPUT)

#%% ===========================================================================
# Simulated finite-sample distribution of test stats
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
# load KPClass.py
# =============================================================================

os.chdir(PATH +'code/')
from KPClass import KP

#%% ===========================================================================
# Function to run the tests
# =============================================================================

def kp_test_add_constant(ds, bound, h, constant):
    '''
    Parameters
    ----------
    ds: analysis dataset
    bound :   ['m','cylr']
    h :       [1,3,6,12]
    constant: np.arange(0,16)

    Returns
    -------
    D_1, p_D_1_finite, D_2, p_D_2_finite
    '''
    
    #Add constant to the bounds   
    ds['lb_' + bound +'_'+str(h)] = ds['lb_' + bound +'_'+str(h)] + constant
    
    ##### KP Test
    gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']
        
    # Hansen-Hodrick Sigma:
    m = KP(ds, h, np.ceil(h*21).astype(int), 'hh', 'lb_'+bound, gw_vars)
    Sigma_hat = m.Sigma()
    
    # If not positive semi-definite, use Newey-West Sigma:
    if np.all(np.linalg.eigvals(m.Sigma()) >= 0)==0:
        m = KP(ds, h, np.ceil(h*21*1.5).astype(int), 'nw', 'lb_'+bound, gw_vars)
        Sigma_hat = m.Sigma() 
    gamma_hat, wald, validity, tightness, exit_flag = m.kp_output()
    # pval_wald, pval_validity_asym, pval_tightness_asym  = m.p_values(wald,validity,tightness,Sigma_hat,num_sim = 10000)           
    
    if bound =='m' or bound=='m_orig':
        bndlookup='lb_m'
    else:
        bndlookup='lb_cylr'
    
    pval_validity_finite = MonteCarloPValue(validity,  sim_res,'validity', bndlookup, str(h), tol=1.0e-8 )
    pval_tightness_finite= MonteCarloPValue(tightness, sim_res,'tightness',bndlookup, str(h), tol=1.0e-8 )
    
    return (validity, tightness, pval_validity_finite, pval_tightness_finite)



#%% ===========================================================================
# Run the tests
# =============================================================================

# Create empty dataframe to store results
cols = ['validity', 'tightness', 'pval_validity_finite', 'pval_tightness_finite']
# cols = ['validity', 'tightness', 'pval_validity_asym', 'pval_validity_finite', 'pval_tightness_asym', 'pval_tightness_finite']
indx = pd.MultiIndex.from_product([['m','cylr'],[1,3,6,12], np.arange(0,16)])
Table = pd.DataFrame(dtype=float,columns=cols, index=indx)

import datetime as dt
start = dt.datetime.now()

for bound in ['m','cylr']:          
    for h in [1,3,6,12]:    
        print('Simulating for ' + bound + '_' + str(h) + ' ...')
            
        # simulate the tests in parallel.
        pool = mp.Pool(NUMBER_OF_CORES)            
        results = [pool.apply_async(kp_test_add_constant, args = (ds, bound, h, i)) for i in np.arange(0,16)]
        pool.close()
    

        # collect the results        
        results = [r.get() for r in results]
        results_df = pd.DataFrame(results, columns=cols)       
        Table.loc[(bound,h)].loc[results_df.index] = results_df    
        
end = dt.datetime.now()
print("Total time was {} seconds.".format((end-start).total_seconds()))

        
#%% ===========================================================================
# Plot p-values and average slackness at each horizon
# =============================================================================


for x in ['m','cylr']:
    fig, ax = plt.subplots()   
    styles = ['-', '--', '-.', ':']
    for i, h in enumerate([1,3,6,12]):
        sub = Table.loc[(x,h)]
        ax.plot(sub.index.values, sub.pval_validity_finite, styles[i], label=str(h)+'-month')
        # ax.plot(sub.index.values, sub.pval_validity_asym, styles[i], label=str(h)+'-month') 
 
    for i, h in enumerate([1,3,6,12]):
        avg_slack = (ds['f_mktrf'+str(h)]-ds['lb_'+ x+'_' +str(h)]).mean()   
        ax.axvline(avg_slack, color='k', linestyle = styles[i], linewidth=1.0, label=str(h)+'-mo Avg Slackness')
    if x=='m':
        ax.set_title('Martin Bound', fontsize=10)
    elif x =='cylr':
        ax.set_title('CYL Bound', fontsize=10)
    ax.set_xlabel("constant added to bound",fontsize=10)
    ax.set_ylabel(r'Validity Test p-value',fontsize=10)
    ax.set_ylim(0,101.0)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc='upper right',frameon=False, ncol=2,fontsize=8)
    
    # save figure
    if x=='m':
        fig.savefig(FIG_VALIDITYP_M)
    elif x =='cylr':
        fig.savefig(FIG_VALIDITYP_CYL)
    
        