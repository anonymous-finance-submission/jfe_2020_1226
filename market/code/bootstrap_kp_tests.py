"""
This file performs short sample simulations of the Kodde/Palm tests. It follows 
the simulation algorithm described in appendix B.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE, NUMBER_OF_CORES
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvnorm
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import wrds
from arch.bootstrap import CircularBlockBootstrap
from numpy.random import RandomState

#%%============================================================================
# define directories
# =============================================================================

project_dir         = PATH_TO_REPLICATION_PACKAGE + 'market/'
interm_dir          = project_dir + 'intermediate/'

# input data that is used in this script
DS_DAILY            = interm_dir + 'ds_mkt_daily.csv'

# output data 
OUTPUTFILE          = interm_dir + 'bootstrap_kp_results.csv'
OUTPUTFIG           = project_dir + 'figures/bootstrap_kp_power_{}.pdf'

os.chdir(project_dir +'code/')
from KPClass import KP


#%%============================================================================
# Estimate parameters of predictive system
# =============================================================================
gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']
n = len(gw_vars)

# define the estimation function
def estimate(df, ret = 'f_mktrf1', bound = 'lb_m_1', h = 1, daily_ret = 'mktrf_d'):
    # calibrate - output is bbar, a, theta, var_b, discounts, Sig_uv, xbar, A,
    # Sig_w, Sig_x, B, Sig_e
    
    ds = df.copy()
    
    # calculate mean slackness
    s = (ds[ret] - ds[bound]).mean()
    
    # fit an AR(1) to the bound
    b = ds[bound].dropna()
    result = AutoReg(b.to_numpy(),lags=1).fit()
    a = result.params[1]
    bbar = result.params[0] / (1-a)
    theta_times_u = pd.Series(result.resid, index=b.index[1:])
    
    # we need the stationary variance of the bound to initialize a simulation
    var_b = theta_times_u.var() / (1-a*a)
    
    # compute theta and u's
    theta = (1-a**(h*21)) / (1-a)
    u = theta_times_u / theta

    # compute v's
    mu = (bbar+s)/(21*h) + (b-bbar)/theta
    v = ds[daily_ret] - mu.shift()
    
    # estimate VAR(1) for logs of GW variables
    n = len(gw_vars)
    X = ds[gw_vars].dropna().resample('M').last()
    result = VAR(X).fit(maxlags=1)
    A = result.params.iloc[1:].T
    A.columns = gw_vars
    xbar = np.linalg.inv(np.identity(n) - A) @ result.params.iloc[0].T
    w = result.resid
    Sig_w = w.cov()
    
    # we'll need the stationary covariance matrix of the log GW variables to initialize a simulation
    vec_Sig_w = Sig_w.to_numpy().reshape((n*n,1))
    Sig_x = np.linalg.solve(np.identity(n*n)-np.kron(A,A),vec_Sig_w).reshape(n,n)
    Sig_x = pd.DataFrame(Sig_x,index=gw_vars,columns=gw_vars)
    

    return [bbar, a, theta, var_b, xbar, A, Sig_x, s], u, v, w

#%%============================================================================
# Define the simulation function
# =============================================================================

# set hyperparameters of the simulation
numSims = 1000


# Number of expecture return standard deviations added to mu_t to create bound
kvals = np.linspace(-8,8, num = 17)/100
# kvals = np.array([0])

# define the simulation function; output is daily r, daily b, and monthly gw.
# r is generated with slackness=0, just add slackness to r when needed
def simulate(attempt, params, bs_uv, bs_w, h):
    
    numDays = len(bs_uv)
    numMonths= len(bs_w)

    # read-in the estimated parameters
    bbar       = params[0]
    a          = params[1]
    theta      = params[2]
    var_b      = params[3]
    xbar       = params[4]
    A          = params[5]
    Sig_x      = params[6]
    sbar       = params[7]

    # initiate the results 
    exit_flag       = np.zeros((len(kvals),))
    wald            = np.zeros((len(kvals),))
    validity        = np.zeros((len(kvals),))
    tightness       = np.zeros((len(kvals),))
    pval_wald       = np.zeros((len(kvals),))
    pval_validity   = np.zeros((len(kvals),))
    pval_tightness  = np.zeros((len(kvals),))
    
    # Draw random innovations for u and w
    np.random.seed(attempt)
    
    # generate b_0
    b0 = bbar + np.sqrt(var_b)*norm.rvs()
    
    # initial value b0 occurs at index=-1
    ds_daily = pd.DataFrame(dtype=float,columns=['bs_month','u','v','b','mu','z','r'],index=np.arange(numDays+1)-1)
    ds_daily.loc[0:,'bs_month']=bs_uv.bs_month
    ds_daily.loc[0:,'u'] = bs_uv.u
    ds_daily.loc[0:,'v'] = bs_uv.v
    ds_daily.index.name='day'
    
    # generate the bound
    ds_daily.loc[-1,'b'] = b0
    for day in range(0,numDays) :
        ds_daily.loc[day,'b'] = (1-a)*bbar + a*ds_daily.loc[day-1,'b'] + theta*ds_daily.u.loc[day]
    
    # generate forward returns assuming bounds have no predictive power;
    # i.e. b does not predict r. So the coeff of b in r must be set to zero    
    ds_daily.mu = bbar/(21*h) + (ds_daily.b-bbar)/theta
    ds_daily.z = ds_daily.mu.shift() + ds_daily.v
    ds_daily.r = ds_daily.z.rolling(21*h).sum().shift(-21*h)   
    
    # generate GW variables; first observation is index=-1
    x0 = xbar + mvnorm.rvs(cov=Sig_x) 
    gw = pd.DataFrame(dtype=float,index=np.arange(numMonths+1)-1,columns=gw_vars)
    gw.index.name = 'month'
    gw.loc[-1] = x0
    for month in range(0,numMonths) :
        gw.loc[month] = (np.identity(n)-A) @ xbar +A @ gw.loc[month-1].T + bs_w.loc[month]
    gw = np.exp(gw)
   
    # put all the simulated variables in a dataset
    # Keep end-of-month GW values and forward fill to days within the next month
    eom = ds_daily.groupby([ds_daily.bs_month]).tail(1).copy()
    eom['day']=eom.index
    eom = eom[['day','bs_month']]
    gw = pd.merge(gw, eom, how='inner', left_on=['month'], right_on=['bs_month'])
    gw.set_index('day', inplace=True)
    ds = ds_daily.merge(gw,how='outer',left_index=True,right_index=True)
    ds.sort_index(inplace=True)
    for i in gw_vars:
        ds[i] = ds[i].fillna(method='ffill')
    ds.drop(columns=['bs_month_y','u','v','z','mu'],inplace=True)
    ds = ds.rename(columns = {'r':'f_mktrf'+str(h), 'b':'bound_'+str(h)})
    
    for i,k in enumerate(kvals):
        # make bounds slack by shifting the returns
        ds['f_mktrf' + str(h)] -= k/(12/h)
        
        # run KP tests; try Hansen-Hodrik first, if it does not work, use Newey-West
        try:
            m = KP(ds, h, 21*h, 'hh', 'bound', gw_vars)
            gamma_hat, wald[i], validity[i], tightness[i], exit_flag[i] = m.kp_output()
            pval_wald[i], pval_validity[i], pval_tightness[i] = m.p_values(wald[i], validity[i], tightness[i], m.Sigma(), num_sim = 1000)
        except Exception:
            m = KP(ds, h, int(np.ceil(21*h*1.5)), 'nw', 'bound', gw_vars)
            gamma_hat, wald[i], validity[i], tightness[i], exit_flag[i] = m.kp_output()
            pval_wald[i], pval_validity[i], pval_tightness[i] = m.p_values(wald[i], validity[i], tightness[i], m.Sigma(), num_sim = 1000)
        
        # shift returns back to tight
        ds['f_mktrf' + str(h)] += k/(12/h)

    # return the results of the KP tests
    return (attempt, exit_flag, wald, validity, tightness, pval_wald, pval_validity, pval_tightness)

#%%============================================================================
# Simulate the predictive system
# =============================================================================
# read data
ds = pd.read_csv(DS_DAILY, index_col='date', parse_dates=['date'])

# get daily S&P 500 returns
conn=wrds.Connection()
sprets = conn.raw_sql(" select caldt, vwretd from crsp.dsp500p where caldt >= '01/01/1990' ")
sprets.caldt = pd.to_datetime(sprets.caldt)
sprets.set_index('caldt',inplace=True)
sprets.index.name = 'date'

# get daily risk-free rate
ff = conn.raw_sql(" select date, rf from ff.factors_daily where date >= '01/01/1990' ")
ff.date = pd.to_datetime(ff.date)
ff.set_index('date',inplace=True)
conn.close()

# calculate daily excess returns
sprets = sprets.merge(ff,how='left',left_index=True,right_index=True)
ds['mktrf_d'] = sprets['vwretd'] - sprets['rf']

# convert bounds and forward returns to non-annualized decimal
for h in [1,3,6,12] :
    ds['f_mktrf'+str(h)] = ds['f_mktrf'+str(h)] / (100*12/h) 
    for bd in ['lb_m_','lb_cylr_'] :
        ds[bd+str(h)] = ds[bd+str(h)] / (100*12/h)
    
# take logs of GW variables
gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']
ds[gw_vars] = np.log(ds[gw_vars])

# generate a dataframe for storing the results
cols = [f'B - E[R] = {100*x:.1f}%' for x in kvals]
sims = list(range(1, numSims+1))
bounds = ['lb_m','lb_cylr']
horizons = ['1','3','6','12']
stats = ['wald', 'validity', 'tightness', 'pval_wald', 'pval_validity', 'pval_tightness']                                  
idx = pd.MultiIndex.from_product([bounds, horizons, stats, sims], names = ['bound','horizon','stats','sim_num'])
sim_res = pd.DataFrame(dtype=float, index=idx, columns=cols)

# run the simulation
for b in bounds:
    for h in [1,3,6,12]:
    
        print('Simulating for ' + b + '_' + str(h) + ' ...')
    
        # estimate the parameters
        params, u, v, w = estimate(ds,ret = 'f_mktrf' + str(h), bound = b + '_' + str(h), h=h)
        
        uv = pd.DataFrame()
        uv['u'] = u
        uv['v'] = v
        
        # bootstrap the residuals
        w['mdate'] = pd.PeriodIndex(year=w.reset_index().date.dt.year, month=w.reset_index().date.dt.month, freq='M')
        uv['mdate']= pd.PeriodIndex(year=uv.reset_index().date.dt.year, month=uv.reset_index().date.dt.month, freq='M')
        uv['dt'] = uv.index
        
        # bootstrap generator; 1st parameter is block size, 2nd parm is dataframe to draw from
        rs = RandomState(1000+h)
        bs = CircularBlockBootstrap(12, w.mdate, random_state=rs) 

        # collect bootstrapped residuals
        bs_uv = []
        bs_w = []
        for i, data in enumerate(bs.bootstrap(numSims)):
        
            bs_data = data[0][0].reset_index()
            obs_num = pd.DataFrame(range(len(bs_data)), index=bs_data.index, columns=['bs_month'])
            bs_data = pd.merge(bs_data.mdate, obs_num, how='left',left_index=True, right_index=True)
            
            # Pull daily observations of uv residuals for each sampled month
            bs_uv.append(pd.merge(bs_data, uv, how='left', left_on='mdate', right_on ='mdate'))
            
            # Pull monthly observations of w residual vector (GW VAR)
            bs_w.append(pd.merge(bs_data, w,  how='left', left_on='mdate', right_on ='mdate')[gw_vars])
                        
        # simulate the tests in parallel.
        pool = mp.Pool(NUMBER_OF_CORES)            
        results = [pool.apply_async(simulate, args = (i, params, bs_uv[i], bs_w[i], h)) for i in range(numSims)]
        pool.close()
    
        results = [r.get() for r in results]
    
        # collect the results        
        exit_flag       = np.zeros((numSims,len(kvals)))
        wald            = np.zeros((numSims,len(kvals)))
        validity        = np.zeros((numSims,len(kvals)))
        tightness       = np.zeros((numSims,len(kvals)))
        pval_wald       = np.zeros((numSims,len(kvals)))
        pval_validity   = np.zeros((numSims,len(kvals)))
        pval_tightness  = np.zeros((numSims,len(kvals)))
    
        for row, result in enumerate(results):
            exit_flag[row,:] = result[1]
            wald[row,:] = result[2]
            validity[row,:] = result[3]
            tightness[row,:] = result[4]
            pval_wald[row,:] = result[5]
            pval_validity[row,:] = result[6]
            pval_tightness[row,:] = result[7]
    
        # save the results into a dictionary
        this_sim_res = {'validity':validity, 'tightness':tightness,'wald':wald,
                    'pval_validity':pval_validity, 'pval_tightness':pval_tightness,
                    'pval_wald':pval_wald}
        
        # update the results        
        for stat in this_sim_res.keys():
            sim_res.loc[(b, str(h), stat), :] = this_sim_res[stat]


# save all the results
sim_res.to_csv(OUTPUTFILE)

#%%============================================================================
# Generate figure R.1
# =============================================================================
# load the data for the particular combination of multiples and correlation
validity = sim_res.loc[('lb_m','1','validity'),:]
tightness = sim_res.loc[('lb_m','1','tightness'),:]

pval_validity = sim_res.loc[('lb_m','1','pval_validity'),:]
pval_tightness = sim_res.loc[('lb_m','1','pval_tightness'),:]

# get the statistics under the null and calculate critical values
critical_values_validity = np.percentile(validity['B - E[R] = 0.0%'].to_numpy(), [90,95,99])
critical_values_validity = dict(zip([10,5,1], critical_values_validity))
critical_values_tightness = np.percentile(tightness['B - E[R] = 0.0%'].to_numpy(), [90,95,99])
critical_values_tightness  = dict(zip([10,5,1], critical_values_tightness ))

sig = 10

# parameters of the figure
label_font_size = 13
title_font_size = 13

# plot the results; it's a 3x1 plot with (upper) validity, (middle) tightness, 
# and (lower) the inference
fig = plt.figure(figsize = (13,9))
grid = fig.add_gridspec(2, 4, hspace = 0.3, wspace = 0.5)
axs1 = fig.add_subplot(grid[0,:-2])
axs2 = fig.add_subplot(grid[0,2:])
axs3 = fig.add_subplot(grid[1,1:-1])

# the validity plot; subplot (a)
y1 = 100*np.sum(validity >= critical_values_validity[1], axis = 0)/numSims
y5 = 100*np.sum(validity >= critical_values_validity[5], axis = 0)/numSims
y10 = 100*np.sum(validity >= critical_values_validity[10], axis = 0)/numSims
axs1.fill_between(kvals*100, 0.0*kvals, y1, facecolor = 'darkblue', label = '1%')
axs1.fill_between(kvals*100, y1, y5, facecolor = 'blue', label = '5%')
axs1.fill_between(kvals*100, y5, y10, facecolor = 'cornflowerblue', label = '10%')
axs1.set_ylabel('Rejection Rate (%)', fontsize = label_font_size)
axs1.set_xlabel('Bound in Excess of Expected Return (%)', fontsize = label_font_size)
axs1.set_xticks(range(-8,9))
axs1.legend(loc = 'upper left')
axs1.set_title('(a) Validity Tests', fontsize = title_font_size, fontstyle = 'italic')
axs1.grid()

# the tightness plot; subplot (b)
y1 = 100*np.sum(tightness >= critical_values_tightness[1], axis = 0)/numSims
y5 = 100*np.sum(tightness >= critical_values_tightness[5], axis = 0)/numSims
y10 = 100*np.sum(tightness >= critical_values_tightness[10], axis = 0)/numSims
axs2.fill_between(kvals*100, 0.0*kvals, y1, facecolor = 'darkblue', label = '1%')
axs2.fill_between(kvals*100, y1, y5, facecolor = 'blue', label = '5%')
axs2.fill_between(kvals*100, y5, y10, facecolor = 'cornflowerblue', label = '10%')
axs2.set_ylabel('Rejection Rate (%)', fontsize = label_font_size)
axs2.set_xlabel('Bound in Excess of Expected Return (%)', fontsize = label_font_size)
axs2.set_xticks(range(-8,9))
axs2.legend(loc = 'upper right')
axs2.set_title('(b) Tightness Tests', fontsize = title_font_size, fontstyle = 'italic')
axs2.grid()

# the inference plot; subplot (c)
y1 = 100*np.sum((validity < critical_values_validity[sig]) & (tightness >= critical_values_tightness[sig]), axis = 0)/numSims
y2 = 100*np.sum((validity < critical_values_validity[sig]) & (tightness < critical_values_tightness[sig]), axis = 0)/numSims
y3 = 100*np.sum(validity >= critical_values_validity[sig], axis = 0)/numSims
axs3.fill_between(kvals*100, 0.0*kvals, y1, hatch = '///', facecolor = 'darkblue', label = 'Slack lower bound')
axs3.fill_between(kvals*100, y1, y1 + y2, facecolor = 'blue', label = 'Valid tight bound')
axs3.fill_between(kvals*100, y1 + y2, y1 + y2 + y3, hatch = 'xxx', facecolor = 'cornflowerblue', label = 'Invalid lower bound')
axs3.set_ylabel('% of Simulations', fontsize = label_font_size)
axs3.set_xlabel('Bound in Excess of Expected Return (%)', fontsize = label_font_size)
axs3.set_xticks(range(-8,9))
axs3.legend(loc = 'upper right')
axs3.set_title('(c) Inference on Validity and Tightness', fontsize = title_font_size, fontstyle = 'italic')
axs3.grid()

# save the plot
fig.savefig(OUTPUTFIG.format(sig))