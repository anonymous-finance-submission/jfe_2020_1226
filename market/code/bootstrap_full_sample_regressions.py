"""
This file performs short sample simulations of the Kodde/Palm tests. 
It matches the simulation algorithm described in appendix B and performs the 
simulation for all horizons.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
from __main__ import NUMBER_OF_CORES
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvnorm
import multiprocessing as mp
import wrds
from numpy.random import RandomState
from arch.bootstrap import CircularBlockBootstrap

#%%============================================================================
# define directories
# =============================================================================
# paths
PATH = PATH_TO_REPLICATION_PACKAGE + 'market/'
INTERMEDIATE_DIR = PATH + 'intermediate/'   

# input data that is used in this script
DS_DAILY        = INTERMEDIATE_DIR + 'ds_mkt_daily.csv'

# output of the script
OUTPUTFILE      = INTERMEDIATE_DIR + 'bootstrap_fs_regs.csv'

# set hyperparameters of the simulation
numSims = 1000

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
# Define the daily simulation function
# =============================================================================

''' 
run simulation - output is predictive regression coefficient and t-stat 
from daily univariate regression
'''

# define the simulation function
def sim_daily_nopredictability(attempt, params, bs_uv, h):
    
    numDays = len(bs_uv)

    # read-in the estimated parameters
    bbar       = params[0]
    a          = params[1]
    theta      = params[2]
    var_b      = params[3]
    # xbar       = params[4]
    # A          = params[5]
    # Sig_x      = params[6]
    sbar       = params[7]

    # Draw random innovations for u and w
    np.random.seed(attempt)
    
    # generate b_0
    b0 = bbar + np.sqrt(var_b)*norm.rvs()
    
    # initial value b0 occurs at index=-1
    ds_daily = pd.DataFrame(dtype=float,columns=['u','v','b','mu','z','r'],index=np.arange(numDays+1)-1)
    ds_daily.loc[0:,'u'] = bs_uv.u
    ds_daily.loc[0:,'v'] = bs_uv.v
    ds_daily.index.name='day'
    
    # generate bound
    ds_daily.loc[-1,'b'] = b0
    for day in range(0,numDays) :
        ds_daily.loc[day,'b'] = (1-a)*bbar + a*ds_daily.loc[day-1,'b'] + theta*ds_daily.u.loc[day]
    
    # generate forward returns assuming bounds have no predictive power;
    # i.e. b does not predict r. So the coeff of b in r must be set to zero    
    ds_daily.mu = (bbar+sbar)/(21*h) + 0.0*(ds_daily.b-bbar)/theta
    ds_daily.z = ds_daily.mu.shift() + ds_daily.v
    ds_daily.r = ds_daily.z.rolling(21*h).sum().shift(-21*h)   

    ds_daily = ds_daily.rename(columns = {'r':'ret', 'b':'bound'})
    
    # Run the daily univariate predictive regression
    ds_daily['const'] = 1.0
    results = sm.OLS(ds_daily['ret'], ds_daily[['const','bound']], missing='drop').fit(cov_type='HAC',
                                                                           cov_kwds={'maxlags':21*h, 'kernel': 'uniform'})
    # save the results
    b = results.params['bound']
    t = results.tvalues['bound']
        
    return (b, t)

#%%============================================================================
# Define the monthly simulation function
# =============================================================================
# define the simulation function;output is predictive regression coefficient 
# and t-stat from monthly multivariate regression
def sim_monthly_nopredictability(attempt, params, bs_uv, bs_w, h):
    
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
    
    # generate bound
    ds_daily.loc[-1,'b'] = b0
    for day in range(0,numDays) :
        ds_daily.loc[day,'b'] = (1-a)*bbar + a*ds_daily.loc[day-1,'b'] + theta*ds_daily.u.loc[day]
    
    # generate forward returns assuming bounds have no predictive power;
    # i.e. b does not predict r. So the coeff of b in r must be set to zero    
    ds_daily.mu = (bbar+sbar)/(21*h) + 0.0*(ds_daily.b-bbar)/theta
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
    ds = ds.rename(columns = {'bs_month_x':'bs_month', 'r':'ret', 'b':'bound'})   
    ds['const'] = 1.0
    
    # Run the monthly multivariate predictive regression   
    ds_monthly = ds.groupby('bs_month').last()
    results = sm.OLS(ds_monthly['ret'], ds_monthly[['const','bound']+gw_vars], missing='drop').fit(cov_type='HAC',
                                                                           cov_kwds={'maxlags':h, 'kernel': 'uniform'})
    # save the results
    b = results.params['bound']
    t = results.tvalues['bound']
    
    return (b, t)

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
cols = ['b', 't(b)']
sims = list(range(0, numSims))
bounds = ['lb_m','lb_cylr']
horizons = [1,3,6,12]
freq = ['daily', 'monthly']
idx = pd.MultiIndex.from_product([bounds, horizons, freq, sims], names = ['bound','horizon','frequency', 'sim_num'])
sim_res = pd.DataFrame(dtype=float, index=idx, columns=cols)


import datetime as dt
start = dt.datetime.now()

# main simulation loop
for b in bounds:
    for h in horizons:
        
        print('Simulating for ' + b + '_' + str(h) + ' ...')  
        
        # estimate the model 
        params, u, v, w = estimate(ds,ret = 'f_mktrf' + str(h), bound = b + '_' + str(h), h= h) 
        uv = pd.DataFrame(u,columns=['u']).merge(pd.DataFrame(v,columns=['v']),left_index=True, right_index=True)
        
        # -------------- daily model simulation -------------------- #
        
        # bootstrap generator; 1st parm is block size, 2nd parm is dataframe to draw from, 3rd parm is seed for replicability
        rs = RandomState(int(h)+3000)
        bs = CircularBlockBootstrap(h*21, uv, random_state=rs)       
        
        # collect bootstrapped residuals to allow for parallel processing
        bs_uv = []
        for i, data in enumerate(bs.bootstrap(numSims)):
        
            # the bootstrapped dataframe is accessed via: data[0][0]
            bs_uv.append(data[0][0].reset_index())
                    
        # simulate the tests in parallel
        pool = mp.Pool(NUMBER_OF_CORES)            
        results = [pool.apply_async(sim_daily_nopredictability, args = (i, params, bs_uv[i], h)) for i in range(numSims)]
        pool.close()
        
        # collect the results
        results = [r.get() for r in results]
        results = pd.DataFrame(results, columns=['b','t(b)'])
        
        # save the results
        sim_res.loc[(b, h, 'daily')]['b']   = results.loc[:,'b']
        sim_res.loc[(b, h, 'daily')]['t(b)'] = results.loc[:,'t(b)']       
        
        
         # -------------- monthly model simulation -------------------- #
        
        # bootstrap pulls blocks of months
        w['mdate'] = pd.PeriodIndex(year=w.reset_index().date.dt.year, month=w.reset_index().date.dt.month, freq='M')
        uv['mdate']= pd.PeriodIndex(year=uv.reset_index().date.dt.year, month=uv.reset_index().date.dt.month, freq='M')
        uv['dt'] = uv.index
        
        # bootstrap generator; 1st parm is block size, 2nd parm is dataframe to draw from, 3rd parm is seed for replicability
        rs = RandomState(int(h)+30)
        bs = CircularBlockBootstrap(h, w.mdate, random_state=rs)  

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
                        
        # simulate the tests in parallel
        pool = mp.Pool(NUMBER_OF_CORES)            
        results = [pool.apply_async(sim_monthly_nopredictability, args = (i, params, bs_uv[i], bs_w[i], h)) for i in range(numSims)]
        pool.close()
        
        # collect the results
        results = [r.get() for r in results]
        results = pd.DataFrame(results, columns=['b','t(b)'])
 
        # save the results
        sim_res.loc[(b, h, 'monthly')]['b']   = results.loc[:,'b']
        sim_res.loc[(b, h, 'monthly')]['t(b)']= results.loc[:,'t(b)']
       
end = dt.datetime.now()
print("Total simulation time was {} seconds.".format((end-start).total_seconds()))

# save the output
sim_res.to_csv(OUTPUTFILE)
