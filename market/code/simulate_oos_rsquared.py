"""
This file performs simulations of out-of-sample R2s and p-values
It matches the new simulation algorithm 
described in appendix B and performs the simulation in all horizons.
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
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import wrds
import seaborn as sns
sns.set_style('whitegrid')

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

#%%============================================================================
# define directories
# =============================================================================

# paths
PATH = PATH_TO_REPLICATION_PACKAGE + 'market/'
CODE_DIR         = PATH + 'code/'
INPUT_DIR        = PATH + 'input/'
INTERMEDIATE_DIR = PATH + 'intermediate/'       
FIGURES_DIR      = PATH + 'figures/'      


# input data that is used in this script
DS_DAILY      = INTERMEDIATE_DIR + 'ds_mkt_daily.csv'

# output of this script
OUTPUTFILE    = INTERMEDIATE_DIR + 'sim_oos.csv'
OUTPUTFIG0    = FIGURES_DIR + 'sim_oosr2_by_ooshorizon_oosr2_martin.pdf'
OUTPUTFIG1    = FIGURES_DIR + 'sim_oosr2_by_ooshorizon_pvalue_martin.pdf'
OUTPUTFIG2    = FIGURES_DIR + 'sim_oosr2_by_ooshorizon_oosr2_cyl.pdf'
OUTPUTFIG3    = FIGURES_DIR + 'sim_oosr2_by_ooshorizon_pvalue_cyl.pdf'

#%%============================================================================
# Estimate parameters of predictive system
# =============================================================================

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
    
    # estimate daily covariance matrix of u and v
    Sig_uv = pd.concat((u,v),axis=1).cov()
    
    return [bbar, a, theta, var_b, Sig_uv]


#%%============================================================================
# OOS R2 functions
# =============================================================================

def diebold_mariano(d,outcome,fcst,benchmrk,lags) :
    ''' Test for H0: benchmrk sqd err <= fcst sqd err
        returns tstat, pvalue, and R-squared relative to the benchmark
        d = name of dataframe
        outcome = target variable
        fcst = forecast to be tested
        benchmrk = benchmark forecast
        lags = number of lags for Hansen-Hodrick std error
    '''
    df = d[[outcome,fcst,benchmrk]].dropna()
    u0 = df[outcome] - df[benchmrk]
    u1 = df[outcome] - df[fcst]
    
    ser = u0**2 - u1**2
    
    result = sm.OLS(ser,np.ones(len(ser))).fit(cov_type='HAC',cov_kwds={'maxlags':lags,'kernel':'uniform'})
    
    t = result.tvalues.item()
    p = result.pvalues.item()

    # If HH SEs are undefined:  
    if np.isnan(t):
        result = sm.OLS(ser,np.ones(len(ser))).fit(cov_type='HAC',cov_kwds={'maxlags':lags,'kernel':'bartlett'})
        t = result.tvalues.item()
        p = result.pvalues.item()
        
    p = p/2 if t>0 else 1-p/2
    R2 = 1 - (u1**2).sum() / (u0**2).sum()
    
    return R2, p


#%%============================================================================
# Define the simulation function
# =============================================================================

''' run simulation - output is daily r and daily b;
    simulation is run under assumption of a tight bound b (that is b=mu here)
    so to generate a slack return, add slackness to r
'''


# define the simulation function
# simulation function also runs the estimation at various horizons for OOS R2 and DM
def simulation(attempt, params, oos_horizon, h = 1, slackness=0):
    '''
    Parameters
    ----------
    attempt : integer; the simulation number (inputs into results matrix)
    params  : set of calibrated parameters for running the simulation
    oos_horizon : array of horizon lengths in years of OOS forecasting period
    h : integer of return horizon length (1,3,6, or 12)
    slackness : annualized, decimal size of slackness in the simulation

    Returns
    -------
    sim_res : pandas dataframe that logs the OOS R2 and DM p-value for each simulation

    '''
    
    # read-in the estimated parameters
    bbar = params[0]
    a = params[1]
    theta = params[2]
    var_b = params[3]
    Sig_uv = params[4]
    
    # Draw random innovations for u and w
    np.random.seed(attempt)
    
    # generate b_0
    b0 = bbar + np.sqrt(var_b)*norm.rvs()

    # day 0 is the last day of the month prior to observing daily returns
    # the forward return on day 0 is the return on days 1 through 21
    # we call days 1 through 21 'month 1'
    # generate u and v - we will not use u and v on day 0
    uv = pd.DataFrame(mvnorm(cov=Sig_uv).rvs(numDays+1),columns=['u','v'],index=range(numDays+1))
    uv['month'] = 1 + ((uv.index-1)/21).astype(int)
    uv.index.name = 'day'
    
    # generate (tight) bound
    b = pd.Series(dtype=float,index=range(numDays+1))
    b.index.name = 'day'
    b.iloc[0] = b0
    for day in range(1,numDays+1) :
        b.loc[day] = (1-a)*bbar + a*b.loc[day-1] + theta*uv.u.loc[day]
    
    # generate forward returns
    mu = bbar/(21*h) + (b-bbar)/theta
    z = mu.shift() + uv.v
    r = z.rolling(21*h).sum().shift(-21*h)
        
    # put all the simulated variables in a dataset
    sim_ds = pd.DataFrame()
    # sim_ds[gw.columns] = gw 
    bnd = 'bound_'+str(h)
    fret  = 'f_mktrf'+str(h)    
    
    sim_ds[fret] = r
    sim_ds[bnd]  = b - slackness*h/12
     
    
    ##### Analysis of simulated data

    #Convert data to monthly frequency
    sim_ds['month'] = 1 + ((sim_ds.index-1)/21).astype(int)
    sim_ds = sim_ds.groupby('month').last()
    
    #Calculate expanding window historical average return over the first 65 years of the sample
    # (lagging h months to eliminate look-ahead bias in forward returns)
    # mu_histavg = sim_ds['f_mktrf'+str(h)].loc[:(numYrs_prebound*12)].mean()
    # sim_ds['benchmark'] = mu_histavg*h
    sim_ds['benchmark'] = sim_ds[fret].expanding(min_periods=numYrs_prebound*12).mean().shift(h)

    #Initialize dataset to store results
    stats = ['oos_r2', 'pvalue', 'mu_bar_hat', 'insample_slackness']     
    fcsts = ['fcst_tight', 'fcst_slack']      
    idx = pd.MultiIndex.from_product([[slackness],[attempt],fcsts, oos_horizon], names = ['slackness','sim_num','fcst','oos_horizon'])                       
    sim_res = pd.DataFrame(dtype=float, index=idx, columns=stats)  
    
    
    # For years (66,66+oos_horizon), calculate OOS R-squared and DM p-value for 2 forecasts:
    #   (0) tight bound
    #   (1) bound + avg slackness
    
    oos_data = sim_ds.loc[(numYrs_prebound*12):]
       
    # for oos_h in [int(i) for i in oos_horizon]:
    for oos_h in oos_horizon:
        test_data = oos_data[:(oos_h*12)].dropna()

        # Calculate the two forecasts (skipping estimation window and h months to eliminate look-ahead bias in average slackness)
        test_data['slack'] = (test_data[fret] - test_data[bnd]).expanding(min_periods=min_window).mean().shift(h)
        test_data['fcst_slack'] = test_data.slack + test_data[bnd]
        test_data['fcst_tight'] = np.where(~np.isnan(test_data.fcst_slack), 0.0, np.nan )  + test_data[bnd]     #tight bound -> slackness of zero        

        # Run OOS R2 and DM for each forecast
        for f in fcsts:
            R2,p = diebold_mariano(test_data,fret,f,'benchmark',h)
            sim_res.loc[(slackness,attempt,f,oos_h),'oos_r2']=R2
            sim_res.loc[(slackness,attempt,f,oos_h),'pvalue'] =p
            sim_res.loc[(slackness,attempt,f,oos_h),'mu_bar_hat'] = test_data['benchmark'].mean()
            sim_res.loc[(slackness,attempt,f,oos_h),'insample_slackness'] = test_data['slack'].mean()
                
        
    return sim_res 
# test = simulation(1,params,oos_horizon, horizon,slackness=0.1)



#%%============================================================================
# Simulate the predictive system
# =============================================================================

# set hyperparameters of the simulation
oos_horizon = np.array([30,50,75,100,150,200, 250, 500])
numYrs_prebound = 65
numYrs_withbound= max(oos_horizon)
numYears = numYrs_prebound + numYrs_withbound
numMonths = 12*numYears
numDays = 21*numMonths
numSims = 1000
min_window = 60




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
    
# generate a dataframe for storing the results
sims = list(range(0, numSims))
bounds = ['lb_m','lb_cylr']
# horizons = ['1','3','6','12']
horizons = ['12']
stats = ['oos_r2', 'pvalue', 'mu_bar_hat', 'insample_slackness']      
fcsts = ['fcst_tight', 'fcst_slack']   
# slackness = list(np.linspace(0.00,0.05,num=2).round(2))   
slackness = [0.0, 0.04, 0.05]                          
idx = pd.MultiIndex.from_product([bounds, horizons,slackness,sims,fcsts,oos_horizon], names = ['bound','horizon','slackness','sim_num','fcst','oos_horizon'])
sim_res = pd.DataFrame(dtype=float, index=idx, columns=stats)






import datetime as dt
start = dt.datetime.now()

# run the simulation
for bound in ['lb_m', 'lb_cylr']:
#     for horizon in [1,3,6,12]:
    for horizon in [12]:
        
        print('Simulating for ' + bound + '_' + str(horizon) + ' ...')
    
        # estimate the parameters
        params = estimate(ds,ret = 'f_mktrf' + str(horizon), bound = bound + '_' + str(horizon), h= horizon)
    
        for slack in slackness:
        
            # simulate the tests in parallel.
            pool = mp.Pool(NUMBER_OF_CORES)            
            results = [pool.apply_async(simulation, args = (i, params, oos_horizon, horizon, slack)) for i in range(numSims)]
            pool.close()
        
    
            # collect the results        
            results = [r.get() for r in results]
            all_results = pd.concat(results)
            sim_res.loc[(bound,str(horizon))].loc[all_results.index]=all_results
       
end = dt.datetime.now()
print("Total simulation time was {} seconds.".format((end-start).total_seconds()))
       
#Save to disk      
sim_res[['oos_r2','pvalue']].to_csv(OUTPUTFILE)        




#%%============================================================================
# Summarize the simulated data - OOS R2 - Martin
# =============================================================================

# Subset data for bound-horizon of interest
bound  ='lb_m'
horizon=12
d       = sim_res.loc[(bound,str(horizon),slice(None),slice(None),slice(None))]
# Subset simulations with data-generating process of tight bound
d_tight = sim_res.loc[(bound,str(horizon),0.0,slice(None),slice(None))]


#Define formatting for plots
flierprops    = dict(marker='.', markerfacecolor='black', markersize=4,
                  markeredgecolor='none')
meanlineprops = dict(linestyle='--', linewidth=1, color='red')
medianprops   = dict(linestyle='-', linewidth=1, color='black')




##### Plot of distributions of simulated OOS R2 as function of OOS horizon
fig, axes = plt.subplots(3,1,figsize=(8,8),sharex=True,sharey=False)

# Top panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_tight'),'oos_r2']
df = df.unstack(level='oos_horizon')
ax=axes[0]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(a) Forecast=bound; DGP=tight bound',fontsize=14)
ax.set_ylabel('OOS $R^2$',fontsize=14)

# Middle panel
slack = 0.05
df = d.loc[(bound,str(horizon),slack,slice(None),'fcst_tight',slice(None)),'oos_r2']
df = df.unstack(level='oos_horizon')
ax=axes[1]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(b) Forecast=bound; DGP=slack bound',fontsize=14)
ax.set_ylabel('OOS $R^2$',fontsize=14)

# Bottom panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_slack'),'oos_r2']
df = df.unstack(level='oos_horizon')
ax=axes[2]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(c) Forecast=bound + average slackness',fontsize=14)
ax.set_xlabel('Sample length (years)',fontsize=14)
ax.set_ylim([-0.25,0.15])
ax.set_ylabel('OOS $R^2$',fontsize=14)


fig.tight_layout()
fig.savefig(OUTPUTFIG0,bbox_inches='tight')

df.describe()

#%%============================================================================
# Summarize the simulated data - pvalues - Martin
# =============================================================================

# Subset data for bound-horizon of interest
bound  ='lb_m'
horizon=12
d       = sim_res.loc[(bound,str(horizon),slice(None),slice(None),slice(None))]
# Subset simulations with data-generating process of tight bound
d_tight = sim_res.loc[(bound,str(horizon),0.0,slice(None),slice(None))]



##### Plot of distributions of simulated OOS R2 as function of OOS horizon
fig, axes = plt.subplots(3,1,figsize=(8,8),sharex=True,sharey=True)

# Top panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_tight'),'pvalue']
df = df.unstack(level='oos_horizon')
ax=axes[0]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(a) Forecast=bound; DGP=tight bound',fontsize=14)
ax.set_ylabel('$p$-value',fontsize=14)

# Middle panel
slack = 0.05
df = d.loc[(bound,str(horizon),slack,slice(None),'fcst_tight',slice(None)),'pvalue']
df = df.unstack(level='oos_horizon')
ax=axes[1]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(b) Forecast=bound; DGP=slack bound',fontsize=14)
ax.set_ylabel('$p$-value',fontsize=14)

# Bottom panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_slack'),'pvalue']
df = df.unstack(level='oos_horizon')
ax=axes[2]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(c) Forecast=bound + average slackness',fontsize=14)
ax.set_xlabel('Sample length (years)',fontsize=14)
ax.set_ylabel('$p$-value',fontsize=14)


fig.tight_layout()
fig.savefig(OUTPUTFIG1,bbox_inches='tight')


#%%============================================================================
# Summarize the simulated data - OOS R2 - CYL
# =============================================================================

# Subset data for bound-horizon of interest
bound  ='lb_cylr'
horizon=12
d       = sim_res.loc[(bound,str(horizon),slice(None),slice(None),slice(None))]
# Subset simulations with data-generating process of tight bound
d_tight = sim_res.loc[(bound,str(horizon),0.0,slice(None),slice(None))]


##### Plot of distributions of simulated OOS R2 as function of OOS horizon
fig, axes = plt.subplots(3,1,figsize=(8,8),sharex=True,sharey=False)

# Top panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_tight'),'oos_r2']
df = df.unstack(level='oos_horizon')
ax=axes[0]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(a) Forecast=bound; DGP=tight bound',fontsize=14)
ax.set_ylabel('OOS $R^2$',fontsize=14)

# Middle panel
slack = 0.04
df = d.loc[(bound,str(horizon),slack,slice(None),'fcst_tight',slice(None)),'oos_r2']
df = df.unstack(level='oos_horizon')
ax=axes[1]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(b) Forecast=bound; DGP=slack bound',fontsize=14)
ax.set_ylim([-0.4,0.3])
ax.set_ylabel('OOS $R^2$',fontsize=14)

# Bottom panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_slack'),'oos_r2']
df = df.unstack(level='oos_horizon')
ax=axes[2]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(c) Forecast=bound + average slackness',fontsize=14)
ax.set_xlabel('Sample length (years)',fontsize=14)
ax.set_ylim([-0.25,0.25])
ax.set_ylabel('OOS $R^2$',fontsize=14)


fig.tight_layout()
fig.savefig(OUTPUTFIG2,bbox_inches='tight')

df.describe()

#%%============================================================================
# Summarize the simulated data - pvalues - CYL
# =============================================================================

# Subset data for bound-horizon of interest
bound  ='lb_cylr'
horizon=12
d       = sim_res.loc[(bound,str(horizon),slice(None),slice(None),slice(None))]
# Subset simulations with data-generating process of tight bound
d_tight = sim_res.loc[(bound,str(horizon),0.0,slice(None),slice(None))]



##### Plot of distributions of simulated OOS R2 as function of OOS horizon
fig, axes = plt.subplots(3,1,figsize=(8,8),sharex=True,sharey=True)

# Top panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_tight'),'pvalue']
df = df.unstack(level='oos_horizon')
ax=axes[0]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(a) Forecast=bound; DGP=tight bound',fontsize=14)
ax.set_ylabel('$p$-value',fontsize=14)

# Middle panel
slack = 0.04
df = d.loc[(bound,str(horizon),slack,slice(None),'fcst_tight',slice(None)),'pvalue']
df = df.unstack(level='oos_horizon')
ax=axes[1]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(b) Forecast=bound; DGP=slack bound',fontsize=14)
ax.set_ylabel('$p$-value',fontsize=14)

# Bottom panel
df = d_tight.loc[(bound,str(horizon),0.0,slice(None),'fcst_slack'),'pvalue']
df = df.unstack(level='oos_horizon')
ax=axes[2]
ax.boxplot(df,labels=df.columns,showmeans=True,meanline=True,meanprops=meanlineprops, medianprops=medianprops, flierprops=flierprops)
ax.set_title('(c) Forecast=bound + average slackness',fontsize=14)
ax.set_xlabel('Sample length (years)',fontsize=14)
ax.set_ylabel('$p$-value',fontsize=14)


fig.tight_layout()
fig.savefig(OUTPUTFIG3,bbox_inches='tight')