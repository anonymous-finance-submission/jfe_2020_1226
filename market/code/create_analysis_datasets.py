"""
This code puts together data from several files to create a dataset ready for analysis.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import wrds
import seaborn as sns
sns.set()
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


# paths
PATH = PATH_TO_REPLICATION_PACKAGE + 'market/'
CODE_DIR         = PATH + 'code/'
INPUT_DIR        = PATH + 'input/'
INTERMEDIATE_DIR = PATH + 'intermediate/'   
TABLES_DIR       = PATH + 'tables/'         
FIGURES_DIR      = PATH + 'figures/'        

# input data that is used in this script
FACTSET_DATA = INPUT_DIR + 'sp500rets_1990_2021_factset.csv'
BOUNDS_DATA  = INTERMEDIATE_DIR + 'market_bounds.csv'
GW_DATA      = INPUT_DIR + 'PredictorData2020.xlsx'

# output of the script 
DS_DAILY   = INTERMEDIATE_DIR + 'ds_mkt_daily.csv'          
DS_MONTHLY = INTERMEDIATE_DIR + 'ds_mkt_monthly.csv'        


#%% ===========================================================================
# Combine returns, bounds, and conditioning variables
# =============================================================================
# read S&P 500 returns from CRSP
conn=wrds.Connection()
sprets_wrds = conn.raw_sql("""
                    select caldt, vwretd
                    from crsp.dsp500p
                    where caldt >= '01/01/1990'
                    """)
sprets_wrds['caldt'] = pd.to_datetime(sprets_wrds['caldt']) 

# add 2021 returns for S&P 500 returns from Factset - Used for 2021 returns
sprets_factset = pd.read_csv(FACTSET_DATA)
sprets_factset['caldt']= pd.to_datetime(sprets_factset[['year','month','day']])
sprets_factset = sprets_factset[['caldt','totalreturnnet']]
sprets_factset = sprets_factset.rename(columns={'totalreturnnet':'vwretd'})

# use WRDS data prior to 2021
maxdate = sprets_wrds.caldt.max()
sprets_factset = sprets_factset[sprets_factset.caldt>maxdate]
sprets = pd.concat((sprets_wrds,sprets_factset))
sprets.sort_values(by='caldt',inplace=True)
sprets.set_index('caldt',inplace=True)
sprets.index.name = 'date'

# add FF risk-free rate and close the WRDS connection
ff_daily_wrds = conn.raw_sql("""
                    select date, rf
                    from ff.factors_daily
                    where date >= '01/01/1990'
                    """) 
conn.close()
ff_daily_wrds['date'] = pd.to_datetime(ff_daily_wrds.date)
ff_daily_wrds = ff_daily_wrds.set_index('date')  

sprets = sprets.merge(ff_daily_wrds,how='left',left_index=True,right_index=True)
sprets['mktrf'] = sprets['vwretd'] - sprets['rf']

# calculate forward returns for each horizon
# the forward return at each date starts at the end of that day (it does not include that day's return)
sprets['cumprod_gross'] = (1 + sprets.mktrf).cumprod()
sprets['f_mktrf1'] = sprets.cumprod_gross.shift(-21) / sprets.cumprod_gross - 1
sprets['f_mktrf3'] = sprets.cumprod_gross.shift(-63) / sprets.cumprod_gross - 1
sprets['f_mktrf6'] = sprets.cumprod_gross.shift(-126)/ sprets.cumprod_gross - 1
sprets['f_mktrf12']= sprets.cumprod_gross.shift(-252)/ sprets.cumprod_gross - 1
cols = ['f_mktrf'+repr(i) for i in [1,3,6,12]]
sprets = sprets[cols]


# add bounds
# based on end-of-day option prices, so they are bounds for the forward returns calculated above
# Use OptionMetrics data for 1996-2019 and CBOE bounds for 1990-1995 and 2020
bounds = pd.read_csv(BOUNDS_DATA)
bounds['date'] = pd.to_datetime(bounds['date'])
bounds = bounds.set_index('date') 
bounds.sort_index(inplace=True)
# bounds.drop(columns=['close','rf'],inplace=True)
ds = sprets.merge(bounds,how='outer',left_index=True,right_index=True)

# annualize the forward realized returns (bounds are already annualized in bounds dataset) and express as percent
ds['f_mktrf1'] = ds['f_mktrf1'] *12 *100
ds['f_mktrf3'] = ds['f_mktrf3'] * 4 *100
ds['f_mktrf6'] = ds['f_mktrf6'] * 2 *100
ds['f_mktrf12']= ds['f_mktrf12']* 1 *100


# add Goyal-Welch conditioning variables and define positive versions
gw = pd.read_excel(GW_DATA, sheet_name='Monthly',index_col='yyyymm')
gw['dp']                = 5 + np.log(gw.D12) - np.log(gw['Index'])
gw['ep']                = 5 + np.log(gw.E12) - np.log(gw['Index'])
gw['bm']                = gw['b/m']
gw['dfy']               = gw.BAA - gw.AAA
gw['ntis_plus1']        = 1 + gw.ntis
gw['lty_spread_plus1']  = 1 + gw.lty - gw.tbl 
gw['infl_plus1']        = 1 + gw.infl

gw['year']  = np.floor(gw.index/100)
gw['month'] = np.mod(gw.index,100)
gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']
cols = ['year','month']+ gw_vars
gw = gw[cols]



# create a daily dataset with forward returns, bounds, and GW conditioning variables, 
# We match each month's GW variables to the last day of the month for the forward returns and bounds.
# The forward returns and bounds start at the end of the day, and we assume the GW variables
# for the month are known at the end of the last day of the month.
eom = ds.groupby([ds.index.year, ds.index.month]).tail(1).copy()
eom['date']=eom.index
eom = eom[['date']]
gw = pd.merge(gw, eom, how='inner', left_on=['year','month'], right_on=[eom.index.year, eom.index.month])
gw.set_index('date', inplace=True)
ds = ds.merge(gw,how='outer',left_index=True,right_index=True)
ds.sort_index(inplace=True)
for i in gw_vars:
    ds[i] = ds[i].fillna(method='ffill')
    ds[i].loc['2021'] = np.nan
ds.drop(columns=['year','month'], inplace=True)

#%% ===========================================================================
# Save analysis datafiles
# =============================================================================

# save daily data
ds = ds.loc['1990':'2020']              # Conditioning variables end in Dec 2020
ds.to_csv(DS_DAILY)


# create a monthly dataset with standardized GW variables, forward returns, and bounds
# monthly dataset contains GW variables (adjusted to be positive) and standardized GW variables (suffix= _sd)
ds_mthly = ds.resample('M').last()

# save monthly dataset
ds_mthly.to_csv(DS_MONTHLY)
