"""
This file calculates stock level bounds.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import datetime as dt
import wrds
from time import sleep
import os
import shutil

path                            = PATH_TO_REPLICATION_PACKAGE + 'stock/'
project_dir                     = path + 'code/'
input_dir                       = path + 'input/'
results_dir                     = path + 'intermediate/'
stock_data_dir                  = results_dir + 'stock data'

# input files
crsp_om_link_clean              = results_dir + 'crsp_om_link_clean.csv'
rf_famafrench                   = input_dir + 'rf_famafrench.csv'

# output files
# make a directory to store intermediate results
if not os.path.exists(results_dir + 'stock data'):
    os.makedirs(results_dir + 'stock data')
    
rfsvix_stock_year_batch         = results_dir + 'stock data/{}_{}.pkl'
rfsvix_mkt_year                 = results_dir + 'stock data/sp_{}.pkl'
stock_bounds                    = results_dir + '/stock_bounds.csv'

os.chdir(project_dir)

#%%============================================================================
# function definitions
# =============================================================================
# Kadan/Tang method of integration
def integrate(x,y):
    # convert all data formats to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # drop missing values
    I = (~np.isnan(x)) & (~np.isnan(y))
    x = x[I]
    y = y[I]
    
    # let's get the minimum of two sides in every rectangle
    dx = x[1:] - x[:-1]
    min_y = np.minimum(y[:-1], y[1:])
    
    return sum(min_y*dx)

# This function calculates the lower bounds in Martin and Wagner (2018).
def MartinWagner(db, idlist, year):
      
      # raw_sql only accepts tuples
      parm = {'ids':tuple(idlist)}

      # downloading option data and data on underlying securities
      options = db.raw_sql('select * from optionm.vsurfd' + str(year) + ' where secid in %(ids)s', params = parm)
      securities = db.raw_sql('select * from optionm.secprd' + str(year) + ' where secid in %(ids)s', params = parm)
      
      # dropping unnecessary columns and renaming
      options = options[['secid','date','days','impl_strike','impl_premium','cp_flag']]
      options = options.rename(columns = {'days':'ttm','impl_strike':'strike','impl_premium':'prc'})
      
      # I tag the missing values first, and then drop every (secid, date, ttm)
      # where there are missing data.
      options = options.replace([-99.99,-99,-999, 0], np.nan)
      options['tag'] = options['ttm'].isna()*1.0 + options['strike'].isna()*1.0 
      grouped = options.groupby(['secid','date','ttm'])['tag'].sum().reset_index()
      options = options.merge(grouped, on = ['secid','date','ttm'], how = 'left')
      options = options.loc[options['tag_y'] == 0, ['secid','date','ttm','strike','prc','cp_flag']]
      
      # just sorting the data 
      options = options.sort_values(by = ['secid','date']).reset_index(drop = True)
      securities = securities.sort_values(by = ['secid','date']).reset_index(drop = True)

      # merging the option data with underlying security data
      merged = options.merge(securities[['secid','date','close']], on = ['secid','date'], how = 'left')
      merged = merged.sort_values(by = ['secid', 'date', 'ttm', 'strike']).reset_index(drop = True)
      
      # strike prices for call and put options do not match in volatility surf-
      # ace data. As a result, I had to change the algorithm for calculating t-
      # he integrals. As a consequence, dropping data points that violate mono-
      # tonicity has been merged into the integration step.
      I = (merged['cp_flag'] == 'C')
      
      merged['call_prc'] = np.nan
      merged.loc[I,'call_prc'] = merged.loc[I,'prc']
      merged['call_prc'] = merged['call_prc'].fillna(1000000)
      merged['running_min'] = merged.groupby(['secid','date','ttm'])['call_prc'].cummin(skipna = False)
      
      merged['put_prc'] = np.nan
      merged.loc[~I,'put_prc'] = merged.loc[~I,'prc']
      
      merged['put_prc'] = merged['put_prc'].fillna(-1000000)
      merged['running_max'] = merged.groupby(['secid','date','ttm'])['put_prc'].cummax(skipna = False)
      
      # Here I drop the data points not used in the integration process. For p-
      # ut options, running_max must be strictly less than running_min. For ca-
      # ll options, running_max must be greater than or equal to running_min.
      merged = merged[(~I & (merged['running_max'] < merged['running_min']))|(I & (merged['running_max'] >= merged['running_min']))]
      merged = merged.drop(columns = ['put_prc','call_prc','running_min','running_max'])
      
      # calculating the integrals    
      grouped = merged.groupby(['secid','date','ttm'])
        
      stats = pd.DataFrame()
      stats = grouped['close'].last().reset_index()
      stats['sum_val'] = grouped.apply(lambda g: integrate(x = g.strike, y = g.prc)).reset_index(drop = True)

      # calculate the svix without the rf.
      stats['svix'] = (2.0*stats.sum_val)/((stats.ttm/365)*(stats.close**2))
      stats = stats.pivot_table(values = 'svix', index = ['secid','date'], columns = 'ttm').reset_index(drop = False)
      stats = stats.drop(columns = [60,122,152,273,547,730])
      stats = stats.rename(columns = {30:'lb_kt_1',91:'lb_kt_3',182:'lb_kt_6',365:'lb_kt_12'})
      
      stats = stats[['secid','date','lb_kt_1','lb_kt_3','lb_kt_6','lb_kt_12']]
      return stats

# this function operates in a similar way to the rangejoin command in stata; 
# it picks observations in df1 whose date are between startdate and enddate 
# in df2.
def rangejoin(df1, df2, keys, date, startdate, enddate, left = True):
    keys = [keys] if isinstance(keys,str) else keys
    sort_keys = keys + [date]
    df1.sort_values(by = sort_keys, inplace = True)
    df_merge = pd.merge(df1, df2, on = keys)
    df_merge = df_merge.query(date + ' >= ' + startdate + ' and ' + date + ' <= ' + enddate)
    
    if left == False:
        return df_merge
    else:
        keys.append(date)
        df = df1.merge(df_merge, on=keys, how='left', suffixes = ('','_yy_'), indicator = True)
        df = df[df['_merge'] == 'left_only']
        columns_to_drop = [i for i in df.columns if i.endswith('_yy_')]
        columns_to_drop.append('_merge')
        df.drop(columns = columns_to_drop, inplace=True)
        return pd.concat([df_merge,df], axis =0).sort_values(by = sort_keys)

#%%============================================================================
# calculating the Martin/Wagner SVIX*Rf
# =============================================================================
# read the list of securities from an pkl file. This file is produced by 
# omsp_generator.py
sp500 = pd.read_csv(crsp_om_link_clean, parse_dates = ['sp_beg','sp_end','lnk_beg','lnk_end'])
sp500['sp_beg'] = sp500['sp_beg'].apply(lambda x: dt.date(x.year, x.month, x.day))
sp500['sp_end'] = sp500['sp_end'].apply(lambda x: dt.date(x.year, x.month, x.day))
sp500['lnk_beg'] = sp500['lnk_beg'].apply(lambda x: dt.date(x.year, x.month, x.day))
sp500['lnk_end'] = sp500['lnk_end'].apply(lambda x: dt.date(x.year, x.month, x.day))
idlist = sp500['secid'].unique()

# WRDS has put in a restriction that stops large memory usage. I divide the job
# into different batches so that each batch takes less space in RAM.
num_batches = 20
last_year = 2020
num_ids = len(idlist)
index = np.floor(np.linspace(0, num_ids, num = num_batches + 1)).astype(int)
batches = []
for b in range(1, num_batches + 1):
    batches.append(idlist[index[b-1]:index[b]])
    

print('\n\n---------------------------------------------')
print(' calculating svix for individual securities')
print('---------------------------------------------\n\n')

db = wrds.Connection()
for i in range(1996,last_year + 1):  
      print('Now processing ' + str(i) + ':')
      for b in range(0,num_batches):
          batch = batches[b]
          print('    batch number ' + str(b))
          # server disconnects are possible - try 3 times 
          for x in range(1, 4):
                try:
                      print('        Attempt ' + str(x))
                      mw_append = MartinWagner(db,batch,i)
                      break
                except Exception:
                      # wait for 2 seconds if error occurs
                      print('              Exception!')
                      sleep(2)
        
          mw_append.to_pickle(rfsvix_stock_year_batch.format(i,b))

print('\n\n---------------------------------------------')
print(' calculating svix for s&p index')
print('---------------------------------------------\n\n')

idlist = [108105]

sp = pd.DataFrame()
for i in range(1996,last_year + 1):  
      print('Now processing S&P for ' + str(i) + ':')
      # server disconnects are possible - try 3 times 
      for x in range(1, 4):
            try:
                  print('    Attempt ' + str(x))
                  sp_append = MartinWagner(db, idlist, i)
                  break
            except Exception:
                  # wait for 2 seconds if error occurs
                  print('          Exception!')
                  sleep(2)

      sp_append.to_pickle(rfsvix_mkt_year.format(i))
db.close()

#%%============================================================================
# bound calculations finished. aggregating all the files
# =============================================================================
# read all the files from current directory and append them to get one dataset
data = pd.DataFrame()

for i in range(1996,last_year + 1):
    for b in range(0, num_batches):
        # read one (year, batch) of data
        data_append = pd.read_pickle(rfsvix_stock_year_batch.format(i,b))
        data = data.append(data_append, ignore_index = True)

# keep only the dates when the secid is in the index
sp500['start'] = sp500[['sp_beg', 'lnk_beg']].max(axis =1)
sp500['end'] = sp500[['sp_end', 'lnk_end']].min(axis = 1)
cols = ['permno','secid','start','end']
data = rangejoin(data, sp500[cols],'secid','date','start','end', left = False).drop(columns = ['start','end'])    

# drop the dates where all observations were zero - probably we did not have 
# data for those days
I = ~((data['lb_kt_1'] == 0) & (data['lb_kt_3'] == 0) & (data['lb_kt_6'] == 0) & (data['lb_kt_12'] == 0))
data = data.loc[I,:]

# generate a month variable - this is needed for building a monthly dataset
data['month'] = data['date'].apply(lambda x: x.year*100 + x.month)

# get the list of permnos to match with CRSP returns
permno_list = data['permno'].unique()

# merge with s&p SVIX
data_sp = pd.DataFrame()
for i in range(1996,last_year + 1):
    data_append = pd.read_pickle(rfsvix_mkt_year.format(i))
    data_sp = data_sp.append(data_append,ignore_index = True)
    
data_sp = data_sp.sort_values('date').reset_index(drop = True)

data_sp = data_sp.rename(columns = {'lb_kt_1':'Rf_x_svix1_sp','lb_kt_3':'Rf_x_svix3_sp',
                                    'lb_kt_6':'Rf_x_svix6_sp','lb_kt_12':'Rf_x_svix12_sp'})

data = data.merge(data_sp[['date','Rf_x_svix1_sp','Rf_x_svix3_sp',
                           'Rf_x_svix6_sp','Rf_x_svix12_sp']], on = 'date', how = 'left')

#%%============================================================================
# merging with crsp daily files and generating other variables
# =============================================================================

# get the CRSP daily data
parm = {'ids':tuple(permno_list)}
db = wrds.Connection()
dsf = db.raw_sql('select permno, permco, date, prc, ret, shrout from crsp.dsf where permno in %(ids)s and date >= \'1994-1-1\'', params = parm)
vwretd = db.raw_sql('select * from crsp.dsi where date >= \'1994-1-1\'')
db.close()

# merge daily stock files with market data
vwretd = vwretd.set_index('date')
market = pd.DataFrame(vwretd['vwretd'], vwretd.index)
market.reset_index(inplace = True)
dsf = dsf.merge(market,on = 'date', how = 'left')

# calculate deltas as in Kadan and Tang (2019)
dsf['cov_mkt'] = 0
dsf['var'] = 0
for i in dsf.permno.unique():
    I = dsf['permno'] == i
    temp = dsf.loc[I,:]
    cov_mkt = np.zeros((temp.shape[0],))
    cov_sp = np.zeros((temp.shape[0],))
    var = np.zeros((temp.shape[0],))
    for j, d in enumerate(temp['date']):
        
        if (d.month == 2) and (d.day == 29):
            min_date = dt.date(d.year - 1, d.month, 28)
        else:
            min_date = dt.date(d.year - 1, d.month, d.day)
            
        prev_year = temp.loc[(temp['date'] >= min_date) & (temp['date'] <= d)]
        prev_year = prev_year[~prev_year['ret'].isna()]
        
        if prev_year.shape[0] <= 200:
            cov_mkt[j] = np.nan
            var[j] = np.nan
        else:
            cov_mkt[j] = prev_year['ret'].cov(prev_year['vwretd'])
            var[j] = prev_year['ret'].cov(prev_year['ret'])
    
    dsf.loc[I,'cov_mkt'] = cov_mkt
    dsf.loc[I,'var'] = var

dsf['delta'] = dsf['var']/dsf['cov_mkt']

dsf.sort_values(by = ['permno','date'], inplace = True)

# generate market cap
dsf = dsf.sort_values(by = ['permno', 'date']).reset_index(drop = True)
dsf['cap'] = abs(dsf['prc'])*dsf['shrout']
dsf = dsf.drop(columns = ['prc','shrout','permco','ret'])
dsf = dsf.dropna(subset = ['cap'])

# merge with the daily bounds
data = data.merge(dsf, on = ['permno','date'], how = 'left')

#%%============================================================================
# generate MW bounds
# =============================================================================

# generate value-weighted average of SVIX's
data['Rf_x_svix1_cap'] = data['lb_kt_1']*data['cap']
data['Rf_x_svix3_cap'] = data['lb_kt_3']*data['cap']
data['Rf_x_svix6_cap'] = data['lb_kt_6']*data['cap']
data['Rf_x_svix12_cap'] = data['lb_kt_12']*data['cap']

grouped = data.groupby('date')

aux = pd.DataFrame()
aux['Rf_x_svix1_bar'] = grouped['Rf_x_svix1_cap'].sum() / grouped['cap'].sum()
aux['Rf_x_svix3_bar'] = grouped['Rf_x_svix3_cap'].sum() / grouped['cap'].sum()
aux['Rf_x_svix6_bar'] = grouped['Rf_x_svix6_cap'].sum() / grouped['cap'].sum()
aux['Rf_x_svix12_bar'] = grouped['Rf_x_svix12_cap'].sum() / grouped['cap'].sum()
aux = aux.reset_index()

data = data.merge(aux, on = 'date', how = 'left')

# generate Martin/Wagner bounds
for i in ['1','3','6','12']:
    data['lb_mw_' + i] = data['Rf_x_svix' + i + '_sp'] + 0.5*(data['lb_kt_' + i] - data['Rf_x_svix' + i + '_bar'])

cols_to_keep = ['secid', 'permno', 'month', 'date', 'lb_kt_1', 
                'lb_kt_3', 'lb_kt_6', 'lb_kt_12', 'lb_mw_1', 'lb_mw_3',
                'lb_mw_6', 'lb_mw_12', 'Rf_x_svix1_sp', 'Rf_x_svix3_sp',
                'Rf_x_svix6_sp', 'Rf_x_svix12_sp', 'Rf_x_svix1_bar', 
                'Rf_x_svix3_bar', 'Rf_x_svix6_bar', 'Rf_x_svix12_bar', 
                'delta']

data = data[cols_to_keep]

bound_cols = ['lb_kt_1','lb_kt_3','lb_kt_6','lb_kt_12','Rf_x_svix1_sp',
        'Rf_x_svix3_sp','Rf_x_svix6_sp','Rf_x_svix12_sp','Rf_x_svix1_bar',
        'Rf_x_svix3_bar','Rf_x_svix6_bar','Rf_x_svix12_bar', 'lb_mw_1', 
        'lb_mw_3', 'lb_mw_6','lb_mw_12']

data[bound_cols] *= 100

#%%============================================================================
# generate monthly datasets
# =============================================================================

# keep EOM observations
data_m = data.groupby(['permno','month']).last().reset_index()
data_m['last_trading_day'] = data_m.groupby('month')['date'].transform('max')
data_m = data_m[data_m['date'] == data_m['last_trading_day']]
data_m = data_m.drop(columns = ['date', 'last_trading_day']).rename(columns = {'month':'date'})

# now get the CRSP monthly data
parm = {'ids':tuple(permno_list)}
db = wrds.Connection()
msf = db.raw_sql('select permno, date, ret from crsp.msf where permno in %(ids)s and date >= \'1994-1-1\'', params = parm)
index = db.raw_sql('select caldt, vwretd from crsp.msp500p where caldt >= \'1995-12-1\'')
db.close()

msf['date'] = msf['date'].apply(lambda x: x.year*100 + x.month)

# generate market cap for future purposes
msf = msf.dropna(subset = ['ret'])
msf['ret'] = 1.0 + msf['ret']

# download and merge risk-free returns with each data point
# when using the wrds server, we should save the rf on the server
# rf = port.DataReader('F-F_Research_Data_Factors','famafrench',start = dt.date(1930,1,1), end = dt.date(last_year + 1,12,31))[0]
rf = pd.read_csv(rf_famafrench).set_index('Date')
rf = rf['RF'].rename('rf')
rf = 1.0 + rf/100
rf = rf.reset_index().rename(columns = {'Date':'date'})
rf['date'] = rf.date.apply(lambda x: int(x.split('-')[0])*100 + int(x.split('-')[1]))

msf = msf.merge(rf,on = 'date', how = 'left')

# now fix s&p500 returns and add to msf
index['date'] = index['caldt'].apply(lambda x: x.year*100 + x.month)
index = index[['date','vwretd']].rename(columns = {'vwretd':'sp5ret'}).reset_index(drop = True)
index['sp5ret'] = 1.0 + index['sp5ret']

msf = msf.merge(index, on = 'date', how = 'left')

# lag returns to create ret1, ret3,ret6,ret12,ret24 and the corresponding rf's
grouped = msf.groupby('permno')

msf['ret1'] = grouped['ret'].shift(-1)
msf['rf1'] = grouped['rf'].shift(-1)
msf['sp5ret1'] = grouped['sp5ret'].shift(-1)

msf['ret3'] = grouped['ret'].rolling(window = 3).agg(lambda x: x.prod()).groupby('permno').shift(-3).reset_index(drop = True)
msf['rf3'] = grouped['rf'].rolling(window = 3).agg(lambda x: x.prod()).groupby('permno').shift(-3).reset_index(drop = True)
msf['sp5ret3'] = grouped['sp5ret'].rolling(window = 3).agg(lambda x: x.prod()).groupby('permno').shift(-3).reset_index(drop = True)

msf['ret6'] = grouped['ret'].rolling(window = 6).agg(lambda x: x.prod()).groupby('permno').shift(-6).reset_index(drop = True)
msf['rf6'] = grouped['rf'].rolling(window = 6).agg(lambda x: x.prod()).groupby('permno').shift(-6).reset_index(drop = True)
msf['sp5ret6'] = grouped['sp5ret'].rolling(window = 6).agg(lambda x: x.prod()).groupby('permno').shift(-6).reset_index(drop = True)

msf['ret12'] = grouped['ret'].rolling(window = 12).agg(lambda x: x.prod()).groupby('permno').shift(-12).reset_index(drop = True)
msf['rf12'] = grouped['rf'].rolling(window = 12).agg(lambda x: x.prod()).groupby('permno').shift(-12).reset_index(drop = True)
msf['sp5ret12'] = grouped['sp5ret'].rolling(window = 12).agg(lambda x: x.prod()).groupby('permno').shift(-12).reset_index(drop = True)

# merge with OptionMetrics data
data_m = data_m.merge(msf, on = ['permno','date'], how = 'left')

# change the bounds and returns to percentages
ret_cols = ['ret','rf','sp5ret','ret1','rf1','sp5ret1','ret3','rf3','sp5ret3',
        'ret6','rf6','sp5ret6','ret12','rf12','sp5ret12']

data_m[ret_cols] -= 1
data_m[ret_cols] *= 100

#%%============================================================================
# save the results and clean up intermediate results
# =============================================================================
# remove the stock data folder
shutil.rmtree(stock_data_dir)

cols_to_keep = ['permno', 'secid', 'date', 'lb_kt_1', 'lb_kt_3', 'lb_kt_6', 'lb_kt_12',
                'lb_mw_1', 'lb_mw_3', 'lb_mw_6', 'lb_mw_12', 'delta', 'ret1', 'rf1', 
                'sp5ret1', 'ret3', 'rf3', 'sp5ret3', 'ret6', 'rf6', 'sp5ret6', 'ret12',
                'rf12', 'sp5ret12']

# save the monthly datasets
data_m[cols_to_keep].to_csv(stock_bounds, index = False)




