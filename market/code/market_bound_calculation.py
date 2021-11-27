"""
This file calculates the bounds based on OptionMetrics and extends them using the CBOE data.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as port
from os import chdir
import wrds

# Define the paths
PATH               = PATH_TO_REPLICATION_PACKAGE + 'market/'
CODE_DIR           = PATH + 'code/'
INPUT_DIR          = PATH + 'input/'
INTERMEDIATE_DIR   = PATH + 'intermediate/'
chdir(CODE_DIR)

# input datasets
#    1- options data from CBOE
#    2- original Martin (2017) bounds
#    3- original Chabi-Yo/Loudis (2020) bounds
INPUT_CBOE         = INPUT_DIR + 'CBOE/CBOE_{}.csv'
INPUT_MARTIN       = INPUT_DIR + 'epbound.csv'
INPUT_CYL          = INPUT_DIR + 'cyl.xlsx'

# output of the script 
BOUNDS_DATA        = INTERMEDIATE_DIR + 'market_bounds.csv'


#%%============================================================================
# function definitions
# =============================================================================

# integrate_martin_method implements the integration method used in Martin(2017)
def integrate_martin_method(x,y):
    # convert all data formats to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # drop missing values
    I = (~np.isnan(x)) & (~np.isnan(y))
    x = x[I]
    y = y[I]
    
    # let's construct dx as in Martin (2017) page 420
    dx1 = x[1:] - x[:-1] # this is dx1 = x(i+1) - x(i)
    dx1 = np.append(dx1, 0)
    dx1[0] *= 2
    
    dx2 = x[1:] - x[:-1] # this is dx2 = x(i) - x(i-1)
    dx2 = np.append(0, dx2)
    dx2[-1] *= 2
    
    dx = 0.5*(dx1 + dx2)
    
    return sum(y*dx)    

# market_level_data_cleaner cleans the Options data. The cleaning steps are 
# explained in Appendix 
def Market_Level_Data_Cleaner(db, year):
    
    if year < 1996: # source is CBOE
        # list of standard SPX options
        root_list = ['SPX', 'SZP', 'SPT', 'SXY', 'SPQ', 'SXZ', 'SXB', 'SPZ', 
                     'SXM', 'SYG', 'SPB', 'SYU', 'SVP', 'SPV', 'SPL', 'SXG', 
                     'SYZ', 'SZJ','SZT', 'SZV', 'SYV']
        
        # read the data
        options = pd.read_csv(INPUT_CBOE.format(year))
        
        # keep standard options only
        options = options[options['root'].isin(root_list)]
        
        # keep the required columns only
        options = options[['quote_date','expiry','strike','type','last_bid_price','last_ask_price']]
        
    elif year < 2020: # source is OptionMetrics
        
        # downloading option data and data on underlying securities
        options = db.raw_sql('select * from optionm.opprcd{} where secid = 108105'.format(year))
        
        # just sorting the data 
        options = options.sort_values(by = 'date').reset_index(drop = True)
        
        # Cleaning the data as suggested by Martin 2017
        # keeping only standard AM-settlement options
        options = options[options['symbol'].apply(lambda x: x.split(' ')[0]) != 'SPXW']
        options = options[options['expiry_indicator'].isnull()]
        
        # dropping quarterly options
        options['exmonth'] = options['exdate'].apply(lambda x: x.month)
        options['exday'] = options['exdate'].apply(lambda x: x.day)
        I = (options['exmonth'] % 3 == 0) & (options['exday'] >= 25)
        options = options.loc[~I,:]
        options.drop(columns = ['exmonth','exday'], inplace = True)
        
        # fix the strike price
        options['strike'] = options.strike_price/1000
        options.drop(columns = ['strike_price'], inplace = True)
        
        options = options[['date','exdate','strike','cp_flag','best_bid','best_offer']]
        
    else:  # 2020; source is CBOE
        
        options = pd.read_csv(INPUT_CBOE.format(year))

        # keep AM settlement options only
        options = options[options['root'] == 'SPX']
        options = options[['quote_date','expiration','strike','option_type','bid_eod','ask_eod']]
        
    # rename the columns for uniform treatment
    options.columns = ['date','exdate','strike','cp_flag','best_bid','best_offer']

    # it seems that 998 is the missing code for bid price and 999 is the missi-
    # ng code for the ask price in the CBOE data
    options = options[(options['best_bid'] != 998) & (options['best_offer'] != 999)]
    
    # define days to maturity
    options['date'] = pd.to_datetime(options.date).apply(lambda x: dt.date(x.year,x.month,x.day))
    options['exdate'] = pd.to_datetime(options.exdate).apply(lambda x: dt.date(x.year,x.month,x.day))
    
    # download the S&P500 levels from CRSP
    security = db.raw_sql("select caldt, spindx from crsp.dsp500p where caldt >= '01/01/1990'")
    security = security.rename(columns={'caldt':'date', 'spindx':'close'})      
       
    # creating a few useful variables
    options['ttm'] = (options.exdate - options.date).apply(lambda x: x.days)
    options['prc'] = 0.5*(options.best_bid + options.best_offer)
    
    # drop strike prices that are only available with a put or a call
    options['count'] = options.groupby(['date','exdate','strike'])['prc'].transform('count')
    options = options[options['count'] == 2]
    options.drop(columns = 'count', inplace = True)
    
    # sort the data so that the next step works well
    options.sort_values(by = ['date','exdate','strike','prc'], inplace = True)
    
    # selecting the option with lower midprice in each day; i.e. OTM options
    options = options.groupby(['date','exdate','strike']).first().reset_index()
    
    # dropping options with best_bid = 0 
    options = options[options['best_bid'] != 0]
    
    # drop options with larger than 550 days or less than 8 days to maturity
    options = options[options['ttm'] <= 550]
    options = options[options['ttm'] > 7]
        
    # dropping (date, exdate) with less than 10 strike prices; this is a 
    # step that I added.
    thres = 10
    options['count'] = options.groupby(['date','exdate'])['prc'].transform('count')
    options = options[options['count'] >= thres]
    options.drop(columns = 'count', inplace = True)
    
    # merging the option data with underlying security data
    options = options.merge(security[['date','close']], on = 'date', how = 'left')
    
    # seleting the columns used later
    options = options[['date','exdate','close','ttm','cp_flag','strike','prc']]
    
    return options

# this function calculates the lower bounds in Martin (2017)   
def Martin(options):
    # prepare to calculate the bounds for every day
    grouped = options.groupby(['date','exdate'])
    
    # take out the common variables for every day
    stats = pd.DataFrame()
    stats = grouped[['ttm','close']].last().reset_index()
    
    # calculate the martin bound
    stats['sum_val'] = grouped.apply(lambda g: integrate_martin_method(x = g.strike, y = g.prc)).reset_index(drop = True)
    stats['lb'] = (2.0*stats.sum_val)/(stats.close**2)
    
    # adjust the maturity. I calculate a linear combination of longer and sh-
    # orter maturities to get to the desired maturity. for example, I use 19-
    # day ttm and 47-day ttm to get to 30 day ttm.
    
    # sort stats so that first() and last() functions work as expected
    stats = stats.sort_values(by = ['date','exdate']).reset_index(drop = True)
    
    # define the maturities
    maturities = [30,90,180,360]
    
    # take out the common variables
    martin = stats.groupby('date').last().reset_index()[['date','close']]
    
    for m in maturities:
        
        # create two datasets for shorter-than-target maturities and longer-
        df_less = stats[stats['ttm'] <= m].groupby('date').last().reset_index()[['date','ttm','lb']].rename(columns = {'ttm':'ttm_less','lb':'lb_less'})
        df_more = stats[stats['ttm'] > m].groupby('date').first().reset_index()[['date','ttm','lb']].rename(columns = {'ttm':'ttm_more','lb':'lb_more'})
        
        # merge and interpolate the maturities
        df = df_less.merge(df_more, on = 'date', how = 'outer')
        df['lb_m_'+str(m//30)] = (df['lb_more']*(m - df['ttm_less']) + df['lb_less']*(df['ttm_more'] - m))/(df['ttm_more'] - df['ttm_less'])
        
        # sometimes one of the two sides used for interpolation is missing. Extrap-
        # olate in these cases
        I_less = df['ttm_less'].isnull()
        I_more = df['ttm_more'].isnull()
        
        # extrapolate if one value is missing
        df.loc[I_less,'lb_m_' + str(m//30)] = df['lb_more']*m/df['ttm_more']
        df.loc[I_more,'lb_m_' + str(m//30)] = df['lb_less']*m/df['ttm_less']
        
        # merge with the martin dataset
        martin = martin.merge(df[['date','lb_m_'+str(m//30)]], on = 'date', how = 'outer')
        
        # annualize the bound and multiply by 100
        martin['lb_m_'+str(m//30)] *= (360/m)*100
        
    return martin

# ChabiYoLoudis_Lower_Bounds gets the data for a specific (date,ttm) and calcu-
# lates the CYLR lower bounds.
def ChabiYoLoudis_Lower_Bounds(options):
    ###############################################################################
    # Calculating the moments
    ###############################################################################
    moments = np.zeros((1,10))
    M = [0,0,0]
    
    for n in (2,3,4):
        M[n-2] = integrate_martin_method(x = options['strike'], y = n*(n-1)*options['rf_t_T']/options['close']**2 * ((options['strike']/options['close'] - options['rf_t_T'])**(n-2) * options['prc']))
    
    rf_t_T = options['rf_t_T'].mean()
    close = options['close'].mean()
    rf = options['rf'].mean()
    
    ###############################################################################
    # Calculating the lower bounds
    ###############################################################################
    
    # defining parameters as estimated in Chabi-Yo and Loudis
    tau = 0.97
    rho = 2.32
    kappa = 3.50
    
    a_1 = 1/tau
    a_2 = (1 - rho)/tau**2
    a_3 = (1 - 2*rho + kappa)/tau**3
    
    # calcultaing the lower bounds
    
    theta_1 = a_1/rf_t_T
    theta_2 = a_2/rf_t_T**2
    theta_3 = a_3/rf_t_T**3
    
    lbr = (M[0]/rf_t_T - M[1]/rf_t_T**2 + M[2]/rf_t_T**3)/(1 - M[0]/rf_t_T**2 + M[1]/rf_t_T**3)
    
    moments[0,:3] = [close, rf, rf_t_T]
    moments[0,3:6] = M
    moments[0,6:] = [theta_1, theta_2, theta_3, lbr]
    
    moments = pd.DataFrame(data = moments, columns = ['close', 'rf', 'rf_t_T', 'M2','M3','M4','theta_1','theta_2','theta_3', 'lbr'])
    return moments

# ChabiYoLoudis calculates the lower bounds of Chabi-Yo and Loudis (2020)
# for the entire dataset; it uses ChabiYoLoudis_Lower_Bounds.
def ChabiYoLoudis(options):
    
    # rf created here is monthly rate; meaning it is the gross return on 3 month 
    # t-bill from index date to one-month after index date.
    rf = port.DataReader('DGS3MO', 'fred', start = dt.date(1900,1,1), end=dt.date(2022,12,31))
    rf = (1 + rf/400)**(1/3)
    
    rf = pd.DataFrame(rf).reset_index()
    rf.rename(columns = {'DATE':'date','DGS3MO':'rf'},inplace = True)
    rf['date'] = rf['date'].apply(lambda x: dt.date(x.year,x.month,x.day))
    rf['rf'] = rf['rf'].fillna(method = 'ffill')
    
    options = options.merge(rf, on = 'date', how = 'left')
    options['rf_t_T'] = options['rf']**(options['ttm']/31)
    options = options.sort_values(by = ['date','ttm','strike','cp_flag']).reset_index(drop = True)
    
    grpd = options.groupby(['date','exdate','ttm'])
    moments = grpd.apply(ChabiYoLoudis_Lower_Bounds)
    moments = moments.reset_index().drop(columns = ['level_3'])
    
    # interpolating the bounds to reach at 30, 60, 90, 180, and 360 day maturity
    df = moments.groupby('date').last().reset_index()[['date','close','rf']]
    
    for m in (30,90,180,360):
        df2 = moments[moments['ttm'] <= m].fillna('nan').groupby('date').last().replace('nan', np.nan).reset_index()[['date','lbr','ttm']].rename(columns = {'lbr':'lbr_less', 'ttm':'ttm_less'})
        df3 = moments[moments['ttm'] > m].fillna('nan').groupby('date').first().replace('nan', np.nan).reset_index()[['date','lbr','ttm']].rename(columns = {'lbr':'lbr_more', 'ttm':'ttm_more'})
        
        df4 = df2.merge(df3,on='date',how = 'outer')
        df4['lb_cylr_'+ str(m//30)] = (df4['lbr_more']*(m - df4['ttm_less']) + df4['lbr_less']*(df4['ttm_more'] - m))/(df4['ttm_more'] - df4['ttm_less'])
 
        # sometimes one of the two sides used for interpolation is missing. Extrap-
        # olate in these cases
        I_less = df4['ttm_less'].isnull()
        I_more = df4['ttm_more'].isnull()
      
        df4.loc[I_less,'lb_cylr_' + str(m//30)] = df4['lbr_more']*m/df4['ttm_more']
        df4.loc[I_more,'lb_cylr_' + str(m//30)] = df4['lbr_less']*m/df4['ttm_less']
    
        df = df.merge(df4[['date','lb_cylr_'+ str(m//30)]],on = 'date',how = 'outer')
        
        df['lb_cylr_'+ str(m//30)] = df['lb_cylr_'+ str(m//30)]*(360/m)*100
        
    return df
    
#%%============================================================================
# bound calculations
# =============================================================================

martin = []
cyl = []

# this loop now calculates the bounds
start = 1990
stop = 2020

db = wrds.Connection()

for year in range(start,stop + 1):  
    
    print('Now processing ' + str(year) + ':')
    for at in range(1,4):
        try:
            print("Attempt " + str(at) + ' ...')
            options = Market_Level_Data_Cleaner(db, year)
            break
        except:
            print('     Failed.')
        
    print('    Calculating Martin bounds ...')
    martin_append = Martin(options)
    
    print('    Calculating ChabiYo/Loudis bounds ...')              
    cyl_append = ChabiYoLoudis(options)
    
    martin.append(martin_append)
    cyl.append(cyl_append) 

db.close()

#%%============================================================================
# Put everything together
# =============================================================================
# get all our bounds
martin = pd.concat(martin, ignore_index = True)
cyl = pd.concat(cyl, ignore_index = True)

# read the original Martin (2017) data
martin_orig = pd.read_csv(INPUT_MARTIN)
martin_orig['date']= pd.to_datetime(martin_orig[['year','month','day']])
martin_orig = martin_orig[['date','epbound_1','epbound_3','epbound_6','epbound_12']]
martin_orig[martin_orig.columns.drop('date')] *= 100
martin_orig['date'] = martin_orig['date'].apply(lambda x: dt.date(x.year, x.month, x.day))

# read the original cyl data
cyl_orig = pd.read_excel(INPUT_CYL, sheet_name='Data')
cyl_orig['date'] = cyl_orig['date'].apply(lambda x: dt.date(x.year, x.month, x.day))
cyl_orig[cyl_orig.columns.drop('date')] *= 100
for m in [1,2,3,6,12]:
    cyl_orig['lb_cylr_orig_'+str(m)] = cyl_orig['LBR_'+str(m*30)]*12/m

cyl_orig = cyl_orig[['date','lb_cylr_orig_1','lb_cylr_orig_3','lb_cylr_orig_6','lb_cylr_orig_12']]

# merge everything together
df = martin.merge(martin_orig, how = 'left', on = 'date').rename(columns = {'epbound_1':'lb_m_orig_1','epbound_3':'lb_m_orig_3','epbound_6':'lb_m_orig_6','epbound_12':'lb_m_orig_12'})
df = df.merge(cyl, on ='date', how = 'outer').drop(columns = 'close_y').rename(columns = {'close_x':'close'})
df = df.merge(cyl_orig, on ='date', how = 'outer')

# drop irrelevant columns
df = df.drop(columns=['rf', 'close'])

# save the data
df['date'] = df['date'].apply(lambda x: dt.datetime(x.year,x.month,x.day))
df.to_csv(BOUNDS_DATA, index=False)
