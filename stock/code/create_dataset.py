from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import seaborn as sns

sns.set()
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 250)


# input data that is used in this script
path            = PATH_TO_REPLICATION_PACKAGE + 'stock/'
BOUNDS_DATA     = path + 'intermediate/stock_bounds.csv'
GW_DATA         = path + 'input/PredictorData2020.xlsx'

# output of the script 
DS_MONTHLY      = path + 'intermediate/ds_stock_monthly.csv'

#%%########################################################################
# Combine returns, bounds, and conditioning variables
###########################################################################

bounds = pd.read_csv(BOUNDS_DATA)

# Calculate forward realized excess returns and annualize 
bounds['f_xret1'] =(bounds['ret1'] - bounds['rf1'])*12
bounds['f_xret3'] =(bounds['ret3'] - bounds['rf3'])*4
bounds['f_xret6'] =(bounds['ret6'] - bounds['rf6'])*2
bounds['f_xret12']=bounds['ret12'] - bounds['rf12']

# pull Goyal-Welch variables and define positive versions
# url = 'http://www.hec.unil.ch/agoyal/docs/PredictorData2020.xlsx'
gw = pd.read_excel(GW_DATA, sheet_name='Monthly',index_col='yyyymm')
gw['dp']                = 5 + np.log(gw.D12) - np.log(gw['Index'])
gw['ep']                = 5 + np.log(gw.E12) - np.log(gw['Index'])
gw['bm']                = gw['b/m']
gw['dfy']               = gw.BAA - gw.AAA
gw['ntis_plus1']        = 1 + gw.ntis
gw['lty_spread_plus1']  = 1 + gw.lty - gw.tbl 
gw['infl_plus1']        = 1 + gw.infl
gw_vars = ['dp', 'ep', 'bm', 'tbl', 'lty_spread_plus1', 'dfy', 'svar', 'ntis_plus1', 'infl_plus1']
gw = gw[gw_vars]

# combine bounds/returns with conditioning variables
ds = pd.merge(bounds, gw, how='inner', left_on='date', right_index=True)
ds.drop(columns=['rf1', 'ret1','sp5ret1','rf3', 'ret3','sp5ret3','rf6', 'ret6','sp5ret6','rf12', 'ret12','sp5ret12'] , inplace=True)


#save to intermediate location
ds.to_csv(DS_MONTHLY, index=False)
