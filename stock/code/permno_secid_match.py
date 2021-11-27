"""
This file collects S&P 500 constituents and matches them to OptionMetrics.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import pandas as pd
import numpy as np
import datetime as dt
import wrds

#%%============================================================================
# define directories
# =============================================================================

input_dir          = PATH_TO_REPLICATION_PACKAGE + "stock/input/"
intermediate_dir   = PATH_TO_REPLICATION_PACKAGE + "stock/intermediate/"

CRSP_OM_LINK       = input_dir + 'crsp_om_link.csv'
CRSP_OM_LINK_CLEAN = intermediate_dir + 'crsp_om_link_clean.csv'

#%%============================================================================
# main code 
# =============================================================================

# get the lining table between CRSP and OptionMetrics
link = pd.read_csv(CRSP_OM_LINK)
link['sdate'] = pd.to_datetime(link['sdate'])
link['edate'] = pd.to_datetime(link['edate'])
link = link.query('score != 6')
link.rename(columns = {'PERMNO':'permno'}, inplace = True)
link['sdate'] = link['sdate'].apply(lambda x: dt.date(x.year,x.month,x.day))
link['edate'] = link['edate'].apply(lambda x: dt.date(x.year,x.month,x.day))
link['edate'] = link['edate'].replace(dt.date(2019,6,28), dt.date(2030,12,31))

# fix some errors in the link file
link.loc[(link['permno'] == 66157) & (link['secid'] == 111306), 'edate'] = dt.date(2001,2,26)

# get the constituents from CRSP
db = wrds.Connection()
crsp_const = db.get_table('crsp', 'dsp500list')

crsp_const = crsp_const[crsp_const['ending'] >= dt.date(1996,1,1)]

# merge the constituents with the link to get secid
df1 = crsp_const.merge(link, on = 'permno', how = 'left')

# keep only cases when a valid link is in the S&P 500
df1['start'] = pd.to_datetime(df1['start'])
df1['ending'] = pd.to_datetime(df1['ending'])
df1['sdate'] = pd.to_datetime(df1['sdate'])
df1['edate'] = pd.to_datetime(df1['edate'])

df1['max_start'] = df1[['start','sdate']].max(axis = 1)
df1['min_end'] = df1[['ending','edate']].min(axis = 1)
df1 = df1[df1['max_start'] <= df1['min_end']]
df1 = df1.dropna(how = 'any')

# keep only the highest quality of matches. I looked at the lower quality 
# matches; they do not add anything.
df1 = df1[df1['score'] == 1]

# let's look at cases where multiple matches are found for a permno
df1['tag'] = df1.groupby('permno')['secid'].transform('nunique')

df2 = df1[df1['tag'] == 1]
df3 = df1[df1['tag'] != 1]

# let's see how many valid price observations we have for each secid
df3['start_year'] = df3['max_start'].apply(lambda x: x.year)
df3['end_year'] = df3['min_end'].apply(lambda x: x.year)
df_prcs = []
for index, row in df3.iterrows():
    prc = []
    for i in range(row['start_year'], row['end_year'] + 1):
        temp = db.raw_sql("select * from optionm.secprd{} where secid = {}".format(i, row['secid']))
        prc.append(temp)
    
    prc = pd.concat(prc, axis = 0)
    df_prcs.append(prc)  
db.close()
df3['obs'] = np.array([len(x) for x in df_prcs])

# drop matches with no price observation
df3 = df3[df3['obs'] != 0]

# tag chains of secids matching one permno
df3['lag_edate'] = df3.groupby('permno')['edate'].shift(1).fillna(dt.datetime(1995,12,31))
df3['tag2'] = 1.0*(df3['lag_edate'] < df3['sdate'])
df3['tag2'] = df3.groupby('permno')['tag2'].transform('prod')

# keep the chains and drop the minimum observation for non-chains
df4 = df3[df3['tag2'] == 1]
df5 = df3[df3['tag2'] != 1]

df5 = df5.sort_values(by = ['permno', 'obs'])
df5 = df5.groupby('permno').last().reset_index()

# join the clean links and save
df = pd.concat([df2, df4, df5], axis = 0)
cols = ['permno', 'secid', 'start', 'ending', 'sdate', 'edate']
df = df[cols]
df.columns = ['permno', 'secid', 'sp_beg', 'sp_end', 'lnk_beg', 'lnk_end']

df.to_csv(CRSP_OM_LINK_CLEAN, index = False)
