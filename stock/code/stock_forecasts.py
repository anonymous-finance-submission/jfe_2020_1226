"""
This script runs out-of-sample forecasting tests for the stock bounds.  It produces a csv file
containing t-stats, p-values, and R2's for Diebold-Mariano tests comparing the forecasts to 
the expanding market mean benchmark (there is also code for a benchmark of zero in the script).  
It also produces a csv file containing t-stats and p-values for 10-1 portfolios formed from 
sorting on forecasts.
"""
from __main__ import PATH_TO_REPLICATION_PACKAGE
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)  # to suppress meaningless pandas performance warnings

import os

#%%============================================================================
# define directories
# =============================================================================

PATH            = PATH_TO_REPLICATION_PACKAGE + 'stock/'
INPUTPATH       = PATH + 'input/'                # location of Stock_Level_Bounds_dta and rpsdata_rfs.sas7bdat
OUTPUTPATH      = PATH + 'tables/'              # location to output the csv files and tex files
INTERMEDIATE    = PATH + 'intermediate/'      # location to output the forecasts

STOCKCHARS      = INPUTPATH + 'rpsdata_rfs.csv'        # created from Jeremiah Green's SAS code for Green-Hand-Zhang
BNDSRETS        = INTERMEDIATE + 'ds_stock_monthly.csv'

#%%============================================================================
# define functions
# =============================================================================

# size of initial window for expanding panel results
MIN_WINDOW = 60     

# names of characteristics in GHZ dataset                               
CHARS = ['mve','bm','agr','operprof','mom12m'] 

# specifies the number of lags to use as a function of the horizon in the HH std errs
def LAG(h) :     
    return h

# define ML models
ols = LinearRegression()

# create_data reads the stock-level bounds and merges them with some characteristics
def create_data() :
    
    # read the stock characteristics
    df = pd.read_csv(STOCKCHARS, dtype={'permno':'int', 'date':'period[M]'})
    df.set_index(['date','permno'],inplace=True)
       
    # read the bounds and return data
    bds = pd.read_csv(BNDSRETS)
    for col in ['permno','date'] :
        bds[col] = bds[col].astype(int)
    bds.date = pd.to_datetime(bds.date.astype(str),format='%Y%m').dt.to_period('M')
    bds.set_index(['permno','date'],inplace=True)
    bds.sort_index(inplace=True)
    
    # keep only bounds, forward returns, and deltas
    cols = [x + repr(h) for x in ['lb_kt_','lb_mw_','f_xret'] for h in [1,3,6,12]]
    bds = bds.rename(columns={'delta_mkt':'delta'})
    bds = bds[cols+['delta']]
    df = bds.join(df,how='left')
    
    # winsorize and standardize each characteristic in each cross-section
    def stdize(d) : 
        d2 = d.clip(d.quantile(0.01),d.quantile(0.99),axis=1)
        return (d2 - d2.mean()) / d2.std()
    
    df[CHARS] = df[CHARS].groupby('date').apply(stdize)

    # compute expanding-window market mean
    factors = pdr.DataReader('F-F_Research_Data_Factors','famafrench','1926-01-01')[0]
    mktrf = 12*factors['Mkt-RF'].expanding().mean()
    mktrf.index.name = 'date'
     
    return df, mktrf


def diebold_mariano_stocks(y,fcst,bmark,lags) :
    ''' Test for H0: benchmrk sqd err = fcst sqd err
        returns tstat, pvalue, and R-squared relative to the benchmark
        y = return series
        fcst = series of forecasts
        bmark = series of benchmark forecasts
        lags = number of lags for Hansen-Hodrick std error
    '''
    df = pd.concat((y,fcst),axis=1)
    df = df.join(bmark,how='left').dropna()
    r, f, b = df.columns.to_list()
    
    # total mse's
    ftot = ((df[r]-df[f])**2).groupby('date').mean()
    btot = ((df[r]-df[b])**2).groupby('date').mean()  
    
    # time series
    fm = df[f].groupby('date').mean()
    bm = df[b].groupby('date').mean()
    rm = df[r].groupby('date').mean()
    fts = (fm-rm)**2
    bts= (bm-rm)**2
    
    # cross-section
    rdevs = df[r].groupby('date').apply(lambda x: x-x.mean())
    fdevs = df[f].groupby('date').apply(lambda x: x-x.mean())
    bdevs = df[b].groupby('date').apply(lambda x: x-x.mean())
    fxs = ((rdevs-fdevs)**2).groupby('date').mean()
    bxs = ((rdevs-bdevs)**2).groupby('date').mean()
    
    # put everything together
    table = pd.DataFrame(dtype=float,columns=['total','tseries','xsection'],index=['t','p','R2'])
    for x,y,z in zip((ftot,fts,fxs),(btot,bts,bxs),('total','tseries','xsection')) :
        result = sm.OLS(y-x,np.ones(len(y))).fit(cov_type='HAC',cov_kwds={'maxlags':lags,'kernel':'uniform'})
        t = result.tvalues.item()
        p = result.pvalues.item()
        table.loc['t',z] = t
        table.loc['p',z] = p/2 if t>0 else 1-p/2
        table.loc['R2',z] = 1 - x.mean()/y.mean()
        
    return table


# ExpandingPanelML creates forecasts using Machine Learning techniques
def ExpandingPanelML(data,y,xvars,model,min_window,h) :
    d = data[[y]+xvars].dropna()
      
    # define end-of-training-period dates and forecast dates
    dateidx = d.index.get_level_values('date')
    mindate = dateidx.min().to_timestamp()
    maxdate = dateidx.max().to_timestamp()

    train_start = mindate + pd.DateOffset(months=min_window-1)
    train_end =   maxdate + pd.DateOffset(months=1-h)
    train_dates = pd.date_range(train_start,train_end,freq='M').to_period('M')

    fcst_start = mindate + pd.DateOffset(months=min_window+h-1)
    fcst_end =  maxdate + pd.DateOffset(months=1)
    fcst_dates = pd.date_range(fcst_start,fcst_end,freq='M').to_period('M')
    
    # run the 'train and predict' loop
    fcsts = None
    for past, future in zip(train_dates,fcst_dates) :
        Train = d[dateidx <= past]
        Test = d[dateidx == future]
        XTrain = Train[xvars]
        XTest = Test[xvars]
        YTrain = np.ravel(Train[y])
        model.fit(XTrain,YTrain)
        fcst = pd.Series(model.predict(XTest).flatten(),index=XTest.index)
        fcsts = pd.concat((fcsts,fcst))
        
    return fcsts   

def oos_table(d,mkt) :
    ''' creates table with R2 and p-value for different forecasts 
        compares models to the market benchmark and to each other
        also outputs forecasts
    '''
    modelnames = ['tight bound', 'bound OLS','Combination(FF)','Combination(FF)+bound']

    rowindx = pd.MultiIndex.from_product([['not_truncated','truncated'],modelnames,['xsection','tseries','total'],['t-stat','p-value','R2']])
    colindx = pd.MultiIndex.from_product([['lb_mw','lb_kt'],['1','3','6','12']])
    table = pd.DataFrame(dtype=float,index=rowindx,columns=colindx)
    
    l_fcsts = []
    l_fcsts_lb = []
    for b in ['lb_mw','lb_kt']:
        
        if b == 'lb_kt':
            d = d[d.delta <= 3]
            
        for h in [1,3,6,12] :
            
            y = 'f_xret'+str(h)
            bnd = b+'_'+str(h)
            
            fcsts = pd.DataFrame(dtype=float,index=d.index,columns=modelnames)

            # single forecasts
            print('working on tight bound for ' + bnd)
            fcsts['tight bound'] = d[bnd]
            
            print('working on bound OLS for ' + bnd)
            fcsts['bound OLS'] = ExpandingPanelML(d,y,[bnd],ols,MIN_WINDOW,h)
            
            # Combination forecasts - use univariate forecasts of return
            varlist=[]
            for v in CHARS:
                fcsts['runivariate_'+v] = ExpandingPanelML(d, y, [v], ols, MIN_WINDOW, h)
                varlist.append('runivariate_'+v)
            fcsts['Combination(FF)'] = fcsts[varlist].mean(axis=1)
            fcsts.drop(columns = varlist, inplace = True)
            
            # combo of FF vars + bound
            fcsts['Combination(FF)+bound'] = 0.5*fcsts['Combination(FF)'] + 0.5*fcsts['tight bound']
            
            # generate truncated forecasts
            fcsts_lb = fcsts.copy()
            for m in modelnames:
                if m == 'Combination(FF)':
                    # truncate at zero
                    fcsts_lb[m] = np.maximum(fcsts_lb[m], 0)
                else:      
                    # truncate at the bound
                    fcsts_lb[m] = np.maximum(fcsts_lb[m], fcsts_lb['tight bound'])
            
            # compare forecasts to market
            for m in modelnames :
                print('working on DM for model ' + m + ' for bound ' + bnd)
                dm = diebold_mariano_stocks(d[y],fcsts[m],mkt,LAG(h))
                for col in dm.columns :
                    table.loc[('not_truncated',m,col,'t-stat'),(b,str(h))] = dm.loc['t',col]
                    table.loc[('not_truncated',m,col,'p-value'),(b,str(h))] = dm.loc['p',col]
                    table.loc[('not_truncated',m,col,'R2'),(b,str(h))] = dm.loc['R2',col]
                    
                print('working on DM for model ' + m + ' for bound ' + bnd + ' with truncation')
                dm = diebold_mariano_stocks(d[y],fcsts_lb[m],mkt,LAG(h))
                for col in dm.columns :
                    table.loc[('truncated',m,col,'t-stat'),(b,str(h))] = dm.loc['t',col]
                    table.loc[('truncated',m,col,'p-value'),(b,str(h))] = dm.loc['p',col]
                    table.loc[('truncated',m,col,'R2'),(b,str(h))] = dm.loc['R2',col]
                    
            # collect forecasts into a comprehensive dataframe
            fcsts = fcsts.rename(columns = dict(zip(fcsts.columns,[bnd+'_'+x for x in fcsts.columns])))
            l_fcsts.append(fcsts)
            
            fcsts_lb = fcsts_lb.rename(columns = dict(zip(fcsts_lb.columns,[bnd+'_'+x for x in fcsts_lb.columns])))
            l_fcsts_lb.append(fcsts_lb)
    
    fcsts = pd.concat(l_fcsts, axis=1)
    fcsts_lb = pd.concat(l_fcsts_lb, axis=1)
    Fcsts = pd.concat([fcsts, fcsts_lb], keys=['not_truncated', 'truncated'])

    return table, Fcsts


#%%============================================================================
# main code
# =============================================================================

# create a monthly dataset 
df, mktrf = create_data()
        
# generate table 10 from DM tests of forecasts
table, fcsts = oos_table(df,mktrf)

#%%============================================================================
# create the latex tables
# =============================================================================

# this function is used in oosr2_tables
def tex_wrapper(table, file_name, header_list):
    table.to_latex(file_name +'_pre.tex', escape=False, header = header_list)
    # make latex file a fragment (comment begin/end tabular)
    file_pre = open(file_name +'_pre.tex', 'r')
    file_out = open(file_name +'.tex', 'w')
    checkWords = ('\\begin', '\end','label', '\\toprule')
    repWords = ('%\\begin', '%\end','%label', '%\\toprule')
    for i, line in enumerate(file_pre):
        for check, rep in zip(checkWords, repWords):
            line = line.replace(check, rep)
        file_out.write(line)
    file_pre.close()
    file_out.close()    
    os.remove(file_name +'_pre.tex')  

# oosr2_tables writes the results into latex tables
def oosr2_tables(oos, b, fileprefix, drop_tight_bound=False):
    oos_r2 = oos[oos.index.get_level_values(level=2).isin(['R2'])]
    oos_p  = oos[oos.index.get_level_values(level=2).isin(['p-value'])]
    print(oos_r2.round(3))
    print(oos_p.round(3))
    
    fcst_labels = ['(0) Tight Bound', '(1) OLS on Bound', '(2) Combination (FF)', '(3) Combination (FF + Bound)']

    
    indx = pd.MultiIndex.from_product([fcst_labels,['Cross-section','Time-series','Total']])  
    
    # convert to string table to facilitate latex formatting
    st = oos_r2[[(b,'1'),(b,'3'),(b,'6'),(b,'12')]]
    st.columns = st.columns.droplevel(0)      
    st.index = st.index.droplevel(2)
    
    stp = oos_p[[(b,'1'),(b,'3'),(b,'6'),(b,'12')]]
    stp.columns = stp.columns.droplevel(0)      
    stp.index = stp.index.droplevel(2)    
    print(stp)
      
    st2 = pd.DataFrame(dtype=str,index=st.index,columns=st.columns)           #string, formatted version
                 
    for h in [1,3,6,12]:
        coef_string = st[str(h)].apply(lambda x: f'{x:.3f}').values      
        pvals = stp[str(h)]
        st2[str(h)] =np.where(pvals<=0.01, coef_string + '$^{***}$', 
                                   np.where(pvals<=0.05, coef_string + '$^{**}$',
                                   np.where(pvals <=0.1, coef_string + '$^{*}$', coef_string)))
        
        r2_num = st[str(h)]
        r2_str = st2[str(h)]
        st2[str(h)] = np.where(r2_num > 0.0    ,  r2_str + '\cellcolor{lightgray}', r2_str)

    st2.index = indx
    stp.index = indx
    
    # keep the decomposition into XS and TS only for the case of tight bound
    index_to_keep = [x for x in indx if (x[0] == '(0) Tight Bound') or (x[1] == 'Total')] 
    st2 = st2.loc[index_to_keep, :]
    
    # add an extra row to the dataframe
    row= pd.DataFrame([['','','','']],index = pd.MultiIndex.from_product([['(0) Tight Bound'],['']]), columns=['1','3','6','12'])
    st2 = pd.concat([row,st2])
    indx = ['(0) Tight Bound', '\hspace{2em} Cross-Section', '\hspace{2em} Time-Series', '\hspace{2em} Total',
            '(1) OLS on Bound', '(2) Combination (FF)', '(3) Combination (FF + Bound)']
    st2.index = indx
    if drop_tight_bound:
        st2 = st2.iloc[4:,:]
    my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in st2.columns]
    tex_wrapper(table=st2, file_name=OUTPUTPATH + fileprefix + b, header_list=my_header)
    
    # now do the same for p-Values
    stp = stp*100
    stp = stp.round(1)
    stp = stp.loc[index_to_keep, :]
    stp = pd.concat([row,stp])
    indx = ['(0) Tight Bound', '\hspace{2em} Cross-Section', '\hspace{2em} Time-Series', '\hspace{2em} Total',
            '(1) OLS on Bound', '(2) Combination (FF)', '(3) Combination (FF + Bound)']
    stp.index = indx
    if drop_tight_bound:
        stp = stp.iloc[4:,:]
    my_header = ['\multicolumn{1}{c}{'+str(x) +'}' for x in stp.columns]
    tex_wrapper(table=stp.round(1), file_name=OUTPUTPATH+ fileprefix + 'pvals_' + b, header_list=my_header)  


##### finally, write the tables

oosr2_tables(table.xs('not_truncated', level=0),'lb_mw', 'OOS_sigstars_')
oosr2_tables(table.xs('not_truncated', level=0),'lb_kt', 'OOS_sigstars_')

oosr2_tables(table.xs('truncated', level=0),'lb_mw', 'OOS_sigstars_truncated_', drop_tight_bound=True)
oosr2_tables(table.xs('truncated', level=0),'lb_kt', 'OOS_sigstars_truncated_', drop_tight_bound=True)