"""
This file replicates all the tables and figures related to the analysis of 
stock bounds in: "Validity, Tightness, and 
Forecasting Power of Risk Premium Bounds".

Stock bounds are those of Martin and Wagner (2019) and Kadan and Tang (2020).
"""

#%% 0 =========================================================================
# This cell controls the flow of the code.
# =============================================================================

# First, enter the path to the replication package.
# Do not include replication package in the name but do include the final '/' 
# in the path. Otherwise you will get an error. Here are some examples:
#     Correct   : '/home/sk83/Documents/replication_package/'
#     Incorrect : '/home/sk83/Documents'
#     Incorrect : '/home/sk83/Documents/'
#     Incorrect : '/home/sk83/Documents/replication_package'
PATH_TO_REPLICATION_PACKAGE = '/sharedata/bck/replication_package/'

# DO NOT CHANGE THE FOLLOWING LINE:
code_dir = PATH_TO_REPLICATION_PACKAGE + 'stock/code/'


# Second, toggle each variable to True if you want the corresponding step to 
# run. Each step overrides some of the files in the package. If you choose 
# not to run a step, you will be using the existing files.

CREATE_CRSP_OPTIONMETRICS_LINK             = False
CALCULATE_STOCK_BOUNDS_FROM_SCRATCH        = False
CREATE_ANALYSIS_DATASETS_FROM_SCRATCH      = False

#%% 1 =========================================================================
# Step 1: Create CRSP-OptionMetrics Link.
# If you wish to skip this step and use the existing link, toggle the variable 
# CREATE_CRSP_OPTIONMETRICS_LINK to False.
# 
# To run this step, you need:
#     - A connection to WRDS. The program will prompt you to enter your WRDS 
#       username and password.
#     - The following Python packages: wrds
# 
# Approximate run time for this step is less than 1 minute.
# =============================================================================

# The inputs to this file are:
#    1- CRSP/OptionMetrics linking table from WRDS: stock/input/crsp_om_link.dta
# Its outputs are:
#    1- A cleaned link between S&P500 constituents in CRSP and OptionMetrics:
#              stock/intermediate/crsp_om_link_clean.csv
if CREATE_CRSP_OPTIONMETRICS_LINK:
    exec(open(code_dir + 'permno_secid_match.py').read())
    
#%% 2 =========================================================================
# Step 2: Calculate stock bounds.
# If you wish to skip this step and use the existing simulation results, toggle 
# CALCULATE_STOCK_BOUNDS_FROM_SCRATCH to False.
# 
# To run this step, you need:
#     - A connection to WRDS. The program will prompt you to enter your WRDS 
#       username and password 3 times.
#     - The following Python packages: wrds
#     
# Approximate run time for this step is 1 day.
# =============================================================================

# The inputs to this file are:
#    1- A cleaned link between S&P500 constituents in CRSP and OptionMetrics:
#              stock/intermediate/crsp_om_link_clean.csv
#    2- Risk-free rates from Fama/French files: stock/input/rf_famafrench.csv
# Its outputs are:
#    1- Stock bounds of MW and KT: stock/intermediate/stock_bounds.csv
    
if CALCULATE_STOCK_BOUNDS_FROM_SCRATCH:
    exec(open(code_dir + 'stock_bound_calculation.py').read())

#%% 3 =========================================================================
# Step 3: Create the stock level dataset.
# If you wish to skip this step and use the existing simulation results, toggle 
# CREATE_ANALYSIS_DATASETS_FROM_SCRATCH to False.
#     
# Approximate run time for this step is less than 1 minute.
# =============================================================================

# The inputs to this file are:
#    1- Stock bounds of MW and KT: stock/intermediate/stock_bounds.csv
#    2- Goyal/Welch variables: stock/input/PredictorData2020.csv
# Its outputs are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
if CREATE_ANALYSIS_DATASETS_FROM_SCRATCH:
    exec(open(code_dir + 'create_dataset.py').read())
    
#%% 4 =========================================================================
# Step 4: Generate summary statistics and time series figures.
# 
# This step replicates:
#     - Table 2
#     - Figures 4, 5, 6, 7, 8
#     
# Approximate run time for this step is less than 1 minute.
# =============================================================================

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Summary stats for stock bounds (table 2): stock/tables/panel_summary_stas_*.tex
exec(open(code_dir + 'summary_table.py').read())

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Plot of stock bounds by delta (figure 4): stock/tables/stockbounds_bydelta.pdf
exec(open(code_dir + 'plot_stockbounds_bydelta.py').read())

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Plot of comparison between MW and KT (figure 5): stock/tables/compare_KTMW.pdf
exec(open(code_dir + 'plot_compare_KTMW.py').read())

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Plot of realized returns vs MW (figure 6): stock/tables/MW_bound_return.pdf
#    2- Plot of realized returns vs KT for conservative stocks (figure 7): stock/tables/KT_conservative_bound_return.pdf
#    3- Plot of realized returns vs KT for liberal stocks (figure 8): stock/tables/KT_liberal_bound_return.pdf
exec(open(code_dir + 'plot_bounds_returns.py').read())


#%% 5 =========================================================================
# Step 5: Run Kodde/Palm tests of validity and tightess.
# 
# This step replicates:
#     - Tables 4, IA.2
#
# To run this step, you need:
#     - A valid license to Gurobi with its Python package installed and loaded.
#       For more information on Gurobi, look at https://www.gurobi.com/ .
#     - The following Python packages: gurobipy
# 
# Approximate run time for this step is 90 minutes.
# =============================================================================

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Result of Kodde/Palm tests for stock bounds (table 4): stock/tables/kp_tests_lb(ub)_valid(tight).pdf
# Approximate run time is 35 minutes
exec(open(code_dir + 'validity_tightness_tests_stock.py').read())

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Result of Kodde/Palm tests for average stock bounds (table IA.2): stock/tables/kp_tests_avgbound_*.pdf
exec(open(code_dir + 'validity_tightness_tests_stock_sample_means.py').read())


#%% 6 =========================================================================
# Step 6: Full-sample regressions.
# 
# This step replicates:
#     - Tables 7, IA.4
# 
# To run this step, you need:
#     - The following Python packages: arch, sigtable
#
# Approximate run time for this step is 1 hour.
# =============================================================================


# Fama-MacBeth regressions

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
#    2- Stock characteristics from Jeremiah Green's website: stock/input/rpsdata_rfs.csv
# Its outputs are:
#    1- Table of univariate Fama-MacBeth regressions (table 7): stock/tables/FM_uni.tex
#    2- Table of multivariate Fama-MacBeth regressions (table IA.4): stock/tables/FM_multi.tex
exec(open(code_dir + 'FamaMacBeth.py').read())

# panel regressions

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
#    2- Stock characteristics from Jeremiah Green's website: stock/input/rpsdata_rfs.csv
# Its outputs are:
#    1- Table of univariate panel regressions with stock FE (table 7): stock/tables/panel_fixed_uni.tex
#    2- Table of univariate panel regressions without stock FE (table 7): stock/tables/panel_nofixed_uni.tex
#    3- Table of multivariate panel regressions with stock FE (table IA.4): stock/tables/panel_fixed_uni.tex
#    4- Table of multivariate panel regressions without stock FE (table IA.4): stock/tables/panel_nofixed_multi.tex
exec(open(code_dir + 'panel_regressions.py').read())



#%% 7 =========================================================================
# Step 7: Out-of-sample forecasts.
# 
# This step replicates:
#     - Tables 9, 10, IA.6
#     - Figures 14, 15
# 
# To run this step, you need:
#     - The following Python packages: pandas_datareader, warnings
#     
# Approximate run time for this step is 10 minutes.
# =============================================================================

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Plot of cumulative SSE for all stocks (figure 14): stock/figures/cum_sse_all.pdf
#    2- Plot of cumulative SSE for conservative stocks (figure 15): stock/figures/cum_sse_conservative.pdf
exec(open(code_dir + 'plot_group_errors.py').read())

# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
#    2- Stock characteristics from Jeremiah Green's website: stock/input/rpsdata_rfs.csv
# Its outputs are:
#    1- OOS tests of stock forecasts (table 9): stock/tables/OOS_sigstars_lb_*.tex 
#                                                              and stock/tables/OOS_sigstars_truncated_lb_*.tex
#    2- p-Values for OOS tests of stock forecasts (table IA.6): stock/tables/OOS_sigstars_pvals_lb_*.tex 
#                                                              and stock/tables/OOS_sigstars_pvals_truncated_lb_*.tex
exec(open(code_dir + 'stock_forecasts.py').read())

# portfolio sorts
# The inputs to this file are:
#    1- The monthly analysis dataset: stock/intermediate/ds_stock_monthly.csv 
# Its outputs are:
#    1- Table of portfolio sorts (table 10): stock/table/MW_decile_returns.tex 
#                                                              and stock/tables/KT_conservative_decile_returns.tex
exec(open(code_dir + 'decile_returns.py').read())








