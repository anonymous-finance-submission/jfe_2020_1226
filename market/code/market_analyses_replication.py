"""
This file replicates all the tables and figures related to the analysis of 
market bounds in:  "Validity, Tightness, and 
Forecasting Power of Risk Premium Bounds".

Market bounds are those of Martin (2017) and Chabi-Yo and Loudis (2020).
"""
#%% 0  ========================================================================
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
code_dir = PATH_TO_REPLICATION_PACKAGE + 'market/code/'

# Second, toggle each variable to True if you want the corresponding step to run. 
# Each step overrides some of the files in the package. If you choose not to 
# run a step, you will be using the existing files.

CALCULATE_MARKET_BOUNDS_FROM_SCRATCH       = False  
CREATE_ANALYSIS_DATASETS_FROM_SCRATCH      = False
RUN_KODDE_PALM_SIMULATION                  = False
RUN_FULL_SAMPLE_REGS_SIMULATION            = False
RUN_OOS_SIMULATION                         = False

# Third, select the number of cores available for parallel processing on your computer

NUMBER_OF_CORES                            = 50

#%% 1 =========================================================================
# Step 1: Calculate market bounds from CBOE files and OptionMetrics.
# If you wish to skip this step and use the bounds we have already 
# calculated, change the variable calculate_market_bounds_from_scratch
# to False. 
# 
# To run this step, you need:
#    - A connection to WRDS. The program will prompt you to enter your WRDS 
#      username and password.
#    - The following Python packages: pandas_datareader, wrds
#
# Approximate run time for this step is 15 minutes.
# =============================================================================

# The inputs to this file are:
#    1- options data from CBOE: market/input/market data/CBOE_yyyy.csv
#    2- original Martin (2017) bounds: market/input/epbounds.dta
#    3- original Chabi-Yo/Loudis (2020) bounds: market/input/cyl.xlsx
# Its outputs are:
#    1- All market bounds (our replications + originals): market/intermediate/market_bounds.csv
 
if CALCULATE_MARKET_BOUNDS_FROM_SCRATCH:
    exec(open(code_dir + 'market_bound_calculation.py').read())


#%% 2 =========================================================================
# Step 2: Put together data from several files for analysis.
# If you wish to skip this step and use the existing dataset toggle the variab-
# le create_analysis_dataset to False.
# 
# To run this step, you need:
#    - A connection to WRDS. The program will prompt you to enter your WRDS 
#      username and password.
#
# Approximate run time for this step is 20 seconds.
# =============================================================================

# The inputs to this file are:
#    1- S&P500 returns from Factset: market/input/sp500rets_1990_2021_factset.csv
#    2- Market bounds from step 1: market/intermediate/market_bounds.csv
#    3- Goyal/Welch predictor variables: market/input/PredictorData2020.xlsx
# Its outputs are:
#    1- Monthly analysis datasets: market/intermediate/ds_mkt_monthly.csv
#    2- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv

if CREATE_ANALYSIS_DATASETS_FROM_SCRATCH:
    exec(open(code_dir + 'create_analysis_datasets.py').read())

#%% 3 =========================================================================
# Step 3: Generate summary statistics, time series figures, and bound vs. retu-
# rn figures.
# 
# This step replicates:
#     - Table 1
#     - Figures 1, 2, 3, IA.1
# 
# Approximate run time for this step is 5 seconds.
# =============================================================================

# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
# Its outputs are:
#    1- Summary statistics tables (table 1): market/tables/summary_stats_extendedH where H = 1,3,6 or 12
exec(open(code_dir + 'summary_tabs.py').read())

# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
# Its outputs are:
#    1- Time series of market bounds (figure 1): market/figures/market_bounds.pdf
#    2- Plot of CYL vs Martin (figure 2): market/figures/martin_cyl.pdf  
exec(open(code_dir + 'plot_bound_timeseries.py').read())

# The inputs to this file are:
#    1- Monthly analysis datasets: market/intermediate/ds_mkt_monthly.csv
# Its outputs are:
#    1- Firgures of relaized returns vs bounds (figures 3, IA.1): market/figures/market_bound_return_monthly.pdf
#           and market/figures/market_bound_return_monthly_cyl.pdf
exec(open(code_dir + 'plot_bound_return.py').read())



#%% 4 =========================================================================
# Step 4: Run bootstrap simulations of the Kodde/Palm tests. If you wish to skip
# this step and use the existing simulation results, toggle 
# run_kodde_palm_simulation to False.
# 
# This step replicates:
#     - Figure R.1
# 
# To run this step, you need:
#     - A connection to WRDS. The program will prompt you to enter your WRDS 
#       username and password.
#     - A valid license to Gurobi with its Python package installed and loaded.
#       For more information on Gurobi, look at https://www.gurobi.com/ .
#     - Multiple cores for parallel processing. 
#     - The following Python packages: arch, multiprocessing, wrds, gurobipy
# 
# This step uses parallel processing to speed up the simulations. We recommend 
# running this step on a server. 
#     
# Approximate run time for this step with 50 processors is 2.5 days. Using fewer
# processors will increase the run time proportionally.
# =============================================================================

# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
# Its outputs are:
#    1- Results of Monte Carlo simulations: market/intermediate/bootstrap_kp_results.csv
#    2- Plots of Kodde/Palm test power (figure R.1): market/figures/bootstrap_kp_power_10.pdf

if RUN_KODDE_PALM_SIMULATION:
    exec(open(code_dir + 'bootstrap_kp_tests.py').read())



#%% 5 =========================================================================
# Step 5: Run Kodde/Palm tests of validity and tightness for the bounds, avera-
# ge bounds, and bounds plus constants.
# 
# This step replicates:
#     - Tables 3, IA.1, R.2
#     
# To run this step, you need:
#     - A valid license to Gurobi with its Python package installed and loaded.
#       For more information on Gurobi, look at https://www.gurobi.com/ .
#     - Multiple cores for parallel processing. 
#     - The following Python packages: gurobipy
# 
# Approximate run time for this step is 35 minutes.
# =============================================================================

# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
#    2- Results of Monte Carlo simulations: market/intermediate/bootstrap_kp_results.csv
# Its outputs are:
#    1- Tables of Kodde/Palm tests for market bounds (table 3): market/tables/valid_tight_tests_*.tex
# Approximate run time is 15 minutes
exec(open(code_dir + 'validity_tightness_tests.py').read())


# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
#    2- Results of Monte Carlo simulations: market/intermediate/bootstrap_kp_results.csv
# Its outputs are:
#    1- Tables of Kodde/Palm tests for average market bounds (table IA.1): market/tables/valid_tight_tests_avgbound_*.tex
# Approximate run time is 15 minutes
exec(open(code_dir + 'validity_tightness_tests_sample_means.py').read())


# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
#    2- Results of Monte Carlo simulations: market/intermediate/monte_carlo_sim.csv
# Its outputs are:
#    1- The plot of Kodde/Palm tests for Martin and CYL bounds (figure R.2): market/figures/fig_bound_plus_constant_validity_pval_lb_*.pdf
# Approximate run time with multiple cores is 5 minutes (computes finite sample p-values only)
exec(open(code_dir + 'validity_tightness_tests_add_constants.py').read())



#%% 6 =========================================================================
# Step 6: Run Monte-Carlo simulations of full-sample regressions. If you wish to 
# skip this step and use the existing simulation results, toggle 
# run_full_sample_simulation to False.
# 
# To run this step, you need:
#     - A connection to WRDS. The program will prompt you to enter your WRDS 
#       username and password.
#     - Multiple cores for parallel processing. 
#     - The following Python packages: arch, multiprocessing, wrds
# 
# This step uses parallel processing to speed up the simulations. We recommend 
# running this step on a server. 
#     
# Approximate run time for this step with 50 processors is 15 minutes. Using fe-
# wer processors will increase the run time proportionally.
# =============================================================================

# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
# Its outputs are:
#    1- Results of bootstrap simulations of full sample regressions: market/intermediate/bootstrap_fs_regs.csv
if RUN_FULL_SAMPLE_REGS_SIMULATION:
    exec(open(code_dir + 'bootstrap_full_sample_regressions.py').read())



#%% 7 =========================================================================
# Step 7: Run full sample regressions.
# 
# This step replicates:
#     - Tables 5, 6, IA.3
# 
# Approximate run time for this step is 5 seconds.
# =============================================================================

# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
#    2- Monthly analysis datasets: market/intermediate/ds_mkt_monthly.csv
#    3- The results of bootstrap full-sample simulations: market/intermediate/bootstrap_fs_regs.csv
# Its outputs are:
#    1- Tables of full sample regressions (tables 5,6,IA.1): market/tables/fs_reg_mkt_bnd_*.tex
exec(open(code_dir + 'full_sample_regression.py').read())



#%% 8 =========================================================================
# Step 8: Run OOS forecasts. 
# 
# This step replicates:
#     - Tables 8, IA.5, 
#     - Figures 9, 10, 11 
# 
# To run this step, you need:
#     - A connection to WRDS. The program will prompt you to enter your WRDS 
#       username and password.
#     - The following Python packages: wrds
# 
# Approximate run time for this step is 1 minute.
# =============================================================================

# The inputs to this file are:
#    1- Monthly analysis datasets: market/intermediate/ds_mkt_monthly.csv
# Its outputs are:
#    1- Various market forecasts using bounds: market/intermediate/market_forecasts.csv
exec(open(code_dir + 'market_forecasts.py').read())

# The inputs to this file are:
#    1- Various market forecasts using bounds: market/intermediate/market_forecasts.csv
# Its outputs are:
#    1- Tables of OOS tests (tables 8, IA.5): market/tables/oos_tests_DM_*.tex
exec(open(code_dir + 'market_diebold_mariano.py').read()) 

# The inputs to this file are:
#    1- Various market forecasts using bounds: market/intermediate/market_forecasts.csv
# Its outputs are:
#    1- Plot of cumulative SSE vs benchmark (figure 9): market/figures/market_cum_sse.pdf
exec(open(code_dir + 'plot_market_gw.py').read())

# The inputs to this file are:
#    1- Various market forecasts using bounds: market/intermediate/market_forecasts.csv
# Its outputs are:
#    1- Plot of excess returns vs forecasts (figure 10): market/figures/market_OLSforecast_return.pdf
exec(open(code_dir + 'plot_OLSforecast_return.py').read())

# The inputs to this file are:
#    1- Various market forecasts using bounds: market/intermediate/market_forecasts.csv
# Its outputs are:
#    1- Plot of cumulative SSE for bounds as forecasts (figure 11): market/figure/cum_sse_market.pdf
exec(open(code_dir + 'plot_delta_cum_sse.py').read())


#%% 9 =========================================================================
# Step 9: Run OOS forecasting simulations. If you wish to skip this step and use
# the existing simulation results, toggle run_oos_simulation to False.
#   
# This step replicates:
#     - Figures 12, 13
# 
# To run this step, you need:
#     - A connection to WRDS. The program will prompt you to enter your WRDS 
#       username and password.
#     - The following Python packages: multiprocessing, wrds
# 
# Approximate run time for this step is 1 hour.
# =============================================================================

# The inputs to this file are:
#    1- Daily analysis datasets: market/intermediate/ds_mkt_daily.csv
# Its outputs are:
#    1- Results of simulating OOS forecasts: market/intermediate/sim_oos.csv
#    1- Plot of OOS R2 vs length of time-series: Martin (figure 12): market/figures/sim_oosr2_by_ooshorizon_oosr2_martin.pdf
#    2- Plot of OOS R2 vs length of time-series: Martin (figure 13): market/figures/sim_oosr2_by_ooshorizon_pvalue_martin.pdf
#    3- Plot of OOS R2 vs length of time-series: CYL    (figure 14): market/figures/sim_oosr2_by_ooshorizon_oosr2_cyl.pdf
#    4- Plot of OOS R2 vs length of time-series: CYL    (figure 15): market/figures/sim_oosr2_by_ooshorizon_pvalue_cyl.pdf
if RUN_OOS_SIMULATION:
    exec(open(code_dir + 'simulate_oos_rsquared.py').read())
    

















