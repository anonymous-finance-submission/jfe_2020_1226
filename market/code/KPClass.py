import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import chi2
import statsmodels.api as sm
import time
import multiprocessing as mp


class KP:
    '''
    Implement Kodde-Palm conditional tests
    
    Parameters
    ----------
    d : name of dataframe containing returns, bounds, and conditioning variables
    h : horizon of the returns
    L : length of lags for HAC estimator
    se: type of standard errors to use ('nw','hh','bootstrap')
    b : name of bound to use ('lb_m','lb_cylr','lb_m_orig'...)
    cv: list of conditioning variables to use (excluding constant)
    '''
    def  __init__(self,d,h,L,se,b,cv) :
        self.horizon = h
        self.lags = L
        self.se_type = se
        self.bound = b
        self.conditioning_vars = cv
         
        # dataset with moments of interest only
        df = sm.add_constant(d[cv].copy())
        df = df.multiply(d['f_mktrf'+str(h)] - d[b + '_'+str(h)], axis='index')        
        self.df = df.dropna()
        
        # sample means
        self.lambda_bar = self.df.mean()
        
        # number of time-series observations
        self.T = self.df.count()['const']
        
        # number of moments
        self.num_moments = len(self.lambda_bar)  
        


    # Estimate covariance matrix of the vector sample mean
    def Sigma(self):
        if self.se_type == 'nw':
            sigma = self.neweywest()
        elif self.se_type == 'hh':
            sigma = self.hansenhodrick()
        # print(sigma)
        return sigma

    
    # Inverse of Sigma
    def Sigma_inv(self):
        return np.linalg.inv(self.Sigma())
    
    
    # Estimate moment cross-products at particular lag length (start)
    # (Hamilton Time-Series equation 10.5.10)
    def Gamma(self, start):
        df_demean = self.df - self.lambda_bar
        df_demean_np = df_demean.to_numpy()
        summand = np.zeros((self.T-start,self.num_moments,self.num_moments))
        for t in np.arange(start, self.T):
            summand[t-start] = np.outer(df_demean_np[t], df_demean_np[t-start])       
        return (1/self.T)*summand.sum(axis=0)


    # Newey-West HAC estimator
    def neweywest(self):
        S_tilde = self.Gamma(0)
        for v in np.arange(1,self.lags+1):
            # print(v)
            S_tilde = S_tilde + (1 - v/(self.lags+1)) * (self.Gamma(v) + self.Gamma(v).T)
        return (1/self.T)*S_tilde             


    # Hansen-Hodrick HAC estimator
    def hansenhodrick(self):
        S_tilde = self.Gamma(0)
        for v in np.arange(1,self.lags+1):
            # print(v)
            S_tilde = S_tilde + (1) * (self.Gamma(v) + self.Gamma(v).T)
        return (1/self.T)*S_tilde        
    
    
    ############## Kodde-Palm Tests
    def kp_estimate(self, lambda_bar, Sigma, tol = 1.0e-8, max_iter = 10000):
        '''
        To calculate Kodde and Palm (1986) test-statistic, we solve the 
        following quadratic programming problem:
            lambda_hat = argmin (lambda_bar - lambda)' Sigma^-1 (lambda_bar - lambda)
                        s.t. lambda >= 0.
        Then, we can calculate the test-statistics as follows:
            For Wald test:  D0 = lambda_bar' * Sigma^-1 * lambda_bar 
            For validity:   D1 = (lambda_bar - lambda_hat)' Sigma^-1 (lambda_bar - lambda_hat)
            For tightness:  D2 = lambda_hat' Sigma^-1 lambda_hat
        '''
       
        # initial point estimates; e.g. from GMM
        lambda_bar = pd.Series(lambda_bar) 
         
        # covariance matrix of lambda_bar
        Sigma = pd.DataFrame(Sigma) 
        
        # inverse of Sigma
        Sigma_inv = pd.DataFrame(np.linalg.inv(Sigma)) 
        
        
        # If sample moments are non-negative, quadratic programming problem is trivial.
        if min(lambda_bar) >= 0.0:
            lambda_hat = lambda_bar
            exit_flag = 2
        else:
            # setting up the quadratic programming problem in gurobi
            # define the model
            m = gp.Model('kodde_palm')
            
            # add the variables
            lam = pd.Series(m.addVars(range(self.num_moments), lb = 0.0))
            
            # update the model
            m.update()
            
            # set up the minimization goal
            goal = (lambda_bar - lam).T @ Sigma_inv @ (lambda_bar - lam)
            m.setObjective(goal, GRB.MINIMIZE)
            
            # set a few parameters
            m.setParam('OutputFlag', 0)
            m.setParam('FeasibilityTol', tol)
            m.setParam('OptimalityTol', tol)
            m.setParam('IterationLimit', max_iter)
            # m.setParam('NonConvex', 2)
            
            # run the model
            m.optimize()     
            
            # check if optimization has been successful
            if m.Status != 2:
                print('Optimization might have failed.')
            exit_flag = m.Status
        
            # extract lambda_hat
            lambda_hat = pd.Series([v.x for v in lam])

        # calculate test-statistics
        wald = lambda_bar.T @ Sigma_inv @ lambda_bar
        validity = (lambda_bar - lambda_hat).T @ Sigma_inv @ (lambda_bar - lambda_hat)
        tightness = lambda_hat.T @ Sigma_inv @ lambda_hat
        
        return lambda_hat.values, wald, validity, tightness, exit_flag    

    
    # Output the Kodde-Palm stats for the empirical moment mean and covariance matrix
    def kp_output(self):
        return self.kp_estimate(self.lambda_bar.to_numpy(), self.Sigma())
    
    
    def p_values(self, wald, validity, tightness, Sigma, tol = 1.0e-8, max_iter = 10000, num_sim = 100000):
        '''
        To calculate p-values for Kodde and Palm (1986) test-statistics, we run
        separate Monte Carlo simulations for each case to find the distribution
        in equation (2.14) of Kodde and Palm (1986).
        '''
        
        # generate a sample of multivariate normal distribution
        x = np.random.multivariate_normal(np.zeros((self.num_moments,)), Sigma, num_sim)
        
        # initiate a vector of estimates
        x_hat = np.zeros((num_sim, self.num_moments)) 
        
        # solve the estimation problem for each draw in the sample
        for i in range(num_sim):
            x_hat[i,:], t1, t2, t3, exit_flag = self.kp_estimate(x[i,:], Sigma, tol, max_iter)
            if exit_flag != 2:
                print('Warning: Optimization failed: exit flag: ' + str(exit_flag) + ' iteration: ' + str(i))

        # pool = mp.Pool(50)            
        # results = [pool.apply_async(self.kp_estimate, args=(x[i,:], Sigma, tol, max_iter)) for i in range(num_sim)]
        # pool.close()
        # pool.join()
        # result_list = [i.get() for i in results]
        # xhat= [i[0] for i in result_list]
        # return x_hat
               
        
        # calculate the vector of weights as in Kodde and Palm (1986)
        pos_x = 1.0*(x_hat >= tol)
        sum_x = pos_x.sum(axis = 1)
        w, bin_edges = np.histogram(sum_x, bins = range(self.num_moments + 2))
        w = w/num_sim
        
        # calculate the distribution
        prob_validity = np.zeros((self.num_moments + 1,))
        prob_tightness = np.zeros((self.num_moments + 1,))
        
        # now let's calculate the rest of chi2 distributions
        for i in range(0, self.num_moments + 1):
            prob_validity[i] = 1.0 - chi2.cdf(validity, self.num_moments - i)
            prob_tightness[i] = 1.0 - chi2.cdf(tightness, i)
        
        # fix the chi2 with zeros DoF.
        if validity <= tol:
            prob_validity[self.num_moments] = 1.0
        else:
            prob_validity[self.num_moments] = 0.0
            
        if tightness <= tol:
            prob_tightness[0] = 1.0
        else:
            prob_tightness[0] = 0.0
            
        # calculate p-values
        pVal_validity  = sum(prob_validity*w)*100.0
        pVal_tightness = sum(prob_tightness*w)*100.0
        pVal_wald      = (1.0 - chi2.cdf(wald, self.num_moments))*100.0
        
        return pVal_wald, pVal_validity, pVal_tightness      
     
  
        
     

    
                