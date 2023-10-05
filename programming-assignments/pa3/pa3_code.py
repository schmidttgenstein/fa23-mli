import math
import numpy as np 
import pandas as pd 
import random as rnd
from numpy import linalg as la
import matplotlib.pyplot as plt 
##Do not import additional libraries for this programming assignment 

class HilbertProj:
    def __init__(self,lim = 10,fun = np.sin, deg = 5, n_data = 150,mu_data = 0, var_data =3,noise_var = .25,N=10000,plotting = False):
        self.lim = lim
        self.deg = deg 
        self.fun = fun 
        self.n_data = n_data
        self.mu_data = mu_data 
        self.var_data = var_data 
        self.t = np.linspace(-lim,lim,N)
        self.y = fun(self.t)
        self.coeff = None
        self.plotting = plotting
        self.noise_var = noise_var
        self.xdat = None 
        self.ydat = None


    def update_xsig(self,sig):
        self.mu_data = sig
    
    def update_deg(self,n):
        self.deg = n
        self.coeff = None

    def gen_data(self, n = None, noise_var = 0,sig_data = None):
        if n is None:
            n = self.n_data 
        if sig_data is None:
            sig_data = self.var_data
        x_dat = np.random.normal(sig_data,self.var_data,n)
        x_dat.sort()
        y_dat = self.fun(x_dat)+ np.random.normal(0,noise_var,n)
        return x_dat, y_dat 


    def update_coeff(self,coeff,):
        self.coeff = coeff 

    def poly(self,t,coeff=None,):
        ### this method evaluates p(t) = a_0 + a_1 * t + ... + a_n t^n
        ## coeff should be array np.array([a_0,...,a_n])
        # and you should be able to return the result as an array, e.g. if t is an np.linspace
        poft = None
        return poft 


    def exact_poly(self,t,n = None,xdat=None,ydat=None):
        ### given xdata, ydata, you should return a polynomial that passes exactly through 
        ## n points (i.e. poly(x(i)) = y(i) for i = 1,...,n)
        if xdat is None:
            if self.xdat is None:
                self.xdat,self.ydat = self.gen_data()
            xdat,ydat = self.xdat,self.ydat
        if n is not None:
            idx = rnd.sample(range(xdat.shape[0]),n)
            xdat = xdat[idx]
            ydat = ydat[idx]
        return None

    def get_exact_coeff(self,x_data,y_data,deg):
        ### This method may be used to find coefficients a_0,...,a_n
        ## for exact polynomial defined in exact_poly method
        # you may need to call exact_poly 
        t = np.linspace(-1,1,deg)
        a = t.reshape([deg,1]) ** np.arange(deg)
        b = self.exact_poly(t,deg,x_data,y_data)
        coeff = self.matrix_inv(a,b)
        return coeff  

    def get_taylor_coeff(self,deg):
        ### return taylor coefficients for sin
        coeff = None
        return np.array(coeff)



    def solve_hp(self,x_data,y_data,deg,a=None,b=None):
        if x_data is not None:
            a,b = self.setup_hp(x_data,y_data,deg)
        coeff = None  #Set coefficients
        return coeff 

    def setup_hp(self,x_data,y_data,n):
        ### Use this method to set up your Hilbert Projection 
        ## per ws problem which expresses as linear (matrix) equation
        # a x = b. This should return a and b
        pass 
        #return covx,covxy  


    def matrix_inv(self,a,b):
        ## you don't need to define this method but it may be convenient
        # use to solve a * x = b, return x satisfying eq.
        pass 
        #return x 

    def base_data(self,xdata,ydata,xdata_test,ydata_test,deg):
        #### you will fit model using xdata, ydata and evaluate using xdata_test, ydata_test
        ### this will compute hilbert projection onto degree deg polynomials, and return bias / variance 
        ## as defined by E((y^* - y^*_H)^2) (bias) and E((yhat_S - y^*_H)^2) (variance)
        # you should observe that bias is identical across the board (why?)

        taycof = self.get_taylor_coeff(deg)
        self.taylor_coeff = taycof
        xsamp, ysamp = self.gen_data(deg)
        excof = self.get_exact_coeff(xsamp,ysamp,deg)
        self.exact_coeff = excof
        hpcof = self.solve_hp(xdata,ydata,deg)
        self.hilbert_coeff = hpcof 
        hstcof =self.hilbert_coeffs[deg-2] ## recall that self.hilbert_coeffs are our standin for y^*_H
        yhst = self.poly(xdata_test,hstcof)
        yex = self.poly(xdata_test,excof)
        yhp = self.poly(xdata_test,hpcof)
        bias = None #self.calc_mse(,)
        exvar = None #self.calc_mse(,)
        hpvar = None #self.calc_mse(,)
        extot = None #self.calc_mse(,)
        hptot = None #self.calc_mse(,)
        ex_results = {'ypred': yex, 'bias':bias,'var':exvar,'tot':extot}
        hp_results = {'ypred': yhp, 'bias':bias,'var':hpvar,'tot':hptot}
        tay_results = {'ypred':ytay, 'bias':bias,'var':0}
        return hp_results, ex_results, tay_results

    def set_coeffs(self,m = 2500000,sig = None,n=30):
        ##### This method you may use for finding noiseless Hilbert Coefficients. 
        #### This is a crude way of approximating--and it very much is an approximation!--
        ### the projection in H. But this approximation should be better than the same obtained on 
        ## noisy smaller data set. Unfortunately you can't compute the projection directly, so this is the best we've got
        # The result from this should appear in your bias plots
        xsamp, ysamp = self.gen_data(m,self.noise_var,sig_data = sig)
        a,b = self.setup_hp(xsamp,ysamp,n)
        h_arr = []
        t_arr = []
        for deg in np.arange(2,n): 
            print(deg)
            taycof = self.get_taylor_coeff(deg)
            t_arr.append(taycof)
            adeg = a[:deg+1,:deg+1]
            bdeg = b[:deg+1]
            hpcof = self.solve_hp(x_data = None,y_data = None, deg=deg,a=adeg,b=bdeg)
            h_arr.append(hpcof)
        self.hilbert_coeffs = h_arr
        self.taylor_coeffs = t_arr 
    

    def calc_mse(self,y_true,y_approx):
        ## returns mse E((y_true - y_approx)^2)
        return None 


    

    def run_sim(self,m = None,drange = 21,noise = 0,plotting = False):
        if m is None:
            m = self.n_data
        xdata,ydata = self.gen_data(m,noise)
        xd_test,yd_test = self.gen_data(m,noise)
        deg_range = np.arange(2,drange)
        results_data  = np.zeros([deg_range.shape[0],7])
        predictions = np.zeros([ydata.shape[0],20])
        j = 0 
        for d in deg_range:
            print(d)
            h,e,t = self.base_data(xdata,ydata,xd_test,yd_test,d)
            row_data = np.array([h['bias'],h['var'],h['tot'], e['bias'],e['var'], e['tot'], t['bias']])
            results_data[j,:] = row_data
            j+=1 
            if plotting:
                plt.plot(xdata,h["ypred"],'-.',label = 'hp')
                plt.plot(xdata,e['ypred'],'--',label = 'exact')
                plt.plot(xdata,t["ypred"],':',label = 'Taylor')
        if plotting:
                plt.plot(xdata,ydata,'k--',label = 'gt')
                plt.legend()
                plt.show()
        return deg_range, results_data

    def plot_sim_data(self,dr,rd):
       pass 



if __name__ == "__main__":
    hp = HilbertProj(n_data = 1000,noise_var = 0)
    plt.close('all')
    hp.set_coeffs()
   
    m = 150
    ddat, rdat = hp.run_sim(m,drange =21,noise =0.25,plotting = False)
    hp.plot_sim_data(ddat,rdat)
  