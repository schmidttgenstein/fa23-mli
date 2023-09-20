import time 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import norm as gauss

class BinClassification:
    def __init__(self,g = None,alpha=None):
        ##fix exponential distro params g0,g1 + marginals alphaj := P_Y(y=j)
        if g is None:
            g_pre = np.random.random(2)
            g_pre.sort()
            g = g_pre[::-1]
            self.g0 = g[0]
            self.g1 = g[1]
        else:
            self.g0 = g[0] 
            self.g1 = g[1]
        if alpha is None:
            alpha = np.random.random()
        self.alpha0 = alpha 
        self.alpha1 = 1-alpha
        self.classes = {0:g[0],1:g[1]}

    def gen_data(self,n = 10000, y_class = 0):
        #Step 0: Generate data from exponential distribution
        #your exp param should depend on class variable y
        return None
    
    def e_cdf(self,data):
        #step 1
        #should have e_cdf from pa1; remember it's a linspace!
        return None
   
    def model_cdf(self,data,g):
        #step 1
        #supposing that data comes from some *exp* distro, 
        #define its cdf
        return None 
    
    def model_pdf(self,t,g):
        #and its pdf 
        return None
    
    def model_error(self,data,g):
        ### step 1
        ## define error between your model_cdf
        # and true (as approximated by e_cdf)
        return None 
    
    def find_g(self,data,g_max = 10,N = 1000):
        ##step 2: define method to find parameter g
        #best representing your data 
        return None 
    
    def xpr(self,data,thresh,g = None):
        ## define tpr/fpr. You should use g to distinguish t vs. f
        # x = f / t (for false positive rate / true positive rate)
        return None 
    
    def pr_density(self,t,prec = False):
        ## define precision / P(y=1|x) 
        # (collapsing into one method because they are similar)
        return None

    def objective_fun(self,t,lamb):
        #this should compute lamb * tpr + (1-lamb)*fpr
        return None
    
    def d_obj(self,t,lamb):
        #d/dt (objective_func(t))
        return None

    def y_tilde(self,t,a,b):
        return None 
    
    def score_error(self,t,a,b):
        return None

    def find_ab(self,t,a_max = 15,b_max = 10,N=100):
        ## parameters a,b for y_tilde, which is a sigmoidal 
        # representing P(y=1|x); again, no sophisticated optimization!
        a,b = None, None
        return a,b 

if __name__ == "__main__":
    bc = BinClassification([5,.25],.95)
    #step 0: define method to generate exp data, generate some, and view
    d = bc.gen_data()
    plt.figure(1)
    d0 = bc.gen_data(50000)
    d1 = bc.gen_data(25000,1)
    plt.hist(d0,bins = 50,density = True, alpha = .5,label = 'class 0')
    plt.hist(d1,bins = 50,density = True, alpha = .5,label = 'class 1')
    plt.xlabel('t')
    plt.ylabel('density')
    plt.legend()
    #step 1: define e_cdf, model_cdf, and error between them
    plt.figure(2)
    cdf0 = bc.e_cdf(d0)
    cdf1 = bc.e_cdf(d1)
    plt.xlabel('t')
    plt.ylabel('P(x<t)')
    plt.plot(d0,cdf0,label = 'class 0')

    ##step 2: define method to find class parameters. 
    #You shouldn't need/use sophisticated optimization here
    g0_app = bc.find_g(d0)
    g1_app = bc.find_g(d1)
    plt.plot(d0,1-np.exp(-g0_app*d0),label = 'class 0 model')
    plt.plot(d1,cdf1,label = 'class 1')
    plt.plot(d1,1-np.exp(-g1_app*d1),label = 'class 1 model')
    plt.legend()

    plt.figure(6)
    plt.plot(d0,cdf0-1+np.exp(-g0_app*d0),label = 'model 0 error')
    plt.plot(d1,cdf1-1+np.exp(-g1_app*d1),label = 'model 1 error')
    plt.xlabel('t')
    plt.ylabel('cdf error')


    ##step 3: define metrics, tpr, fpr, precision, and P(y=1|x)
    plt.figure(3)
    t = np.linspace(0,20,15000)
    tpr = bc.xpr(None,t,g1_app)
    fpr = bc.xpr(None,t,g0_app)
    prec = bc.pr_density(t,prec = True)
    p1 = bc.pr_density(t)
    plt.plot(t,tpr,label = 'tpr')
    plt.plot(t,fpr,label = 'fpr')
    plt.plot(t,prec,label = 'precision')
    plt.xlabel('t')
    plt.ylabel('metric')
    ##step 4: find optimal threshold for joint objective lambda tpr + (1-lambda) prec, 
    ##4a: by taking absolute max of the objective function 
    #4b: by finding the correct zero of its derivative 
    lamb = .5
    obj = bc.objective_fun(t,lamb)
    idx = np.argmax(obj)
    t_opt = t[idx]
    plt.plot(t,obj,label='objective')
    plt.plot(t_opt,obj[idx],'r*',label = 'optimal point')
    plt.title(f"optimal thresh at {t_opt:.5f}")
    plt.legend()

    plt.figure(4)
    dodt = bc.d_obj(t,lamb)  
    sort_idx = np.argsort(np.abs(y))
    opt_idx = sort_idx[np.argmax(obj[sort_idx[:10]])]
    n = 1500
    plt.plot(t[:n],dodt[:n],label = 'derivative of objective')
    #plt.plot(t[:n],y1[:n],label = 'first order Taylor')
    #plt.plot(t[:n],y2[:n], label = 'second order Taylor')
    plt.plot(t[opt_idx],dodt[opt_idx],'*',label = 'zero')
    plt.legend()
    plt.title(f"opt thresh at {t[opt_idx]:.5f}")

    ##step 5: plot P(y=1|x) and model y_tilde for P(y=1|x)
    # find parameters a,b for which sigma_a,b(t) matches P(y=1|x)
    t = np.linspace(0,5,1000)
    py1gx = bc.pr_density(t,prec = False)
    a,b = bc.find_ab(t)
    y_tilde = bc.y_tilde(t,a,b)
    plt.figure(5)
    plt.plot(t,py1gx,label = 'P(y=1|x)')
    plt.plot(t,y_tilde,label = 'model score')
    plt.xlabel('t')
    plt.ylabel('y_tilde')
    plt.legend()

    plt.figure(7)
    plt.plot(t,py1gx-y_tilde,label = 'model error')
    plt.xlabel('t')
    plt.ylabel('P(y=1|x) model error')
    plt.show()