import numpy as np 
import numpy.linalg as la 
import matplotlib.pyplot as plt 
import scipy.special as sm

class MLModel:
    def __init__(self,epochs = 250,lr = 0.05):
        ###In this constructor we set number of epochs for training and learning rate
        ## The only method you will need to modify in this class is fit
        self.epochs = epochs
        self.lr = lr 

    def gen_data(self,):
        raise NotImplementedError

    def loss(self,):
        raise NotImplementedError

    def forward(self,):
        raise NotImplementedError

    def grad(self,):
        raise NotImplementedError 

    def update(self):
        raise NotImplementedError  

    def metrics(self,x,y):
        raise NotImplementedError     

    def fit(self,x_data,y_data,x_eval,y_eval, printing = False):
        ### This method implements our "1. forward 2. backward 3. update" paradigm
        ## it should call forward(), grad(), and update(), in that order. 
        # you should also call metrics so that you may see progress during training
        if printing:
            self.x_eval = x_eval 
            self.y_eval = y_eval
        for epoch in range(self.epochs):
            ## TODO (implement train step ) 
            if printing: 
                m = self.metrics(x_eval,y_eval)
                print(f"epoch {epoch} and train loss {loss.mean():.2f}, test metrics {m:.2f}, grad {grad}, grad norm {la.norm(grad):.2f}")

    
    def e2e(self,n_train = 100, n_test = 10000,printing = False,data = None):
        #end to end method generates data + trains model 
        if data is None:
            x_train, y_train = self.gen_data(n_train)
            x_test, y_test = self.gen_data(n_test)
            data = (x_train,y_train,x_test,y_test)
        else:
            x_train,y_train,x_test,y_test = data
        self.fit(x_train,y_train,x_test,y_test,printing)
        m = self.metrics(x_test,y_test)
        return m , data


class LinReg(MLModel):
    def __init__(self,fun = np.square, deg =1,n_data = 10000,lr = .05,x_sig = 1,epoch = None):
        if epoch is not None:
            super().__init__(epochs=epoch,lr=lr)
        else:
            super().__init__(lr=lr)
        self.x_sig = x_sig
        self.lr = lr
        self.n_data = n_data
        self.fun = fun
        self.degree = deg
        self.sf = np.ones(deg+1) #??? 
        self.set_hilberts()
        self.model_coeff = 5*np.random.random(self.hp_coeff.shape[0])

    def set_hilberts(self):
        x,y = self.gen_data()
        hp_coeff = self.solve_hp(x,y,self.degree)
        self.hp_coeff = hp_coeff 

    def gen_data(self, n = None, noise_var = .1):
        if n is None:
            n = self.n_data 
        x_dat = np.random.normal(0,self.x_sig,n)
        x_dat.sort()
        y_dat = self.fun(x_dat) + np.random.normal(0,noise_var,n)
        return x_dat, y_dat 

    def forward(self,x):
        # TODO: implement forward
        y_pre = None
        return y_pr

    def loss(self,y_approx,y_true):
        # TODO: implement loss
        return None

    def metrics(self,x,y):
        # TODO: Implement some metric for evaluating reg performance
        return None

    def grad(self,x,y):
        # TODO: implement the gradient
        return None

    def update(self,grad):
        # TODO: update model coeffs


    def poly(self,t,coeff=None,):
            deg = coeff.shape[0]
            if not isinstance(t,np.ndarray):
                t = np.array([t])
            nts = t.shape
            if len(nts) == 1:
                t = t.reshape([nts[0],1])
            tp = t ** np.arange(deg) 
            result = tp @ coeff 
            return result 

    def solve_hp(self,x_data,y_data,deg,a=None,b=None):
        if x_data is not None:
            a,b = self.setup_hp(x_data,y_data,deg)
        coeff = self.matrix_inv(a,b)
        return coeff 

    def setup_hp(self,x_data,y_data,n):
        covx_rows = np.zeros(2*n+1)
        covxy = np.zeros(n+1)
        covx = np.zeros([n+1,n+1])
        for j in range(2*n+1):
            tj = x_data**j
            if j < n+1:
                covxy[j] = ((tj) * y_data).mean()
            covx_rows[j] = tj.mean() 
        for j in range(n+1):
            covx[j,:] = covx_rows[j:j+n+1]
        return covx,covxy  

    def matrix_inv(self,a,b):
        ainv = la.inv(a)
        x = ainv @ b 
        return x 

class LogReg(MLModel):
    def __init__(self,p1 = .5,loss = 'nll',x_sig = 1,epochs=1000):
        super().__init__(epochs = epochs)
        self.model_coeff =  np.random.random(2)
        self.p1 = p1 
        self.p0 = 1-p1 
        self.x_sig = x_sig
        if loss == 'nll':
            self.grad = self.log_grad 
            self.loss = self.nll_loss 
        else:
            self.grad = self.sq_grad
            self.loss = self.mse_loss 

    def gen_data(self,n = 10000):
        n1 = int(self.p1 * n)
        n0 = int(self.p0 * n)
        x0 = np.random.normal(-2,self.x_sig,n0)
        x1 = np.random.normal(2,self.x_sig,n1)
        x0.sort()
        x1.sort()
        y0 = np.zeros(n0)
        y1 = np.ones(n1)
        x = np.concatenate([x0,x1])
        y = np.concatenate([y0,y1]).astype(int)
        return x, y

    def forward(self,x,alpha = None,):
        # TODO: implement forward for standard sigmoidal 
        ytild = None
        return ytild 


    def mse_loss(self,y_approx,y_true):
        # TODO: implement mse loss
        return None

    def nll_loss(self,y_approx,y_true):
        # TODO: implement nll loss
        return None

    def metrics(self,x,y):
        # TODO: implement method to evaluate accuracy
        acc = None
        return acc


    def base_grad(self,x,y,ytilde = None, alpha = None,):
        # TODO: implement one portion of chain rule, 
        # may call this method in the next two methods
        gr_base = None
        return gr_base 

    def sq_grad(self,x,y,alpha = None,):
        # TODO: implement gradient for mse loss
        gr = None
        return gr

    def log_grad(self,x,y,alpha = None,eps = 0.001):
        # TODO: implement gradient for nll loss
        gr = None
        return gr 

    def update(self,grad):
        # TODO: update model params

    


def testinglinreg(regr,n_tr,n_eval):
    m = regr.e2e(n_train = n_tr,n_test=n_eval,printing = True)
    print(m)
    plt.plot(regr.x_eval,regr.y_eval,'-.',label = 'data')
    t = np.linspace(-5,5,1000)
    plt.plot(t,regr.poly(t,regr.hp_coeff),label = 'hp')
    plt.plot(t,regr.forward(t),label = 'linreg')
    plt.plot(t,reg.fun(t),'-.',label = 'truth')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def testinglogreg(logregr1,logregr2,n_train,n_test):
    m1,d = logregr1.e2e(n_train,n_test,printing = True,)
    m2,_ = logregr2.e2e(printing = True,data = d)
    x_dat = np.linspace(logregr1.x_eval.min(),logregr1.x_eval.max(),500)
    ytild1 = logregr1.forward(x_dat)
    ytild2 = logregr2.forward(x_dat)
    x_eval = d[2]
    y_eval = d[3]
    x0 = x_eval[y_eval == 0]
    x1 = x_eval[y_eval == 1]
    m1 = logregr1.m 
    m2= logregr2.m
    thresh1 = x_dat[np.argmin(np.abs(ytild1-0.5))]
    thresh2 = x_dat[np.argmin(np.abs(ytild2-0.5))]
    y1 = np.ones(x_dat.shape[0])
    y2 = np.ones(x_dat.shape[0])
    y1[x_dat < thresh1] = 0 
    y2[x_dat < thresh2] = 0 

    plt.hist(x0,density = True,bins = 50,alpha = .5,label='y=0')
    plt.hist(x1,density = True,bins = 50,alpha = .5,label ='y=1')
    plt.plot(x_dat,ytild1,'-.',label = f"y_score w nll loss (accuracy {m1:.3f})")
    plt.plot(x_dat,ytild2,'-.',label = f"y_score w mse loss (accuracy {m2:.3f})")
    plt.plot(x_dat,y1,'-.',label = 'y_pred w nll loss')
    plt.plot(x_dat,y2,'-.',label = 'y_pred w mse loss')
    plt.legend()
    plt.show()
    return logregr1,logregr2

if __name__ == "__main__": 
    reg = LinReg(n_data = 1500,deg = 9,fun = np.sin,x_sig = 1.5,epoch=5000)
    testinglinreg(reg,500,1000)
    logreg1 = LogReg(loss = 'nll',p1 =.025,epochs = 500,x_sig=2)
    logreg2 = LogReg(loss = 'not nll',p1=.025,epochs = 500,x_sig=2)
    r1,r2 =  testinglogreg(logreg1,logreg2,5000,250000)
    



    
