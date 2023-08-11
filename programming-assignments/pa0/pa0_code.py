import numpy as np 
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



class FirstClassModel:
    # Every method (except for performance metrics) should be defined 
    # a one line call to the appropriate method from the library you use
    def __init__(self):
        self.model = LogisticRegression(random_state = 0)

    def fit(self,x,y):
        ## Your fit model may simply call that of sklearn or any other
        # library you wish to use (i.e. should be one line)
        pass 

    def predict_class(self,x):
        ### This method must return an array with size y
        ## whose entries are either 0 or 1#
        y_pred = None
        return y_pred
         

    def predict_probabilities(self,x):
        ### This method must return an array with size y
        ## whose entries denote the "likelihood" that input
        # x belongs to (each) class, with shape (#data points, #classes)
        y_prob = None 
        return y_prob

    def performance_metrics(self,x,y):  
        ### report accuracy, true positive rate, false positive rate
        ## as well as score mean and standard deviation (corresponding to class ONE)
        # ... 
        return None #accuracy, tpr, fpr, yprob_mean, yprob_stdev


if __name__ == "__main__":
    # Step -1: instantiate a model. I've already biased you with 
    # one in the constructor
    fcm = FirstClassModel()

    ### Step 0: load data 
    data =  pd.read_csv('./data/pa0_train.csv',index_col = False)  #load_data_function  
            #pd.read_csv('filepath')  ## You will want to include extension '.csv'

    ### Step 1: process data, extract features / labels. You do not need to do 
    ## a train/test split (but you certainly can if you'd like!)
    x,y =  np.array(data.iloc[:,:-1]), np.array(data.iloc[:,-1])
            # separate input data from labels and convert to numpy arrays

    ### Step 2:  train model 
    ## will need to define .fit() method of FirstClassModel
    # Calling this method will fit parameters of your model
    fcm.fit(x,y)

    
    ### Step 3: define prediction methods
    ## one should give the probability associated to class 1, and
    # the other the discrete class predictions
    y_probs = fcm.predict_probabilities(x)
    y_pred = fcm.predict_class(x)

    # printing for debugging purposes, not needed 
    print(y_probs)

    ### Step 4: define metrics method
    ## evaluate performance
    metrics = fcm.performance_metrics(x,y)
