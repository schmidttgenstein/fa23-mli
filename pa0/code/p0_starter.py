import numpy as np 
import pandas as pd 
import sklearn as skl 
import matplotlib.pyplot as plt 


class FirstClassModel:
    def __init__(self):
        pass 


    def fit(self,x,y):
        ## Your fit model may simply call that of sklearn or any other
        # library you wish to use 
        pass 

    def predict_class(self,x):
        ## This method must return an array with size y
        # whose entries are either 0 or 1
        pass 

    def predict_probabilities(self,x):
        ### This method must return an array with size y
        ## whose entries denote the "likelihood" that input
        # x belongs to class y=1
        pass



    



if __name__ == "__main__":
    fcm = FirstClassModel()

    ### load data 
    data =  #load_data_function  
            #pd.read_csv('filepath')  ## You will want to include extension '.csv'

    ### process data
    x,y = #process_data_function(data)
            # essentially separate input data from labels and convert to numpy arrays
            # can filter (pd) dataframe data with e.g. df.iloc[:,0:col_num]
            # or df.iloc[:,col_num]

    ### train model 
    ## Calling this method will fit parameters of your model
    fcm.fit(x,y)


    ### return predictions
    #class prediction
    y_pred = fcm.predict_class(x)
    y_probs = fcm.predict_probabilities(x)


    ###evaluate performance
    metrics = fcm.performance_metrics(x,y)
