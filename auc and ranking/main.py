'''
Created on Oct 29, 2014

@author: Lior Kirsch
'''

import numpy as np
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from pylab import subplot 
from dataGeneration import generate_data
from __builtin__ import int
import auc_and_rank
        
if __name__ == '__main__':
    ''' 
        1. Create a dataset
        2. Center the data to zero mean and unit variance
        3. Transform the features (Xnew = Xpos - Xneg)
        4. Apply a classifier
        5. Get the prediction
    '''
    
    max_pair_samples=100000
    num_ranks=5
    (X,Y, w_opt, possible_ranks) = generate_data(num_ranks=num_ranks) 
    X = preprocessing.scale(X, with_mean=False) # unit variance
    
    subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], c=Y, s=60)
    plt.gray()
    plt.title('Original data')
        
    (X_new, Y_new) = auc_and_rank.createFeaturesForAUC(X,Y, max_samples=max_pair_samples)
    #=================== 
    # since scipy does not like that all elements in the classifier are all positive we shall
    # make a workaround, we randomly flip half the samples and also their labels
    #===================
    [n_new, d] = X_new.shape
    rand_int = np.random.randint(n_new, size=np.round(n_new/2))
    X_new[rand_int,:] = -1 * X_new[rand_int,:]
    Y_new[rand_int,:] = -1 * Y_new[rand_int,:]

    clf = LinearSVC(fit_intercept=False)
    clf.fit(X_new, Y_new) 
    w = clf.coef_
    
    predictions = auc_and_rank.predict_score_for_auc(X, w)
    
    subplot(1,2,2)
    plt.scatter(X[:,0], X[:,1], c=predictions.transpose(), s=60)
    plt.gray()
    plt.title('Predictions')
    plt.show()
    
    