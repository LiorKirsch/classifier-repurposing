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
import close_set_rank
        
if __name__ == '__main__':
    ''' 
        1. Create a dataset
        2. Center the data to zero mean and unit variance
        3. Transform the features
        4. Apply a classifier
        5. Check that the walls are sorted
        6. Get the prediction
    '''
    
    num_ranks=5
    (X,Y, w_opt, possible_ranks) = generate_data(num_ranks=num_ranks) 
    X = preprocessing.scale(X, with_mean=False) # unit variance
    
    subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], c=Y, s=60)
    plt.gray()
    plt.title('Original data')
        
    (X_new, Y_new) = close_set_rank.createFeaturesForOrdinalRepurpose(X,Y, possible_ranks, samples='minimum')
    
    clf = LinearSVC(fit_intercept=False)
    clf.fit(X_new, Y_new) 
    w = clf.coef_
    
    close_set_rank.check_that_walls_are_ordered(w, possible_ranks)
    
    predictions = close_set_rank.predict_rank(X, w, possible_ranks)
    
    subplot(1,2,2)
    plt.scatter(X[:,0], X[:,1], c=predictions.transpose(), s=60)
    plt.gray()
    plt.title('Predictions')
    plt.show()
    
    