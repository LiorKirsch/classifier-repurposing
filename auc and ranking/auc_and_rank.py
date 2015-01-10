'''
Created on Jan 4, 2015

@author: Lior Kirsch
'''

import numpy as np
from __builtin__ import int
from numpy.random.mtrand import random_integers

def createFeaturesForAUC(X,Y, max_samples=None):
    ''' 
        X is the data [samples X features]
        Y are the labels (1 , -1)
        use max samples to control the maximum number of samples that are generated for the AUC
        if max_samples is not specified or it is left None the default is to use all combination
        of a positive sample and a negative samples
        This function creates new feature vector of the sort x_pos - x_neg with a label of y=1.
    '''
    
    (n, d) = X.shape
    Y = Y.flatten()
    
    positive_samples_mask = Y>0
    negative_samples_mask = np.logical_not(positive_samples_mask)
    
    num_positives = positive_samples_mask.sum()
    num_negatives = negative_samples_mask.sum()
    
    X_pos = X[Y>0,:]
    X_neg = X[negative_samples_mask,:]

    all_combinations = cartesian( [xrange(num_positives), xrange(num_negatives)] )
    if max_samples is None:
        max_samples = num_positives * num_negatives
    else:
        rand_int = np.random.randint(num_positives * num_negatives, size=max_samples)#  random_integers
        all_combinations = all_combinations[rand_int,:]  
    
    pos_samples_ind = all_combinations[:,0].flatten()
    neg_samples_ind = all_combinations[:,1].flatten()
        
    X_new = X_pos[pos_samples_ind,:] - X_neg[neg_samples_ind,:]
    Y_new = np.ones((max_samples,1))
    
    return (X_new, Y_new)
            

def predict_score_for_auc(X, w):
    prediction_scores = X.dot( w.transpose())
    return prediction_scores

def cartesian(arrays, out=None):
    """
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out