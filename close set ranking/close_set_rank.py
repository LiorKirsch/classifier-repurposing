'''
Created on Oct 29, 2014

@author: Lior Kirsch
'''

import numpy as np
from __builtin__ import int

def createFeaturesForOrdinalRepurpose(X,Y, ranks, samples='minimum'):
    ''' 
        Using 'minimum' each sample is compared to its two closest walls.
        Using 'full' each sample is compared to all walls.
    '''
    
    (n, d) = X.shape
    num_ranks = len(ranks)
    num_walls = num_ranks -1
        
    X_new = []
    Y_new = []
    if samples == 'full':
        # for every sample create num_ranks samples
        
        for i in range(n):
            for j in range(num_walls):
                to_concat = np.zeros(num_walls)
                to_concat[j] = -1
                X_new.append( np.concatenate((X[i], to_concat), axis=0)  )
                
                if j < Y[i]:
                    Y_new.append(1)
                else:
                    Y_new.append(-1)
                
    else:
        for i in range(n):
            
            # This part creates a sample and the wall that is below that sample
            if 0 < Y[i]:
                below_wall = np.zeros(num_walls)
                below_wall[ Y[i] -1] = -1
                X_new.append( np.concatenate((X[i], below_wall), axis=0)  )
                Y_new.append(1)
            
            # This part creates a sample and the wall that is above that sample    
            if Y[i] < num_ranks - 1:    
                above_wall = np.zeros(num_walls)
                above_wall[ Y[i] ] = -1
                X_new.append( np.concatenate((X[i], above_wall), axis=0)  )
                Y_new.append(-1)
                
    return (X_new, Y_new)
            
            
def predict_rank(X, w, ranks):

    (n, d) = X.shape
    num_ranks = len(ranks)
    num_walls = num_ranks -1
    
    walls = w[:, -num_walls : ]
    w = w[:,: - num_walls]
    
    prediction_scores = X.dot( w.transpose())
    
    walls = walls.transpose()
    predictions = np.zeros((n,1), dtype=int)
    for i in range(0, num_walls):
        predictions[ prediction_scores > walls[i] ] = i+1
        
    
#     import scipy.stats as ss
#     prediction_with_walls = np.concatenate( ( prediction_scores,walls.transpose() ) )
#     ranks = ss.rankdata(prediction_with_walls) -1 # because ranks start at 1
#     just_wall_ranks = ranks[-num_walls:].astype(int)
#     without_wall_ranks = ranks[:-num_walls].astype(int)
#     mask = np.zeros(ranks.shape, dtype=bool)
#     mask[just_wall_ranks] = True
#     opposite_mask = np.logical_not(mask)
#     
#     z = np.cumsum(mask)
#     predictions = z[without_wall_ranks]
    
    return predictions
    
    
    
def check_that_walls_are_ordered(w, ranks):
    # check that wall_0 < wall_1 < ... < wall_n
    
    num_ranks = len(ranks)
    num_walls = num_ranks -1
    just_wall_ranks = w[:,-num_walls:]
    if not np.all( (sorted(just_wall_ranks) == just_wall_ranks) ):
        print('walls are not sorted might be better to use "full"')
            