import random
import numpy as np

def generate_data(m=10000, num_ranks=5, noise=0.2):
    
    # generate data
    
    num_walls = num_ranks-1
    w_opt = np.array([1.0,2])
    w_opt = w_opt / np.linalg.norm(w_opt)
    X = np.random.rand(m,2)  
    # convert to numbers in [-1,1]
    X = 2.0*X - 1.0
    
    prediction_scores = X.dot( w_opt.transpose())
    min_val = np.min(prediction_scores)
    max_val = np.max(prediction_scores)
    sorted_scores = np.sort(prediction_scores)
    
    walls = []
    for i in range(num_walls):
        wall_index = random.randrange(m)
        walls.append( sorted_scores[wall_index] )
    walls.sort()
  
    Y = np.zeros((m,1), dtype=int)
    for i in range(0, num_ranks -1):
        Y[ prediction_scores > walls[i] ] = i+1

    count_how_many_in_each_class(Y, num_ranks)
    possible_ranks = range(0,num_ranks)

    X = X + np.random.rand(m,2)*noise  
    return (X,Y, w_opt, possible_ranks)



def count_how_many_in_each_class(Y, num_ranks):
    how_many = np.zeros( (num_ranks, 1) )
    for i in range(0, num_ranks):
        how_many[i] = (Y == i).sum() 
    print(how_many)
