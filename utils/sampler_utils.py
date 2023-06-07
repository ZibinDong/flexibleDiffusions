import numpy as np


def sample_t_array_uniform(T, N):
    ''' Sample t array uniformly '''
    return np.asarray(list(range(0, int(T), T//N)))
    
def sample_t_array_quad(T, N, coeff = 0.8):
    ''' Sample t array quadratically '''
    return ((np.linspace(0, np.sqrt(T * coeff), N)) ** 2).astype(int)