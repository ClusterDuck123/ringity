import numpy as np
import scipy.stats as ss

from itertools import islice

def gap_ring_score(seq, base = None):
    """"Calculates gap ring score from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. However, there are 
    no checks to test if the sequence is non-negative."""
    
    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1
    
    seq_iter = iter(sorted(seq, reverse = True))
    p0 = next(seq_iter)
    p1 = next(seq_iter)
    
    return 1 - p1/p0    

def geometric_ring_score(seq, nb_pers = 2, base = 2, tol = 1e-10):
    """"Calculates geometric ring score from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative."""
    
    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1
        
    assert nb_pers >= 2
    
    if base is None:
        base = 2

    if nb_pers == np.inf:
        max_score = 1 / (base-1)
        # weights of later persistences are below tolerance level 
        nb_pers =  int(-np.log(tol) // np.log(base)) + 1
    else:
        max_score = (1 - base**(-nb_pers + 1)) / (base-1)
        
    seq_iter = iter(sorted(seq, reverse = True))
    p0 = next(seq_iter)
    noise_score = sum((pi / (p0 * base**i) for i, pi in enumerate(islice(seq_iter, nb_pers-1), 1)))
    score = 1 - noise_score/max_score
    return score
    
def linear_ring_score(seq, nb_pers = 2):
    """"Calculates linear ring score from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative.
    Extending the weights from 1/i to 1/(base*i) yields back the same score."""
    
    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1
        
    assert nb_pers >= 2
    
    if nb_pers == np.inf:
        raise Exception(f"Linear ring score for base == np.inf not defined!")
        
    max_score = sum(1/i for i in range(1, nb_pers))
        
    seq_iter = iter(sorted(seq, reverse = True))
    p0 = next(seq_iter)    
    noise_score = sum((pi / (p0*i) for i, pi in enumerate(islice(seq_iter, nb_pers-1), 1)))
    score = 1 - noise_score/max_score
    return score
    
def amplitude_ring_score(seq, nb_pers = 2):
    """"Calculates amplitude ring score from sequence of positive numbers.
    
    Score is linearly scaled to have the range [0,1]; i.e. the score is 
    defined as ``score = 1 - (N/(N-1)) * (1 - p0/(sum pi))``.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative."""
    
    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1
        
    assert nb_pers >= 2
    
    if nb_pers == np.inf:
        mass = np.sum(seq)
        max_score = 1
    else:
        mass = sum(pi for pi in islice(seq, nb_pers))
        max_score = (nb_pers - 1) / nb_pers
    
    noise_score = 1 - max(seq) / mass
    score = 1 - noise_score/max_score
    return score
    
def entropy_ring_score(seq, nb_pers = 2, base = np.e):
    """"Calculates entropy ring score from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative."""
    
    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1
        
    assert nb_pers >= 2
    
    if base is None:
        base = np.e
        
    if nb_pers == np.inf:
        nb_pers = len(seq)
    else:
        seq = [pi for pi in islice(seq, nb_pers)]
    noise_score = ss.entropy(seq, base = base)
    max_score = np.log(nb_pers) / np.log(base)
    score = 1 - noise_score/max_score
    return score
    
    
def ring_score_from_sequence(seq, flavour = 'geometric', nb_pers = 2, base = None):
    """Calculates ring score of different flavours from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative."""
    if len(seq) == 0:
        return 0
    if flavour in 'geometric':
        return geometric_ring_score(seq, nb_pers = nb_pers, base = base)
    elif flavour == 'amplitude':
        return amplitude_ring_score(seq, nb_pers = nb_pers)
    elif flavour == 'entropy':
        return entropy_ring_score(seq, nb_pers = nb_pers, base = base)
    elif flavour == 'linear':
        return linear_ring_score(seq, nb_pers = nb_pers)
    else:
        raise Exception(f"Ring score flavour {flavour} unknown.")