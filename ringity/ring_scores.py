import numpy as np
import scipy.stats as ss

from itertools import islice

def geometric_ring_score(seq, nb_pers = 2, base = 2, tol = 1e-10):
    """"Calculates geometric ring score from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative."""
    
    if base is None:
        base = 2

    if nb_pers == np.inf:
        max_score = 1 / (1-base)
        # weights of persistences after this one are below tolerance level 
        nb_pers = tol // np.log10(base)
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
    if base is None:
        base = 1
    
    if nb_pers == np.inf:
        raise Exception(f"Linear ring score for base == np.inf not defined!")
    
    max_score = sum(1/i for i in range(1, nb_pers)) / base
        
    seq_iter = iter(sorted(seq, reverse = True))
    p0 = next(seq_iter)    
    noise_score = sum((pi / (p0*base*i) for i, pi in enumerate(islice(seq_iter, nb_pers-1), 1)))
    score = 1 - noise_score/max_score
    return score
    
def amplitude_ring_score(seq, nb_pers = 2):
    """"Calculates amplitude ring score from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative."""
    if nb_pers == np.inf:
        return seq[0] / np.sum(seq)
    else:
        pass
    
def entropy_ring_score(seq, nb_pers = 2, base = np.e):
    """"Calculates entropy ring score from sequence of positive numbers.
    
    ``seq`` can be any interator and will be sorted. 
    However, there are no checks to test if the sequence is non-negative."""
    if base is None:
        base = np.e
        
    if nb_pers == np.inf:
        # This can probably be simplified!
        normalisation = np.log(len(dgm.trimmed()))
        return 1 - np.log(base)*ss.entropy(dgm.sequence, base = base) / normalisation
    else:
        pass
    
    
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
        
        
#  ------------------------------------- LEGACY --------------------------------------

def ring_score(X):
    """Calculates ring score from point clud of different flavours.
    
    If X is a numpy array it is converted into a ringity PersistenceDiagram object.
    Otherwise X is assumed to be a PersistenceDiagram itself. 
    A PersistenceDiagram can easily be converted into a numpy array via np.array(dgm)."""
    
    if isinstance(X, np.ndarray):
        dgm = rng.vietoris_rips_from_point_cloud(X)
    else:
        dgm = X
        
    if len(dgm.trimmed()) == 0:
        return 0
    
    if flavour == 'geometric':
        if N_trim is None:
            return dgm.ring_score
        else:
            return trimmed_geometric_score(dgm, N_trim)
    elif flavour == 'amplitude':
        if N_trim is None:
            return dgm.sequence[0] / np.sum(dgm.sequence)
        else:
            dgm = dgm.trimmed(N_trim)
            p0 = dgm.sequence[0]
            p_sum = np.sum(dgm.sequence)
            return (N_trim*p0 - p_sum) / (p_sum*(N_trim-1))
    elif flavour == 'entropy':
        # Needs to be checked / adapted
        if base is None:
            base = np.e
        
        # Normalize by uniform distribution on non-zero persistences.
        
        # dgm.trimmed() reduces the diagram to points with positive length. 
        # TODO: trimm dgm automatically in the packeage! 
        if N_trim is None:
            normalisation = np.log(len(dgm.trimmed()))
            return 1 - np.log(base)*ss.entropy(dgm.sequence, base = base) / normalisation
        else:
            dgm = dgm.trimmed(N_trim)
            normalisation = np.log(N_trim)
            return 1 - np.log(base)*ss.entropy(dgm.sequence, base = base) / normalisation
    else:
        raise Exception(f"Flavour {flavour} unknown.")