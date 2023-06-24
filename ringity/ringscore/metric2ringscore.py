import numpy as np
import scipy.stats as ss

def ring_score_from_sequence(seq,
                             flavour = 'geometric',
                             nb_pers = None,
                             exponent = 2):
    """Calculates ring score from sequence of positive numbers.

    ``seq`` can be any iterator containing numbers and will be sorted.

    Caution: there are no checks to test if the sequence is non-negative."""
    if len(seq) == 0:
        return 0
    if flavour in 'geometric':
        return geometric_ring_score(seq, nb_pers = nb_pers, exponent = exponent)
    elif flavour == 'gap':
        return gap_ring_score(seq)
    elif flavour == 'amplitude':
        return amplitude_ring_score(seq, nb_pers = nb_pers)
    elif flavour == 'entropy':
        return entropy_ring_score(seq, nb_pers = nb_pers)
    elif flavour == 'linear':
        return linear_ring_score(seq, nb_pers = nb_pers)
    else:
        raise Exception(f"Ring score flavour {flavour} unknown.")


# =============================================================================
#  -------------------------- RING SCORE FLAVOURS ----------------------------
# =============================================================================

def gap_ring_score(seq):
    """"Calculates gap ring score from sequence of positive numbers.

    ``seq`` can be any interator and will be sorted. However, there are
    no checks to test if the sequence is non-negative."""

    seq = sorted(seq, reverse = True)

    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1

    p0 = seq[0]
    p1 = seq[1]

    return 1 - p1/p0

def geometric_ring_score(seq, nb_pers = np.inf, exponent = 2, tol = 1e-10):
    """"Calculates geometric ring score from sequence of positive numbers.

    ``seq`` can be any interator and will be sorted.
    However, there are no checks to test if the sequence is non-negative."""
    if exponent == 1:
        return linear_ring_score(seq, nb_pers = nb_pers)
    if exponent == np.inf:
        return gap_ring_score(seq)

    if nb_pers is None:
        nb_pers = np.inf

    seq = sorted(seq, reverse = True)

    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1


    if nb_pers == np.inf:
        max_score = 1 / (exponent-1)
        # weights of later persistences are below tolerance level
        nb_pers =  int(-np.log(tol) // np.log(exponent)) + 1
    else:
        max_score = (1 - exponent**(-nb_pers + 1)) / (exponent-1)

    assert nb_pers >= 2

    p0 = seq[0]
    noise = sum(pi / (p0 * exponent**i) for i, pi in enumerate(seq[1:nb_pers], 1))
    score = 1 - noise/max_score

    return score

def linear_ring_score(seq, nb_pers = 2):
    """"Calculates linear ring score from sequence of positive numbers.

    ``seq`` can be any interator and will be sorted.
    However, there are no checks to test if the sequence is non-negative.
    Extending the weights from 1/i to 1/(base*i) yields back the same score."""

    if nb_pers is None:
        nb_pers = 2

    seq = sorted(seq, reverse = True)

    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1

    if nb_pers == np.inf:
        raise Exception(f"Linear ring score for `nb_pers == np.inf` not defined!")

    if nb_pers is None:
        nb_pers = 2

    assert nb_pers >= 2

    max_score = sum(1/i for i in range(1, nb_pers))
    p0 = seq[0]

    noise = sum(pi / (p0*i) for i, pi in enumerate(seq[1:nb_pers], 1))
    score = 1 - noise / max_score
    return score

def amplitude_ring_score(seq, nb_pers = np.inf):
    """"Calculates amplitude ring score from sequence of positive numbers.

    Score is linearly scaled to have the range [0,1]; i.e. the score is
    defined as ``score = 1 - (N/(N-1)) * (1 - p0/(sum pi))``.

    ``seq`` can be any interator and will be sorted.
    However, there are no checks to test if the sequence is non-negative!"""

    if nb_pers is None:
        nb_pers = np.inf

    seq = sorted(seq, reverse = True)

    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1

    if nb_pers == np.inf:
        mass = sum(seq)
        max_score = 1
    else:
        mass = sum(seq[:nb_pers])
        max_score = (nb_pers - 1) / nb_pers

    assert nb_pers >= 2

    noise = 1 - max(seq) / mass
    score = 1 - noise/max_score
    return score

def entropy_ring_score(seq, nb_pers = 2):
    """"Calculates entropy ring score from sequence of positive numbers.

    ``seq`` can be any interator and will be sorted.
    However, there are no checks to test if the sequence is non-negative."""

    if nb_pers is None:
        nb_pers = np.inf

    seq = sorted(seq, reverse = True)

    if len(seq) == 0:
        return 0
    if len(seq) == 1:
        return 1

    if nb_pers == np.inf:
        nb_pers = len(seq)
    else:
        seq = seq[:nb_pers]

    assert nb_pers >= 2

    noise = ss.entropy(seq)
    max_score = np.log(nb_pers)
    score = 1 - noise/max_score
    return score