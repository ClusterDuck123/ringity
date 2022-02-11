import numpy as np
from ringity.classes.new_diagram import PDgm

def read_pdiagram(fname, **kwargs):
    """
    Wrapper for numpy.genfromtxt.
    """
    return PDgm(np.genfromtxt(fname, **kwargs))
    
def write_pdiagram(dgm, fname, **kwargs):
    """
    Wrapper for numpy.savetxt.
    """
    array = np.array(dgm)
    np.savetxt(fname, array, **kwargs)