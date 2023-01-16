import time

def print_calc_status(nr, nb_loops, t1):
    """Display calculation status of a loop."""
    t2 = time.time()
    p = (nr+1)/nb_loops
    diff = t2-t1
    print(f'progress: {100*p:.2f}% - {diff:.2f}sec; eta: {diff*(1/p - 1):.2f}sec', end = '\r')