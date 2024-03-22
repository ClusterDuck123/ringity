import numpy as np
import .transformations as rtrafos

def simulate_activity(r, beta=None, rate=None, n_cycles = 5, N = 2**9) -> float:
    rate = rtrafos._get_rate(beta=beta, rate=rate)
    thetas = np.random.exponential(scale = 1/ rate, size = N)

    t = np.linspace(0, n_cycles*2*np.pi, 2**9)
    a = [np.mean([activity(ti, r=r, t0=t0) for t0 in thetas]) for ti in t]