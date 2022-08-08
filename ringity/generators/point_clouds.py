import numpy as np

def two_adjacent_circles(N, 
                         r = 1, 
                         noise = 0,
                         seed = None):
    """Generates sample points of two circles with radius `r` and `1-r`."""
    
    np.random.seed(seed=seed)
    s,t = np.random.uniform(0, 2*np.pi, [2, N // 2])
    
    x = np.hstack([r*np.cos(s) + r, (1-r)*np.cos(t) + (1+r)])
    y = np.hstack([r*np.sin(s), (1-r)*np.sin(t)])
    
    return np.vstack((x,y)).T

def circle(N, 
           noise = 1,
           seed = None):
    """Generates sample points of circle with radius 1 and gaussian noise 
    standard deviation `scale`."""
    
    np.random.seed(seed=seed)
    
    t = np.random.uniform(0, 2*np.pi, size = N)
    noise_x, noise_y = np.random.normal(scale = noise, size = [2, N])
    
    x = np.cos(t) + noise_x
    y = np.sin(t) + noise_y
    
    return np.vstack((x,y)).T


def vonmises_circles(N, 
            kappa = 1, 
            noise = 0.1,
            seed = None):
    """Generates sample points of two circles with radius `r` and `1-r`."""
    
    np.random.seed(seed=seed)
    
    if kappa == 0:
        t = np.random.uniform(0, 2*np.pi, size = N)
    else:
        t = ss.vonmises(kappa = kappa).rvs(size = N)
        
    noise_x, noise_y = np.random.normal(scale = noise, size = [2, N])
    
    x = np.cos(t) + noise_x
    y = np.sin(t) + noise_y
    
    return np.vstack((x,y)).T


def annulus(N, 
            r = 1,
            noise = 0, 
            seed = None):
    """Generates sample points of a 2-dimensional Annulus with inner radius `r`.
    Outside radius is taken to be 1."""
    
    np.random.seed(seed=seed)
    u,v = np.random.uniform(0, 1, [2,N])
    
    phi = 2*np.pi*u
    r = np.sqrt((1-r**2)*v + r**2)
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    
    return np.vstack((x,y)).T


def cylinder(N,
             height = 1,
             noise = 0,
             seed = None):
    """Generates sample points of a cylinder in 3D with unit radius and height `height`."""
    np.random.seed(seed=seed)
    u,v = np.random.uniform(0, 1, [2,N])
    
    phi = 2*np.pi*u
    
    x = np.cos(phi)
    y = np.sin(phi)
    z = np.random.uniform(0, height, N)
    
    return np.vstack((x,y,z)).T


def torus(N = 100,
          r  = 1,
          noise = 0,
          seed = None):
    "Generates sample points of a cylinder in 3D with radius of revolution being 1 and outer radius 'r'"
    
    np.random.seed(seed=seed)
    u,v = np.random.uniform(0, 2*np.pi, [2,N])
    
    x = (1 + r*np.cos(v))*np.cos(u)
    y = (1 + r*np.cos(v))*np.sin(u)
    z = r*np.sin(v)
    
    return np.vstack((x,y,z)).T