import numpy as np
import ringity as rng
      
      
X = np.random.uniform(size = (2**4,2))
dgm = rng.pdiagram_from_point_cloud(X)


dgm2 = dgm.copy()
dgm2.append((3,4))

print(type(dgm))
print(type(dgm2))

print(dgm)
print(dgm2)

print(repr(dgm))
print(str(dgm))