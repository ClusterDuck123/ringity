import numpy as np

from gtda.homology import VietorisRipsPersistence
from ringity.generators.diagram import random_pdiagram, random_pdiagram_point
from ringity.classes.diagram import PDgm, PDgmPt
from ringity.classes.exceptions import DgmTimeError

# Single diagram point

pt = PDgmPt((0,1))
try:
    pt = PDgmPt((1,))
except DgmTimeError:
    pass
    
    
X = np.random.uniform(size = (10, 3))
dgm, = VietorisRipsPersistence().fit_transform([X])

pdgm = PDgm(dgm[:,:2])
pdgm.append([0,10])

print(pdgm[:10])
print(pdgm[0])

pdgm = PDgm([(0,1), (0,3), (0,2)])
pdgm.append((0, 3.5))
pdgm.append((2, 2))
pdgm.append((3, 3))
pdgm.append((5, 5))

print()

print(pdgm)

print()

print(pdgm.trimmed(10))
print(pdgm.trimmed(2))
print(pdgm.trimmed())
print(pdgm)

print()

print(pdgm.persistences)
print(pdgm.sequence)

print()

pdgm = PDgm([(0,1)] + [(0,0)]*10)
print(pdgm.score())

pdgm = PDgm([(0,2)] + [(0,0)]*10)
print(pdgm.score())

pdgm = PDgm([(0,1)]*3 + [(0,0)]*10)
print(pdgm.score())
print(pdgm.score(skip=1))
print(pdgm.score(skip=1))
print(pdgm.score(skip=2))

print()

random_dgm = random_pdiagram(2**10, 'exponential')
print(random_dgm[:10])

print(random_dgm.score())
