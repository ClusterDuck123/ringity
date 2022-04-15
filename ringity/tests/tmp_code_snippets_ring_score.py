import numpy as np
import ringity as rng

seq1 = sorted(np.random.uniform(size = 10), reverse=True)
#seq1 = (1,0,0,0,0)
seq2 = seq1[:3]

print(seq2)

random_base1 = np.random.uniform(0,10)
random_base2 = np.random.uniform(0,10)

ring_score = rng.amplitude_ring_score

print(rng.gap_ring_score(seq1))
print(rng.gap_ring_score(seq2))

print()

print(rng.geometric_ring_score(seq1, base = random_base1))
print(rng.geometric_ring_score(seq2, base = random_base1))
print(rng.geometric_ring_score(seq1, base = random_base2))
print(rng.geometric_ring_score(seq2, base = random_base2))

print()

print(ring_score(seq1, nb_pers = 2))
print(ring_score(seq2, nb_pers = 2))

print()

print(ring_score(seq1, nb_pers = 3))
print(ring_score(seq2, nb_pers = 3))

print()

print(ring_score(seq1, nb_pers = 4))
print(ring_score(seq2, nb_pers = 4))