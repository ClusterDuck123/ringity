import numpy as np
import ringity as rng

seq1 = (1, 0, 0, 0, 0)
seq2 = (1, 1, 0, 0, 0)

print("Geometric ring score of circle: "
      f"{rng.ring_score_from_sequence(seq1, 'geometric'):.3f}")
print("Geometric ring score of figure-eight: "
      f"{rng.ring_score_from_sequence(seq2, 'geometric'):.3f}")

print("Gap ring score of circle: "
      f"{rng.ring_score_from_sequence(seq1, 'gap'):.3f}")
print("Gap ring score of figure-eight: "
      f"{rng.ring_score_from_sequence(seq2, 'gap'):.3f}")
