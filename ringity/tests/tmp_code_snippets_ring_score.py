import scanpy
import ringity as rng
import random

# breast cancer data set
adata1 = scanpy.datasets.ebi_expression_atlas("E-GEOD-75688")

# melanoma data set
adata2 = scanpy.datasets.ebi_expression_atlas("E-GEOD-81383")

cc_genes = rng.get_cell_cycle_genes(gene_id='ensembl')

rand_genes1 = random.sample(list(adata1.var_names),len(cc_genes))
rand_genes2 = random.sample(list(adata2.var_names),len(cc_genes))

cc_ring_score1 = rng.ring_score(adata1[:,cc_genes].X)
rand_ring_score1 = rng.ring_score(adata1[:,rand_genes1].X)

cc_ring_score2 = rng.ring_score(adata2[:,cc_genes].X)
rand_ring_score2 = rng.ring_score(adata2[:,rand_genes2].X)

print(f"Ring-score of breast cancer cell-cycle genes: {cc_ring_score1:.3f}")
print(f"Ring-score of breast cancer random genes    : {rand_ring_score1:.3f}")

print()

print(f"Ring-score of melanoma cell-cycle genes: {cc_ring_score2:.3f}")
print(f"Ring-score of melanoma random genes    : {rand_ring_score2:.3f}")
