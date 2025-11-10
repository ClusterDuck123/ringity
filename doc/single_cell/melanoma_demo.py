import scanpy as sc
import ringity as rng

# Single cell RNA-seq of three human melanoma cell lines: Ma-Mel-123, Ma-Mel-108 and Ma-Mel-93
experiment = "E-GEOD-81383"

# Download ebi_expression_atlas experiment
adata = sc.datasets.ebi_expression_atlas(experiment)

# Get cell cycle genes
cc_genes = rng.get_cell_cycle_genes(gene_id="ensembl")

# Calculate ring score of subspace
for cell_line in adata.obs["Sample Characteristic[cell line]"].unique():
    adata_cl = adata[adata.obs["Sample Characteristic[cell line]"] == cell_line]
    ring_score = rng.ring_score(adata_cl[:, cc_genes].X)
    print(f"{cell_line}: {ring_score}")
