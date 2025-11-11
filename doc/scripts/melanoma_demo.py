import scanpy as sc
import ringity.singlecell.singlecell as rsc

# Single cell RNA-seq of three human melanoma cell lines: Ma-Mel-123, Ma-Mel-108 and Ma-Mel-93
experiment = "E-GEOD-81383"

# Download ebi_expression_atlas experiment
adata = sc.datasets.ebi_expression_atlas(experiment)

# Get cell cycle genes
cc_genes = rsc.get_cell_cycle_genes(gene_id="ensembl")

# Calculate ring score of subspace
for cell_line in adata.obs["Sample Characteristic[cell line]"].unique():
    adata_cl = adata[adata.obs["Sample Characteristic[cell line]"] == cell_line]
    ring_score = rsc.ring_score_from_anndata(adata_cl, var_names=cc_genes)
    print(f"{cell_line}: {ring_score}")
