import re
import csv

from  pathlib import Path

"""
This module is for supplementary material needed in demonstrations of ringity.
"""

def get_canonical_gene_id(gene_id):
    """Convert string to match ringity-conventions of gene id names."""
    gene_id = gene_id.lower()
    gene_id = ''.join(filter(str.isalnum, gene_id))
    gene_id = re.sub('id|gene', '', gene_id)
    return gene_id

def get_cell_cycle_genes(gene_id = 'symbol', geneset = 'tirosh_cc'):
    """Returns list of cell cycle genes stored in ringity/_data.

    Returns
    -------
    gs_ls : list of cell cycle genes


    Parameters
    ----------
    gene_id : string, optional, default: ``"symbol"``
        Specifies gene identifiers of genes to return. Available gene id's are:
        "symbol", "ensembl" and "entrez". If input string deviates from
        these cases it will try to find an appropriate match; e.g., "SYMBOL_ID"
        will be matched with "symbol".

    geneset : string, optional, default: ``"symbol"``
        Specifies gene sets available at ringity/data. Available gene sets are:
        "tirosh_cc", "tirosh_g1s", "tirosh_g2m",
        "hallmark_e2f" and "hallmark_g2m".
    """

    # TODO: Include UNIPRO ID and extend data tables.

    gs_table_path = Path(__file__).parent.parent / "_data" / (geneset.lower() + ".csv")
    with open(gs_table_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        gs = {row[header.index(gene_id)] for row in reader}

    gs_ls = list(gs)
    return gs_ls


def process_adata_from_counts(adata, 
                              min_genes = 1_000,
                              min_cells = 3,
                              target_sum = 10_000,
                              diffuse = True,
                              diffuse_t = 1,
                              highly_variable_subset = True,
                              ):
    """Follows Seurat guidelines to preprocess data."""
    
    sc.pp.filter_cells(adata, min_genes = min_genes)
    sc.pp.filter_genes(adata, min_cells = min_cells)
    sc.pp.normalize_total(adata, target_sum = target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, inplace = highly_variable_subset)
    
    if diffuse:
        magic_op = magic.MAGIC(t = diffuse_t)
        adata = magic_op.fit_transform(adata)
        
    return adata