import re
import csv

from  pathlib import Path

"""
This module is for supplementary material needed in demonstrations of ringity.
"""

def _get_canonical_gene_id(gene_id):
    """Convert string to match ringity-conventions of gene id names."""
    gene_id = gene_id.upper()
    gene_id = ''.join(filter(str.isalnum, gene_id))
    gene_id = re.sub('ID|GENE', '', gene_id)
    return gene_id

def get_cell_cycle_genes(geneset = 'TIROSH_ALL_MARKERS', gene_id = 'SYMBOL'):
    """Returns list of cell cycle genes stored in ringity/_data.

    Returns
    -------
    gs_ls : list of cell cycle genes


    Parameters
    ----------
    gene_id : string, optional, default: "symbol"
        Specifies gene identifiers of genes to return. Available gene id's are:
        "SYMBOL", "ENSEMBL" and "ENTREZ". String recognition will be case 
        insensitive. So, e.g., "symbol" will be matched with "SYMBOL".

    geneset : string, optional, default: "TIROSH_ALL_MARKERS"
        Specifies gene sets available at ringity/data. Available gene sets are:
        "TIROSH_ALL_MARKERS", "TIROSH_G1S_MARKERS", "TIROSH_G2M_MARKERS",
        "HALLMARK_E2F_TARGETS" and "HALLMARK_G2M_CHECKPOINT".
    """

    gs_table_path = Path(__file__).parent.parent / "_data" / (geneset.upper() + ".csv")
    with open(gs_table_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        gs = {row[header.index(gene_id)] for row in reader}

    gs_ls = list(gs)
    return gs_ls