from ringity.core.ringscore_flavours import ring_score_from_sequence
from ringity.core.ringscore_functions import (
                                        pdiagram_from_point_cloud, 
                                        ring_score_from_point_cloud)

import magic
import numpy as np
import scanpy as sc

def ring_score_from_anndata(adata, 
                            var_names = None, 
                            obs_names = None, 
                            flavour = 'geometric',
                            base = None):
    """Calculate ring score from AnnData object.

    Parameters
    ----------
    adata : AnnData
    var_names : list of variable names to subset
    obs_names : list of observation names to subset

    Returns
    -------
    score : float, ring-score of given object.
    """

    var_names = _parse_var_names(adata, var_names)
    obs_names = _parse_obs_names(adata, obs_names)
        
    X = _parse_X(adata[obs_names, var_names])
    score = ring_score_from_point_cloud(X, 
                            flavour = flavour, # NEEDS POTENTIALLY GENERALIZATION FOR LIST OF FLAVOURS
                            base = base)
    return score


def pdiagram_from_AnnData(
                        adata,
                        var_names = None, 
                        obs_names = None, 
                        persistence = 'VietorisRipsPersistence',
                        metric = 'euclidean',
                        metric_params = {},
                        homology_dim = 1,
                        **kwargs):
    """Constructs a PersistenceDiagram object from an AnnData object.

    Parameters
    ----------
    adata : AnnData
    var_names : list of variable names to subset
    obs_names : list of observation names to subset

    Returns
    -------
    score : float, ring-score of given object.
    """
    
    var_names = _parse_var_names(adata, var_names)
    obs_names = _parse_obs_names(adata, obs_names)
    X = _parse_X(adata[obs_names, var_names])
    dgm = pdiagram_from_point_cloud(X)
    return dgm
        

def process_adata_from_counts(adata, 
                              min_genes = 1_000,
                              min_cells = 3,
                              target_sum = 10_000,
                              diffuse = True,
                              diffuse_t = 1,
                              highly_variable_subset = True,
                              verbose = False
                              ):
    """Follows Seurat guidelines to preprocess data."""
    
    sc.pp.filter_cells(adata, min_genes = min_genes)
    sc.pp.filter_genes(adata, min_cells = min_cells)
    sc.pp.normalize_total(adata, target_sum = target_sum)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, inplace = highly_variable_subset)
    
    if diffuse:
        magic_op = magic.MAGIC(t = diffuse_t, verbose = verbose)
        adata = magic_op.fit_transform(adata)
        
    return adata


def _parse_var_names(adata, var_names):
    if var_names is None:
        return adata.var_names
    else:
        return adata.var_names.intersection(var_names)
    
def _parse_obs_names(adata, obs_names):
    if obs_names is None:
        return adata.obs_names
    else:
        return adata.obs_names.intersection(obs_names)

def _parse_X(adata):
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return X