"""
Module to deal with single cell data. Collects all related functions 
from around the ringity package. Will probably become its own 
subpackage at some point.
"""
import numpy as np
import ringity as rng
import importlib.util

scanpy_spec = importlib.util.find_spec("scanpy")
magic_spec = importlib.util.find_spec("magic")

if scanpy_spec is None:
    pass # TODO: Deal with this dependency properly
else:
    import scanpy as sc

if magic_spec is None:
    pass # TODO: Deal with this dependency properly
else:
    import magic

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
    score = rng.ring_score_from_point_cloud(X, 
                            flavour = flavour, # NEEDS POTENTIALLY GENERALIZATION FOR LIST OF FLAVOURS
                            base = base)
    return score


def pdiagram_from_anndata(
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
    pdgm = rng.pdiagram_from_point_cloud(X)
    return pdgm
        

def process_adata_from_counts(adata, 
                              min_genes = 1_000,
                              min_cells = 3,
                              target_sum = 10_000,
                              diffuse = True,
                              diffuse_t = 1,
                              highly_variable_subset = True,
                              verbose = False
                              ):
    """Filters adata for genes and cells. Normalizes counts per cell and 
    performs a log(x+1) transformation.

    Parameters
    ----------
    adata : Anndata
        Raw count data.
    min_genes : int, optional
        Removes cells that have less than ``min_genes`` expressed.
        By default 1_000.
    min_cells : int, optional
        Removes genes that have less than ``min_cells`` expressed.
        By default 3.
    target_sum : int, optional
        Normalize each cell, by dividing through the total sum
        and multiplying by ``target_sum``. By default 10_000.
    diffuse : bool, optional
        Diffuse normalized counts using ``magic``. By default True.
    diffuse_t : int, optional
        Number of diffusion steps in ``magic``. By default 1.
    highly_variable_subset : bool, optional
        Filter genes further after normalization, based on their variance.
        By default True
    verbose : bool, optional
        Prints output about individual steps. By default False.

    Returns
    -------
    Anndata
        Normalized count data.
    """    
    
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