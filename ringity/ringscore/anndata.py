from ringity.ringscore.ring_score_flavours import ring_score_from_sequence
from ringity.ringscore.core import pdiagram_from_point_cloud

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
    if var_names is None: 
        var_names = adata.var_names
    else:
        var_names = adata.var_names.intersection(var_names)
        
    if obs_names is None: 
        obs_names = adata.obs_names
    else:
        obs_names = adata.obs_names.intersection(obs_names)
        
    X = adata[obs_names, var_names].X
    
    if type(X) == np.ndarray:
        dgm = pdiagram_from_point_cloud(X)
    else:
        dgm = pdiagram_from_point_cloud(X.toarray())
        
    if isinstance(flavour, str):
        score = ring_score_from_sequence(dgm.sequence, 
                                            flavour = flavour, 
                                            base = base)
        return score
    else:
        scores = [ring_score_from_sequence(dgm.sequence, 
                                               flavour = flav, 
                                               base = base)
                              for flav in flavour]
            
        
        return scores


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
    
    if var_names is None: 
        var_names = adata.var_names
    else:
        var_names = adata.var_names.intersection(var_names)
        
    if obs_names is None: 
        obs_names = adata.obs_names
    else:
        obs_names = adata.obs_names.intersection(obs_names)
        
    X = adata[obs_names, var_names].X
    
    if type(X) == np.ndarray:
        dgm = pdiagram_from_point_cloud(X)
    else:
        dgm = pdiagram_from_point_cloud(X.toarray())    
        
        return dgm
        

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
        magic_op = magic.MAGIC(t = 1)
        adata = magic_op.fit_transform(adata)
        
    return adata