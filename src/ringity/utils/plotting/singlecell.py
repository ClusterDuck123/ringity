import umap
import pandas as pd
import plotly.express as px

from ringity.utils.singlecell.singlecell import _parse_var_names, _parse_obs_names

def plot_dimreduction_from_anndata(adata, 
                                   var_names = None, 
                                   obs_names = None,
                                   color = None,
                                   reduction = 'umap', 
                                   dim = 2,
                                   plotter = 'plotly',
                                   return_data = False):
    
    var_names = _parse_var_names(adata, var_names)
    obs_names = _parse_obs_names(adata, obs_names)
    
    X = adata[obs_names, var_names].X
    
    if reduction == 'umap':
        reducer = umap.UMAP(n_components=dim)
    
    plot_data = pd.DataFrame(reducer.fit_transform(X), 
                             index = adata.obs_names,
                             columns = [f"UMAP{i}" for i in range(1, dim +1)])

    plot_data = pd.concat([plot_data, adata.obs], 
                          axis = 1)
    
    if plotter == 'plotly':
        if dim == 2:
            fig = px.scatter(data_frame = plot_data, 
                       x = 'UMAP1', y = 'UMAP2',
                       color = color)
        if dim == 3:
            fig = px.scatter_3d(data_frame = plot_data, 
                       x = 'UMAP1', y = 'UMAP2', z = 'UMAP3',
                       color = color)
           
    fig.show()
    
    if return_data:
        return plot_data 
