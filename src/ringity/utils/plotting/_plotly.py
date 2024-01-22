import networkx as nx
import importlib.util

plotly_spec = importlib.util.find_spec("plotly")

if plotly_spec is None:
    pass # TODO: Deal with this dependency properly
else:
    import plotly.graph_objects as go

from ringity.utils.plotting.styling import CEMM_COL1, CEMM_COL2

def _get_edge_coord_3d(G, pos):
    edge_x = []
    edge_y = []
    edge_z = []
    for u,v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    return edge_x, edge_y, edge_z

def _get_edge_coord_2d(G, pos):
    edge_x = []
    edge_y = []
    for u,v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    return edge_x, edge_y

def _get_node_coord_3d(G, pos):
    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
    return node_x, node_y, node_z

def _get_node_coord_2d(G, pos):
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    return node_x, node_y


def plot_nx_plotly_3d(G, pos = None, hoverinfo = None, node_color = None):
    if pos is None:
        pos = nx.spring_layout(G, dim = 3)
        
    edge_x, edge_y, edge_z = _get_edge_coord_3d(G, pos)
    node_x, node_y, node_z = _get_node_coord_3d(G, pos)

    edge_trace = go.Scatter3d(
        x = edge_x, y = edge_y, z = edge_z,
        line = dict(width = 0.5, color = f'rgba({CEMM_COL2[0]}, {CEMM_COL2[1]}, {CEMM_COL2[2]}, 1)'),
        hoverinfo = 'none',
        mode = 'lines')

    node_trace = go.Scatter3d(
        x = node_x, y = node_y, z = node_z,
        mode = 'markers',
        hoverinfo = 'text',
        marker = dict(
            size = 2.5,
            color = f'rgba({CEMM_COL1[0]}, {CEMM_COL1[1]}, {CEMM_COL1[2]}, 0.95)'))
    
    if node_color == 'degree':
        degs = list(dict(nx.degree(G)).values())
        node_trace.marker = dict(
            showscale = True,
            colorscale = 'Blues',
            reversescale = True,
            color = [],
            size = 2.5,
            colorbar = dict(
                thickness = 15,
                title = 'Node Connections',
                xanchor = 'left',
                titleside = 'right'
        ))
        node_trace.marker.color = degs
    
    if hoverinfo == 'degree':
        degs = list(dict(nx.degree(G)).values())
        node_text = [f'degree: {deg}' for deg in degs]
        node_trace.text = node_text
        node_trace.marker.size = 5
    
    fig = go.Figure(data = [edge_trace, node_trace],
                    layout = go.Layout(showlegend = False))
    fig.update_scenes(xaxis_visible = False, yaxis_visible = False, zaxis_visible = False )
    fig.show()
    
    
def plot_nx_plotly_2d(G, pos = None, hoverinfo = None, node_color = None):
    if pos is None:
        pos = nx.spring_layout(G, dim = 2)
        
    edge_x, edge_y = _get_edge_coord_2d(G, pos)
    node_x, node_y = _get_node_coord_2d(G, pos)

    edge_trace = go.Scatter(
        x = edge_x, y = edge_y,
        line = dict(width = 0.5, color = f'rgba({CEMM_COL2[0]}, {CEMM_COL2[1]}, {CEMM_COL2[2]}, 1)'),
        hoverinfo = 'none',
        mode = 'lines')

    node_trace = go.Scatter(
        x = node_x, y = node_y,
        mode = 'markers',
        hoverinfo = 'text',
        marker = dict(
            size = 2.5,
            color = f'rgba({CEMM_COL1[0]}, {CEMM_COL1[1]}, {CEMM_COL1[2]}, 0.95)'))
        
    if node_color == 'degree':
        degs = list(dict(nx.degree(G)).values())
        node_trace.marker = dict(
            showscale = True,
            colorscale = 'Blues',
            reversescale = True,
            color = [],
            size = 2.5,
            colorbar = dict(
                thickness = 15,
                title = 'Node Connections',
                xanchor = 'left',
                titleside = 'right'
        ))
        node_trace.marker.color = degs
        
        
    if hoverinfo == 'degree':
        degs = list(dict(nx.degree(G)).values())
        node_text = [f'degree: {deg}' for deg in degs]
        node_trace.text = node_text
        node_trace.marker.size = 5
    
    fig = go.Figure(data = [edge_trace, node_trace],
                    layout = go.Layout(
                                showlegend = False,
                                paper_bgcolor = 'rgba(0,0,0,0)',
                                plot_bgcolor = 'rgba(0,0,0,0)'))

    fig.update_xaxes(visible = False)
    fig.update_yaxes(visible = False)
    fig.show()