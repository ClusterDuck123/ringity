from matplotlib import animation

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import warnings

"""THIS IS LEGACY CODE THAT NEEDS TO BE UPDATED!"""

def get_edges(X,i,j):
    A = np.where((X>i) & (X<=j), 1,0)
    return nx.from_numpy_array(A)


def animate_persistence_plot(G, *args,                             
                             toa='weight', 
                             layout=None, 
                             reverse=False, 
                             dgm = None, 
                             fps=10, 
                             length=5):
    pass
    
#     if args:
#         warnings.warn('Non-keyworded arguments were skipped; Please use keyword arguments!')
    
    
# # Set up function parameters 
#     if dgm is None:
#         dgm, X = diagram(G = G, toa=toa, reverse=reverse, verbose=True, index=True)
#     else:
#         X = indexify(nx.to_numpy_array(G, weight=toa), reverse=reverse)
        
#     if layout is None:
#         layout = nx.spring_layout(G)
        
        

# # Set up animation parameters
#     births, deaths = map(list,zip(*[(k.birth,k.death) for k in dgm]))
#     x_max = max(births)
#     y_max = max(max([death for death in deaths if death != np.inf]), x_max)
    
#     for i, death in enumerate(deaths):
#         if death == np.inf:
#             deaths[i] = y_max+1

#     max_toa = max(deaths)
#     diagonal = [0, max_toa*1.1]

#     frames = fps*length
#     interval = 1000/fps
#     time = [k *max_toa*1.05/(frames-1) for k in range(frames)]
    
#     fig = plt.figure(figsize=(10,5))
    
    
# # Set up network plot parameters
#     plt.subplot(121)
#     global edges
#     edges = 0
#     ax1 = plt.gca()
#     ax1.set_title(f'#edges = {edges}')
    
#     x_coord, y_coord = zip(*layout.values())    
#     ax1.set_xlim(min(x_coord), max(x_coord))
#     ax1.set_ylim(min(y_coord), max(y_coord))
    
    
# # Set up persistence plot diagram parameters
#     plt.subplot(122)
#     ax2 = plt.gca()

#     ax2.set_xlim(diagonal)
#     ax2.set_ylim(diagonal)

#     square = ax2.add_patch(
#             patches.Rectangle(
#                 (0,0),
#                 0, 0,
#                 alpha=0.2, facecolor="red"
#             ))

#     rectangle = ax2.add_patch(
#             patches.Rectangle(
#                 (0,0),
#                 0, max(deaths)*1.1,
#                 alpha=0.1, facecolor="red"
#             ))


# # Initialize animation with static plot
#     def init():     
#         S = nx.Graph()
#         S.add_nodes_from(G.nodes())
#         nx.draw(S, pos=layout, node_size=10, alpha=0.3, ax=ax1)      
        
#         ax2.plot(births, deaths, '*', color='mediumorchid')
#         ax2.plot(diagonal, diagonal, 'b', lw=2, zorder=10)
#         return ()


# # Finally we get to the interesting part
#     def animate(i):        
#         S = get_edges(X,time[i-1], time[i])
#         global edges
#         edges += S.number_of_edges()
#         nx.draw_networkx_edges(S, pos=layout, alpha=0.3, ax=ax1)
#         ax1.set_title(f'#edges = {edges}')
                
#         square.set_height(time[i])
#         square.set_width(time[i])
#         rectangle.set_width(time[i])

#         born = len([x for x in births if x <= time[i]])

#         if born == 0:
#             return () 

#         x_born, y_born = births[:born], deaths[:born]
#         xy_dead = [(x,y) for x,y in zip(x_born,y_born) if y <= time[i]]
#         if not xy_dead:
#             x_dead, y_dead = [0], [0]
#         else:
#             x_dead, y_dead = zip(*xy_dead)

#         ax2.plot(x_born, y_born, '*', color='darkviolet')
#         ax2.plot(x_dead, y_dead, '*', color='indigo')

#         return ()

    
#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                    frames=frames, interval=interval, blit=True)    
#     return anim