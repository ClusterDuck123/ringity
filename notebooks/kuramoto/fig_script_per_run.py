import os
import networkx as nx
from load_saved_info import NetworkLoader
from kuramoto import Kuramoto
import matplotlib.pyplot as plt
import numpy as np

# Choose the network and run you want to load
subfolder = os.listdir("data/verbose/")[1]
folder = f"data/verbose/{subfolder}"
run_i = 5
loader = NetworkLoader(folder, verbose=True)

# loads or calculates some things about the network
G = nx.from_numpy_array(loader.adj_mat)
N = G.order()
pos_spring = nx.spring_layout(G)
pos_construction = {
    i: np.array([np.cos(v), np.sin(v)]) for i, v in enumerate(loader.positions)
}
pos_construction_dodged = {}
for i, v in enumerate(loader.positions):
    r = 1 + 0.1 * np.random.randn()
    pos_construction_dodged[i] = np.array([r * np.cos(v), r * np.sin(v)])

# loads or calculates some things about the run
run = loader.runs[run_i]
activity = run["activity"]
T = run["T"]
dt = run["dt"]
timeframe = np.linspace(0, T, activity.shape[1])

print(timeframe.shape)
print(activity.shape)

# plot the activities over time simulatenously for all nodes
fig, ax = plt.subplots()
for i in range(activity.shape[0]):
    ax.plot(timeframe, activity[i, :], c="#00000055")
plt.show()
ax.set_xlabel("time")
ax.set_ylabel("phase")

# Plot the coherence over time
fig, ax = plt.subplots()
ax.plot(
    timeframe,
    [Kuramoto.phase_coherence(activity[:, i]) for i in range(activity.shape[1])],
    c="k",
)
ax.set_xlabel("time")
ax.set_ylabel("phase coherence")

mean = run["mean"]
std = run["std"]

fig.text(0.0, 0.03, rf"$\sigma$:{round(std,2)}")
fig.text(0.0, 0.06, rf"$\mu$:{round(mean,2)}")
plt.show()

# plot the values on the nodes at a particular timestep
timestep = 5
activity_at_timestep = activity[:, timestep]

fig, ax = plt.subplots()
nx.draw_networkx_nodes(
    G,
    pos=pos_spring,
    ax=ax,
    nodelist=list(range(N)),
    node_size=np.sqrt(N),
    node_color=activity_at_timestep,
)

nx.draw_networkx_edges(
    G, pos=pos_spring, ax=ax, nodelist=list(range(N)), alpha=300 / G.size()
)
ax.axis("off")
plt.show()
