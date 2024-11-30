import main
import numpy as np
import ringity as rg
import os

import matplotlib.pyplot as plt


N = 500
density = 0.01

beta = np.random.rand()
r = 0.5 * np.random.rand()

parameters = {"N":N,"beta": beta, "r": r, "density": density}

G,positions = rg.network_model(
    N, rho=density, beta=beta, r=r, return_positions=True
)
G, positions = main.get_largest_component_with_positions(G, positions)

position_object,fitter = main.run_analysis(G)



topfolder = "test_network_model/"
subfolder = "_".join([f"{i}_{str(round(v,3)).replace(".","_")}" for i,v in parameters.items()])
folder = os.path.join(topfolder,subfolder)
os.makedirs(folder)


fig,ax = plt.subplots()
ax.scatter(positions,
           [position_object.rpositions[i] for i in position_object.nodelist],
           c="k")
ax.set_xlabel("True Positions")
ax.set_ylabel("Recovered Positions")
fig.savefig(f"{folder}/recover.png")

main.save_figures(folder,G,fitter,position_object)
