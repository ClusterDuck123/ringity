import matplotlib.pyplot as plt

"""TODO: Rewrite this part to a proper matplotlib style sheet API."""

CEMM_COL1 = (  0/255,  85/255, 100/255)
CEMM_COL2 = (  0/255, 140/255, 160/255)
CEMM_COL3 = ( 64/255, 185/255, 212/255)
CEMM_COL4 = (212/255, 236/255, 242/255)

DARK_CEMM_COL1 = (0/255, 43/255, 50/255)
BAR_COL = (0.639, 0.639, 0.639)

def set():
    plt.rc('axes', labelsize=24, titlesize=28)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)

def ax_setup(ax, labelsize=24):

    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    ax.patch.set_alpha(0)

    ax.spines['left'].set_linewidth(2.5)
    ax.spines['left'].set_color('k')

    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['bottom'].set_color('k')

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)