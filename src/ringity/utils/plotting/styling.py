import matplotlib as mpl

CEMM_COL1 = (  0/255,  85/255, 100/255)
CEMM_COL2 = (  0/255, 140/255, 160/255)
CEMM_COL3 = ( 64/255, 185/255, 212/255)
CEMM_COL4 = (212/255, 236/255, 242/255)

DARK_CEMM_COL1 = (0/255, 43/255, 50/255)
BAR_COL = (0.639, 0.639, 0.639)

def set_theme(style = 'ringity'):
    if style == 'ringity':
        mpl.rcParams['xtick.labelsize'] = 'xx-large'
        mpl.rcParams['ytick.labelsize'] = 'xx-large'

        mpl.rcParams['axes.linewidth'] = 2.5
        mpl.rcParams['axes.labelsize'] = 'xx-large'
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
    if style == 'default':
        mpl.rcdefaults()

def ax_setup(ax, labelsize=24):

    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    ax.patch.set_alpha(0)

    ax.spines['left'].set_linewidth(2.5)
    ax.spines['left'].set_color('k')

    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['bottom'].set_color('k')

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)