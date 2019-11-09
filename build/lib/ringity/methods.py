import matplotlib.pyplot as plt
import numpy as np



def get_coordinates(dgm,
                    infinity = True):

    if len(dgm) == 0:
        return 0, 0

    z = [(k.birth, k.death) for k in dgm if k.death != np.inf]
    x_max = max(k.birth for k in dgm)
    if z:
        x, y = map(list,zip(*z))
        y_max = max(max(y), x_max)
    else:
        x, y = [], []
        y_max = x_max


    z2 = [(k.birth, y_max+1) for k in dgm if k.death == np.inf]

    if z2 and infinity:
        x2, y2 = map(list,zip(*z2))
    else:
        x2, y2 = [], []
        for x_tmp, y_tmp in z2:
            x.append(x_tmp)
            y.append(y_tmp)

    return x, y , x2, y2



def draw_diagram(dgm,
                 line=None,
                 title=True,
                 infinity=True,
                 ax = None,
                 verbose = False):
    """Plot persistence diagram."""

    x,y,x2,y2 = get_coordinates(dgm, infinity=infinity)

    if not y:
        y_max = max(x + x2)
    else:
        y_max = max(max(y), max(x + x2))

    if x2 and infinity:
        fig, (ax2, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios':[1, 10]})
    else:
        if ax is None:
            fig, ax = plt.subplots()

    if line is None:
        line = [0, y_max]


    ax.plot(x,y, '*', line, line)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.set_xlabel('Time of Birth')
    ax.set_ylabel('Time of Death')

    if x2 and infinity:
        ax.set_ylim(0, y_max)
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-0.015, 0.015), (1, 1), **kwargs)  # top of y-axis

        ax2.plot(x2,y2, '*');
        ax2.set_ylim(y_max+0.5, y_max+1.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_yticks([])
        ax2.set_ylabel('inf', rotation=0, position=(0,0.25))

    if type(title) == str:
        pass
        #ax2.title(title)
    else:
        pass
        #ax2.title('max. persistence = %.3f'%p_max)

    if verbose:
        return fig

def _yes_or_no(answer):
    answer = answer.lower()
    while answer not in {*_ringity_parameters['yes'], *_ringity_parameters['no']}:
        answer = input("Please provide a readable input, e.g. 'y' or 'n'! ")

    if answer in _ringity_parameters['yes']:
        return True
    elif answer in _ringity_parameters['no']:
        return False
    else:
        assert False, _assertion_statement


def dict2numpy(bb, fill=np.inf):
    """
    Returns a numpy array corresponding to the given dictionary and fills the
    vacancies with fill.
    """

    nodes = bb.keys()
    N = len(nodes)
    D = np.full((N, N), np.inf)

    for v in nodes:
        for w in bb[v]:
            D[v,w] = bb[v][w]
    return D


def dict2ddict(bb):
    sources, targets = zip(*bb.keys())
    nodes = {*set(sources), *set(targets)}
    new_bb = {v:{
                **{s:bb[(s,v)] for s in sources if (s,v) in bb},
                **{t:bb[(v,t)] for t in targets if (v,t) in bb},
                } for v in nodes}

    return new_bb


def ddict2dict(bb):
    edges = set.union(
            *( {frozenset({s,t}) for s in bb[t].keys() if s!=t}
                                            for t in bb.keys() )
                      )
    new_bb = {(s,t):bb[s][t] for (s,t) in edges}
    return new_bb
