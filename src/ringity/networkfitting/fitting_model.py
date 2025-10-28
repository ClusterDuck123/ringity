import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import itertools as it


def trunc_exp_left(theta, m, lambda_, x_0):
    return m * lambda_ * np.exp(lambda_ * np.mod(-(theta - x_0), 2 * np.pi))


def slope_down(theta, c, w):
    temp = w - theta
    return c * 0.5 * (np.abs(temp) + temp)


class FitNetwork:
    """

    Use like this:

        fitter = FitNetwork(G,
                        rpositions
                       )

    fig1 = fitter.links_by_distance()
    fitter.fit_interaction_p_function()
    fig2 = fitter.neighbor_proportion()
    fig3 = fitter.draw_edge_edge_and_fit()


    """

    def __init__(self, G, rpositions):

        self.G = G
        self.nodelist = list(G.nodes())
        self.rpositions = rpositions

    def links_by_distance(self, color="k"):

        G_sub = self.G.subgraph(self.nodelist)
        out_n = []
        for i in self.nodelist:
            for j in G_sub.neighbors(i):

                euc_dist = np.abs(self.rpositions[j] - self.rpositions[i])
                angular_dist = np.min([euc_dist, 2 * np.pi - euc_dist])
                out_n.append(angular_dist)

        out_t = []
        for i, j in it.product(self.nodelist, repeat=2):

            euc_dist = np.abs(self.rpositions[j] - self.rpositions[i])
            angular_dist = np.min([euc_dist, 2 * np.pi - euc_dist])
            out_t.append(angular_dist)

        fig, ax = plt.subplots()

        self.counts_total, bins, _ = ax.hist(
            out_t, bins=50, color="#AAAAAA", density=False
        )
        self.counts_neighbors, bins, _ = ax.hist(out_n, bins=bins, color=color)

        self.fraction_neighbors = (self.counts_neighbors / self.counts_total,)

        self.midpoints = (bins[1:] + bins[:-1]) / 2

        ax.set_xlabel("Distance", fontsize=15)
        ax.set_ylabel("Count", fontsize=15)

        ax.axis("tight")

        return fig

    def fit_interaction_p_function(self):

        # [1:] because we want to avoid counting the absence of self-loops
        p, _ = scipy.optimize.curve_fit(
            slope_down,
            self.midpoints[1:],
            (self.counts_neighbors / self.counts_total)[1:],
            p0=None,
            sigma=None,
        )

        self.c, self.w = p

    def draw_true_neighbor_proportion(self, ax, color="k"):

        #
        ax.plot(
            self.midpoints,
            self.counts_neighbors / self.counts_total,
            c=color,
            linewidth=7,
        )
        ax.axis("tight")

    def draw_fit_neighbor_proportion(self, ax):

        theta_space = np.linspace(0, np.pi, 100)
        fspace = slope_down(theta_space, self.c, self.w)
        ax.plot(theta_space, fspace, c="#777777", linewidth=7)
        ax.axis("tight")

    def neighbor_proportion(self, color="k"):

        fig, ax = plt.subplots()

        self.draw_true_neighbor_proportion(ax)
        self.draw_fit_neighbor_proportion(ax)
        ax.set_ylim((0, None))
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(
            ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
        )

        ax.set_xlabel("Distance", fontsize=15)
        ax.set_ylabel("Proportion Connected", fontsize=15)

        ax.axis("tight")

        return fig

    def draw_edge_edge_and_fit(self, color="k"):

        fig, ax = plt.subplots()

        # get the positions of pairs of nodes at each end of an edge
        xy = np.array(
            [[self.rpositions[i], self.rpositions[j]] for i, j in self.G.edges()]
        )

        # swap the order of each pair of positions and
        # add back to the list to make the list symmetrical
        swap = np.vstack([xy[:, 1], xy[:, 0]]).T
        xy = np.vstack([xy, swap])

        # plot the points
        ax.scatter(xy[:, 0], xy[:, 1], c=color, s=50 / np.sqrt(len(self.G.edges())))

        ax.set_xlabel(r"position $i$", fontsize=15)
        ax.set_ylabel(r"position $j$", fontsize=15)

        # Having the ticks as fractions of pi reinforces that we are working on a cyclical
        # domain here.
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(
            ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
        )
        ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_yticklabels(
            ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
        )

        fig.suptitle(r"Edges $i\sim j$")

        # Draw the lines indicating the interaction width
        #  as obtained by the fitting procedure.
        plt.plot([0, 2 * np.pi], [self.w, 2 * np.pi + self.w], c="#777777", linewidth=4)
        plt.plot(
            [0, 2 * np.pi], [-self.w, 2 * np.pi - self.w], c="#777777", linewidth=4
        )

        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])

        ax.axis("tight")

        return fig


def position_hist(n_nodes, rpositions, n_bins=None, color="k"):

    if n_bins is None:
        n_bins = int(np.sqrt(n_nodes))

    fig, ax = plt.subplots()

    # get a histogram of point positions
    counts, bins, _ = ax.hist(
        rpositions.values(), bins=n_bins, color=color, density=True
    )

    ax.set_xlabel("Position", fontsize=15)
    ax.set_ylabel("Density", fontsize=15)

    # Having the ticks as fractions of pi reinforces
    # that we are working on a cyclical domain here.
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    )

    freqs = counts
    midpoints = (bins[1:] + bins[:-1]) / 2

    # find a best fit truncated exponential distribution to the
    # histogram of point positions. They also have a
    # spare x_0 parameter for truncation point
    p, _ = scipy.optimize.curve_fit(
        trunc_exp_left, midpoints, freqs, p0=None, sigma=None
    )

    m, lambda_, x_0 = p

    # plot the best fit curve.
    theta_space = np.linspace(0, 2 * np.pi, 100)
    fspace = trunc_exp_left(theta_space, m, lambda_, x_0)
    ax.plot(theta_space, fspace, c="#777777", linewidth=7)

    # Having the ticks as fractions of pi reinforces
    # that we are working on a cyclical domain here.
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    )

    ax.axis("tight")

    return fig, m, lambda_, x_0
