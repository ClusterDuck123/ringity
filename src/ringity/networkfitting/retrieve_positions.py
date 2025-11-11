import numpy as np
import tqdm
import networkx as nx
from scipy.ndimage import convolve
import scipy.stats


def circular_mean(theta):
    x = np.mean(np.cos(theta))
    y = np.mean(np.sin(theta))
    return np.mod(np.arctan2(y, x), 2 * np.pi)


def estimate_std(array_of_values, window_size=30):
    """
    Estimate the standard deviation over a sliding window for a given array of values.

    This function computes an estimate of the standard deviation using a sliding window
    approach. For each window of data, it calculates the mean and the mean of the squares,
    then uses these to estimate the variance, and hence the standard deviation.

    Args:
        array_of_values (list of numpy arrays): A list where each element is a numpy array
                                                of sample values for which the standard
                                                deviation is to be estimated.
        window_size (int, optional): The size of the sliding window over which to compute
                                     the standard deviation. Default is 30.

    Returns:
        numpy.ndarray: An array containing the estimated standard deviations for each window.

    Notes:
        - The convolution operation is used to apply the sliding window over the input arrays.
        - The function applies a correction factor to account for the degrees of freedom
          in the variance estimate.
        - The convolution mode is set to "wrap", meaning the window wraps around the array
          edges, effectively treating the array as circular.

    Example:
        >>> import numpy as np
        >>> data = [np.random.randn(100) for _ in range(10)]
        >>> std_estimates = estimate_std(data, window_size=20)
        >>> print(std_estimates)
    """

    count = []
    sum_value = []
    sum_square = []
    for samples in array_of_values:
        count.append(len(samples))
        sum_value.append(sum(samples))
        sum_square.append(sum(samples**2))

    smooth_count = convolve(count, np.ones(window_size), mode="wrap")
    smooth_sum_value = convolve(sum_value, np.ones(window_size), mode="wrap")
    smooth_sum_square = convolve(sum_square, np.ones(window_size), mode="wrap")

    correction = (smooth_count - 1) / smooth_count
    return np.sqrt(
        correction
        * (smooth_sum_square / smooth_count - (smooth_sum_value / smooth_count) ** 2)
    )


def reconstruct_icdf(u_samples, ipdf_values):
    """
    Reconstruct the inverse cumulative distribution function (ICDF) from samples.

    Parameters:
    u_samples (numpy.ndarray): Array of sample points from the distribution.
    ipdf_values (numpy.ndarray): Array of interpolated probability density function values corresponding to u_samples.

    Returns:
    tuple: A tuple containing:
        - u_midpoints (numpy.ndarray): Array of midpoints for piecewise linear interpolation.
        - cumulative_icdf (numpy.ndarray): Array of cumulative ICDF values.
    """

    # Sort the sample points and corresponding IPDF values
    to_sorted_order = np.argsort(u_samples)
    u_sorted = u_samples[to_sorted_order]
    ipdf_sorted = ipdf_values[to_sorted_order]

    # Calculate midpoints between sorted sample points for piecewise linear interpolation
    u_midpoints = (u_sorted[:-1] + u_sorted[1:]) / 2
    # Add 0 at the beginning and 1 at the end to cover the entire range [0, 1]
    u_midpoints = np.array([0] + list(u_midpoints) + [1])

    # Calculate the widths of each interval defined by the midpoints
    interval_widths = u_midpoints[1:] - u_midpoints[:-1]

    # Compute the cumulative ICDF by integrating the IPDF over each interval
    cumulative_icdf = np.array([0] + list((interval_widths * ipdf_sorted).cumsum()))

    return u_midpoints, cumulative_icdf


class PositionGraph:
    def __init__(self, G, network_type="positionless"):

        self.G = G
        self.edgelist = list(self.G.edges())
        self.nodelist = list(self.G.nodes())

        self.n_nodes = len(self.nodelist)

    def make_circular_spring_embedding(self, verbose=False, k=None):
        # copied and adapted from
        # https://networkx.org/documentation/networkx-1.11/_modules/networkx/drawing/layout.html#fruchterman_reingold_layout

        A = nx.to_numpy_array(self.G, nodelist=self.nodelist, weight="weight")
        dim = 2

        pos = 2 * (
            np.asarray(np.random.random((self.n_nodes, dim)), dtype=A.dtype) - 0.5
        )
        iterations = 300  # 500

        # K NEEDS ADJUSTMENT
        if k is None:

            k = 50 * 2 * np.pi / self.n_nodes
            if verbose:
                print(k)
        # the initial "temperature"  is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        # Calculate domain in case our fixed positions are bigger than 1x1
        t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        dt = t / float(iterations + 1)
        delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
        # the inscrutable (but fast) version
        # this is still O(V^2)
        # could use multilevel methods to speed this up significantly
        if verbose:
            iterator_ = tqdm.trange(iterations)
        else:
            iterator_ = range(iterations)

        for iteration in iterator_:

            if iteration == iterations:
                break

            # matrix of difference between points
            for i in range(pos.shape[1]):
                delta[:, :, i] = pos[:, i, None] - pos[:, i]

            # distance between points
            distance = np.sqrt((delta**2).sum(axis=-1))

            # enforce minimum distance of 0.01
            distance = np.where(distance < 0.01, 0.01, distance)

            # displacement "force"
            displacement = np.transpose(
                np.transpose(delta) * (k * k / distance**2 - A * distance / k)
            ).sum(axis=1)

            # update positions
            length = np.sqrt((displacement**2).sum(axis=1))
            length = np.where(length < 0.01, 0.01, length)
            delta_pos = np.transpose(np.transpose(displacement) * t / length)

            # pull the node back towards the unit circle
            radius = np.linalg.norm(pos, axis=1)
            alpha = (iteration / iterations) ** 2  # 0.5 also seems to work
            delta_pos += -alpha * pos * (radius - 1)[:, None]

            pos += delta_pos
            # cool temperature
            t -= dt

        self.pos = pos

        unadjusted_embedding_array = np.mod(np.arctan2(pos[:, 0], pos[:, 1]), 2 * np.pi)
        self.unadjusted_embedding_array = (
            2
            * np.pi
            * scipy.stats.rankdata(unadjusted_embedding_array)
            / len(unadjusted_embedding_array)
        )

        self.unadjusted_embedding_dict = dict(
            zip(self.nodelist, self.unadjusted_embedding_array)
        )

    def smooth_neighborhood_widths(self):

        self.nodelist_sorted_unadjusted = sorted(
            self.nodelist, key=self.unadjusted_embedding_dict.get
        )

        out = []
        for i in self.nodelist_sorted_unadjusted:
            # Calculate the rank-transformed value using the cumulative distribution function (cdf)
            # Generate neighboring values around x and calculate their rank-transformed values
            # add the position of i itsself, to ensure the list is not empty
            rank_neighbors = np.array(
                [self.unadjusted_embedding_dict[j] for j in self.G.neighbors(i)]
                + [self.unadjusted_embedding_dict[i]]
            )
            # rank_neighbors_center_pi = np.mod(rank_neighbors -np.mean(rank_neighbors) + np.pi, 2*np.pi)
            circ_mean = circular_mean(rank_neighbors)
            rank_neighbors_center_pi = np.array(
                [
                    np.mod(np.array(x) - circ_mean + np.pi, 2 * np.pi)
                    for x in rank_neighbors
                ]
            )
            out.append(rank_neighbors_center_pi)

        # get a smooth measure of the width of the neighborhoods
        # of each node, in the order around the ring
        N = len(self.nodelist)

        self.deleteme = out
        self.stds_smoothed = estimate_std(out, window_size=round(N / 20))

    def recenter_and_reorient_calcs(self):

        n_nodes = self.n_nodes
        stds_smoothed = self.stds_smoothed

        # the original embedding in generall picks an arbitrary for the zero point
        # look for the point where the density changes fastest, this will be our new zero point
        bin_width = round(len(self.nodelist) / 10)
        edge_filter = [-1] * bin_width + [1] * bin_width

        look_for_discontinuity_std = np.convolve(
            list(stds_smoothed) + list(stds_smoothed) + list(stds_smoothed),
            edge_filter,
            mode="same",
        )

        self.look_for_discontinuity_std = look_for_discontinuity_std

        # look_for_discontinuity_iqr = np.convolve(list(iqrs_smoothed)+list(iqrs_smoothed)+list(iqrs_smoothed),
        #                     edge_filter,
        #                     mode='same

        unadjusted_embedding_array = np.array(
            [self.unadjusted_embedding_dict[i] for i in self.nodelist_sorted_unadjusted]
        )

        # Calculate the postion of max change in neighborhood width
        i = np.argmax(np.abs(look_for_discontinuity_std[n_nodes : 2 * n_nodes]))
        self.embedding_cutoff = unadjusted_embedding_array[i]
        self.sign_change = np.sign(look_for_discontinuity_std[n_nodes : 2 * n_nodes][i])

    def reparametrize(self, stds_smoothed):
        # save a representation of the smoothed neighborhood width of each node
        # which willl be used as an estimate of
        ipdf_smooth_dict = dict(
            zip(self.nodelist_sorted_unadjusted, self.stds_smoothed)
        )
        self.nodelist_sorted = sorted(self.nodelist, key=self.embedding_dict.get)
        self.ipdf_smooth_adjusted = np.array(
            [1 / ipdf_smooth_dict[i] for i in self.nodelist_sorted]
        )

    def recenter_and_reorient(self, reparametrize=True, calculate=True):
        if calculate:
            self.recenter_and_reorient_calcs()

        # re-centre the distribution and re-orient if necessary
        self.embedding_dict = {
            i: np.mod(self.sign_change * (v - self.embedding_cutoff), 2 * np.pi)
            for i, v in self.unadjusted_embedding_dict.items()
        }

        if reparametrize:
            self.reparametrize(self.stds_smoothed)

    def reconstruct_positions(self):

        # apply the inverse cdf transformation to the embedding positions
        # using the estimate of the inverse (recipriocal) pdf derived previously
        embedding_positions = np.array(
            [self.embedding_dict[j] for j in self.nodelist_sorted]
        )
        u_midpoints, cumulative_icdf = reconstruct_icdf(
            embedding_positions / (2 * np.pi), self.ipdf_smooth_adjusted
        )

        # set the derived vaues as the beest guesss at the original positions
        cumulative_icdf = 2 * np.pi * cumulative_icdf / cumulative_icdf[-1]
        self.rpositions = dict(zip(self.nodelist_sorted, cumulative_icdf))
