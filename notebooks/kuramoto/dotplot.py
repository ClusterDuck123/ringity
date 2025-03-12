import matplotlib.pyplot as plt
import numpy as np
import ringity as rng
from itertools import product
import uuid
import os

# ------------------------------------------------------------------------
# Global settings / theme
# ------------------------------------------------------------------------
rng.set_theme()

# Constants for dot_plot
DOT_SIZE = 1500
SCATTER_VMIN = 0.6
SCATTER_VMAX = 1.0


def dot_plot(
    coherences: np.ndarray,
    fractions: np.ndarray,
    scale_by: str = "area",
    ax: plt.Axes = None,
    with_fraction_labels: bool = True,
    with_coherence_labels: bool = True,
) -> plt.Axes:
    """
    Draw a 'dot plot' where each point has two attributes:
      - A color corresponding to the coherence value.
      - A size (relative area or length) corresponding to the fraction value.

    Parameters
    ----------
    coherences : np.ndarray
        2D array of coherence values (must match shape of fractions).
    fractions : np.ndarray
        2D array of fraction values (must match shape of coherences).
    scale_by : {'area', 'length'}, optional
        How to scale the marker size by the fraction value:
          - 'area': scale marker's area by fraction (default)
          - 'length': scale marker's diameter by fraction
    ax : plt.Axes, optional
        Matplotlib axes object on which to draw the plot. If not provided,
        a new figure and axes are created.
    with_fraction_labels : bool, optional
        If True, displays fraction labels below each marker.
    with_coherence_labels : bool, optional
        If True, displays coherence labels centered on each marker.

    Returns
    -------
    plt.Axes
        The axes on which the plot is drawn.
    """
    assert coherences.shape == fractions.shape, (
        "coherences and fractions must have the same shape."
    )

    # Validate 'scale_by' input
    if scale_by not in {"area", "length"}:
        raise ValueError("scale_by must be one of {'area', 'length'}.")

    # Set up axes
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, zorder=0)

    n_rows, n_cols = coherences.shape

    for row_idx, col_idx in product(range(n_rows), range(n_cols)):
        # Reverse indexing from the original code
        curr_cohe = coherences[-row_idx - 1, col_idx]
        curr_frac = fractions[-row_idx - 1, col_idx]

        # Determine how to scale the circle
        if scale_by == "area":
            # fraction scales area directly
            size_multiplier = curr_frac
        else:  # scale_by == "length"
            # fraction scales radius
            size_multiplier = curr_frac ** 2

        # Draw the outer white circle
        ax.scatter(
            col_idx, row_idx,
            s=DOT_SIZE,
            c="white",
            cmap="none",
            linewidth=1,
            edgecolors="black",
            zorder=2,
        )

        # Draw the scaled circle based on fraction
        ax.scatter(
            col_idx, row_idx,
            s=DOT_SIZE * size_multiplier,
            c=curr_cohe,
            cmap="cividis",
            linewidth=1,
            edgecolors="black",
            zorder=3,
            vmin=SCATTER_VMIN,
            vmax=SCATTER_VMAX,
        )

        # Optionally add text labels
        if with_fraction_labels:
            ax.text(
                col_idx,
                row_idx - 0.35,
                f"{round(100 * curr_frac)}%",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            )

        if with_coherence_labels:
            ax.annotate(
                f"{curr_cohe:.2f}",
                (col_idx, row_idx),
                ha="center",
                va="center",
                color="black",
            )

    return ax


def main():
    # --------------------------------------------------------------------
    # Example data
    # --------------------------------------------------------------------
    coherences = np.array([
        [0.00, 0.05, 0.11, 0.16],
        [0.21, 0.26, 0.32, 0.37],
        [0.42, 0.47, 0.53, 0.58],
        [0.63, 0.68, 0.74, 0.79],
        [0.84, 0.89, 0.95, 1.00],
    ])

    fractions = np.array([
        [0.00, 0.05, 0.11, 0.16],
        [0.21, 0.26, 0.32, 0.37],
        [0.42, 0.47, 0.53, 0.58],
        [0.63, 0.68, 0.74, 0.79],
        [0.84, 0.89, 0.95, 1.00],
    ])

    # --------------------------------------------------------------------
    # Create figure and draw the plot
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    dot_plot(
        coherences,
        fractions,
        scale_by="area",  # or "length"
        ax=ax,
        with_fraction_labels=True,
        with_coherence_labels=True,
    )

    # Add colorbar for coherence
    # We pass dummy scatter args because we only need the colorbar
    dummy_scatter = ax.scatter([], [], c=[], cmap="cividis", vmin=0.5, vmax=1.0)
    cbar = fig.colorbar(dummy_scatter, ax=ax)
    cbar.set_label(r"Avg. Coherence ($\langle R_{\infty} \rangle$)")

    # Axis settings
    n_rows, n_cols = coherences.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.75, n_rows - 0.5)
    ax.set_xticks(
        range(n_cols),
        map("{:.2f}".format, np.linspace(0.10, 0.25, n_cols))
    )
    ax.set_yticks(
        range(n_rows),
        map("{:.2f}".format, np.linspace(0.80, 1.00, n_rows))
    )
    ax.set_xlabel(r"Response Length ($r$)")
    ax.set_ylabel(r"Delay Parameter ($\beta$)")
    ax.set_title("Synchronization Behaviour", size=24)

    plt.show()


if __name__ == "__main__":
    main()