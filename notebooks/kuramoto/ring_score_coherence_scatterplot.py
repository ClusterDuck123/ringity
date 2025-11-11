import argparse
import pandas as pd
import ringity as rng
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

rng.set_theme()


def main(args):

    output_file = args.o

    df = pd.read_csv(args.i, index_col=0)
    print(df)
    print(df.columns)
    scatter_info = df.groupby("network_folder")[
        ["ring_score", "terminal_mean", "beta", "r", "c"]
    ].mean()

    r = scatter_info["r"]

    x = scipy.stats.rankdata(scatter_info["ring_score"])
    y = 1 - scatter_info["terminal_mean"]
    c = r

    unique_r = [0.1, 0.15, 0.2, 0.25]

    fig, ax = plt.subplots()

    m = [None] * 4

    c = ["#2D4452", "#20313C", "#132027", "#142935", "#102C3D"]

    for i in range(4):
        select = r == unique_r[i]

        m[i] = ax.scatter(x[select], y[select], c=c[i])
        res = scipy.stats.linregress(x[select], y[select])
        ax.plot(x[select], res.slope * x[select] + res.intercept, c[i], linewidth=3)

        print(unique_r[i], res.slope, res.pvalue)

    print(m)

    # # ax.set_xlim((0.7,1.0))
    ax.set_xlabel("rank ring score")
    ax.set_ylabel(r"1-Avg. Coherence ($1-\langle R_{\infty} \rangle$)")

    plt.legend(m, unique_r, title=r"$r$")

    fig.savefig(output_file, dpi=300, transparent=True, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--i", type=str, default="data/parameter_array_summary.csv", help=""
    )
    parser.add_argument(
        "--o", type=str, default="figures/ring_score_coherence_scatterplot.png", help=""
    )

    args = parser.parse_args()

    main(args)
