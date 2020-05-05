#!/usr/bin/env python3
"""This script uses a Bayesian (soon-to-be) hierarchical model to perform the
pairwise comparison between each sample.
This models compute the difference of the junction counts and credible
intervals based on a difference of beta-binomial model as mentioned in this
blog post by Gelman.
https://statmodeling.stat.columbia.edu/2009/10/13/what_is_the_bay/
The purpose of this model is provide a bayesian equivalent to fishers exact
test.
"""

import argparse
import numpy as np
import pandas as pd
import pymc3 as pm


def get_col_idx_from_arr(x, y):
    """
    takes in list of strings = x
    and finds list index in array = y
    """

    return np.nonzero(np.isin(y, x))


def get_cluster(fname):
    data = dict()
    with open(fname) as fin:
        for i in fin:
            left, right = i.rstrip().split()
            mxes = right.split(",")
            data[left] = mxes
    return data


def add_parser(parser):
    parser.add_argument(
        "--inclusionMESA",
        type=str,
        required=True,
        help="Compressed NPZ formatted Inclusion count matrix from quantMESA.",
    )
    parser.add_argument(
        "-c",
        "--clusters",
        type=str,
        required=True,
        help="Clusters table.",
    )
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        help="Output file name")
    parser.add_argument("-j",
                        "--n_cpus",
                        required=False,
                        default=None,
                        type=int,
                        help="Number of CPU cores, by default uses all cores")


def run_with(args):
    pmesa = args.inclusionMESA
    cmesa = args.clusters

    # load psi
    data = np.load(pmesa)
    clusters = get_cluster(cmesa)

    # table has 3 arrays, cols, rows and data
    cols, rows, matrix = data["cols"], data["rows"], data["data"]

    comparisons = set()
    for i, v in enumerate(cols):
        for j, v in enumerate(cols):
            if i == j:
                continue
            comparisons.add(tuple(sorted([i, j])))

    # do the math
    # comps = list(comparisons)
    # comps_to_idx = {comp: idx for idx, comp in enumerate(comps)}

    inclusion_total_counts = []
    for n, vals in enumerate(matrix):
        event_id = rows[n]
        mxes = matrix[np.isin(rows, clusters[event_id])]

        incl = vals
        excl = np.sum(mxes, axis=0)

        for i in range(len(cols)):
            # comp_idx = comps_to_idx[i]
            left = i
            left_total = incl[left] + excl[left]
            data_row = (event_id, left, incl[left], left_total)
            inclusion_total_counts.append(data_row)

    jxn_counts = pd.DataFrame(
        inclusion_total_counts,
        columns=[
            "event_id",
            "index",
            "inclusion",
            "total",
        ],
    )
    jxn_counts.dropna(axis=0)
    jxn_counts = jxn_counts[(jxn_counts.inclusion != 0)
                            & (jxn_counts.total != 0)]

    # nrows = len(jxn_counts.incl_left)
    # left_theta = np.random.beta(
    #     jxn_counts.incl_left.values + 1,
    #     (jxn_counts.left_total.values - jxn_counts.incl_left.values) + 1,
    #     size=(2000, nrows))
    # right_theta = np.random.beta(
    #     jxn_counts.incl_right.values + 1,
    #     (jxn_counts.right_total.values - jxn_counts.incl_right.values) + 1,
    #     size=(2000, nrows))
    # jxn_counts["diff"] = (left_theta - right_theta).mean(axis=0)
    # jxn_counts["Pr(diff > 0)"] = ((left_theta - right_theta) > 0.0).mean(
    #     axis=0)
    # float_format_dict = {"diff": 3, "Pr(diff > 0)": 3}
    # jxn_counts = jxn_counts.round(float_format_dict)
    totals = jxn_counts.total.values
    incl_counts = jxn_counts.inclusion.values
    n_models = len(incl_counts)

    with pm.Model():
        theta = pm.Beta("theta", alpha=1.0, beta=1.0, shape=n_models)
        _ = pm.Binomial(
            "count",
            p=theta,
            n=totals,
            observed=incl_counts,
        )
        trace = pm.sample()
        # pm.backends.text.dump("trace.sav")
        posterior = pm.sample_posterior_predictive(
            trace,
            samples=2000,
            var_names=["theta"])
    # jxn_counts["left_Epsi"] = (
    #     (posterior["left"].mean(axis=0) / jxn_counts.left_total) * 100)
    # jxn_counts["right_Epsi"] = (
    #     (posterior["right"].mean(axis=0) / jxn_counts.right_total) * 100)
    # jxn_counts["diff"] = (posterior["left_prob"] -
    #                       posterior["left_prob"]).mean(axis=0)
    # jxn_counts["Pr(diff > 0)"] = (
    #    (posterior["left_prob"] - posterior["right_prob"]) > 0.0).mean(axis=0)
    jxn_counts["theta"] = posterior["theta"].mean(axis=0)

    jxn_counts.to_csv(args.output, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser)
    args = parser.parse_args()
    run_with(args)


if __name__ == "__main__":
    main()
