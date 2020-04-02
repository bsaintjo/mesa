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
            data[left] = mxes + [left]
    return data


def add_parser(parser):
    parser.add_argument(
        "--inclusionMESA",
        type=str,
        required=True,
        help="Compressed NPZ formatted Inclusion count matrix from quantMESA.",
    )
    parser.add_argument(
        "-c", "--clusters", type=str, required=True, help="Clusters table.",
    )
    parser.add_argument(
        "--chi2",
        action="store_true",
        default=False,
        help="Use X^2 instead of fishers. Quicker, not as sensitive.",
    )
    parser.add_argument(
        "--no-correction",
        action="store_true",
        default=False,
        help="Output raw p-values instead of corrected ones. Correction is "
        "done via Benjamini-Hochberg",
    )


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
    comps = list(comparisons)

    inclusion_total_counts = []
    for n, vals in enumerate(matrix):
        event_id = rows[n]
        mxes = matrix[np.isin(rows, clusters[event_id])]

        inc = vals
        exc = np.sum(mxes, axis=0)

        for i in comps:
            left, right = i
            inc_total = inc[left] + inc[right]
            exc_total = exc[left] + exc[right]
            data_row = (event_id, i, inc[left], inc_total, exc[left],
                        exc_total)
            inclusion_total_counts.append(data_row)

    jxn_counts = pd.DataFrame(
        inclusion_total_counts,
        columns=[
            "event_id",
            "comparison",
            "incl_left",
            "incl_total",
            "excl_left",
            "excl_total",
        ],
    )
    with pm.Model():
        left = pm.BetaBinomial(
            "left",
            alpha=1.0,
            beta=1.0,
            n=jxn_counts.incl_total,
            observed=jxn_counts.incl_left,
        )
        right = pm.BetaBinomial(
            "right",
            alpha=1.0,
            beta=1.0,
            n=jxn_counts.excl_total,
            observed=jxn_counts.excl_total,
        )
        pm.Deterministic("diff", left - right)
        trace = pm.sample(1000)
        posterior = pm.sample_posterior_predictive(trace, samples=1000)
    jxn_counts["E(dpsi)"] = posterior["diff"].mean(axis=0)
    jxn_counts["Pr(-10 < dpsi < 10)"] = (
        (posterior["diff"] > -10) & (posterior["diff"] < 10)
    ).mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser)
    args = parser.parse_args()
    run_with(args)


if __name__ == "__main__":
    main()
