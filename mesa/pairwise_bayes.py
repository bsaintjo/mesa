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
import arviz
import tqdm


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


def full_model(df):
    """Applies an Bayesian (multilevel) model to analyze contingency tables.
    Takes a pandas DataFrame that contains the count and total data for every
    comparison."""
    totals = df.left_total.values
    counts = df.left_count.values
    n_models = len(counts)

    with pm.Model():
        a_bar = pm.Normal("a_bar", mu=0.0, sd=1.5)
        sigma = pm.Exponential("sigma", lam=1)
        alpha = pm.Normal("alpha", mu=a_bar, sd=sigma, shape=n_models)
        theta = pm.Deterministic("theta", pm.math.invlogit(alpha))
        _ = pm.Binomial(
            "count",
            p=theta,
            n=totals,
            observed=counts,
        )
        trace = pm.fit(
            20_000,
            method="advi",
            callbacks=[pm.variational.callbacks.CheckParametersConvergence()])
    posterior = trace.sample(5000)
    df["diff"] = posterior["theta"].mean(axis=0)


def simple_model(df):
    """Applies a simple 'bayesian' fishers test/contingency table analysis. It
    uses two beta-binomial model to get the posterior distribution of the
    probability of a given count and total. The model also adds a few relevant
    summaries of the posterior distribution:
        diff: expected difference in PSI
        hpd_05.5: left end of the 89% highest posterior density interval
        hpd_94.5: right end of the 89% highest posterior density interval
        Pr(|diff| > 0.2): Proportion of the posterior distribution that is
            between -0.2 and 0.2
    """
    nrows = len(df.incl_left)
    left_theta = np.random.beta(df.left_count.values + 1,
                                (df.left_total.values - df.left_count.values) +
                                1,
                                size=(1000, nrows))
    right_theta = np.random.beta(
        df.right_count.values + 1,
        (df.right_total.values - df.right_count.values) + 1,
        size=(1000, nrows))
    df["diff"] = (left_theta - right_theta).mean(axis=0)
    diff_hpd = arviz.hpd(left_theta - right_theta, credible_interval=0.89)
    df["hpd_05.5"] = diff_hpd[:, 0]
    df["hpd_94.5"] = diff_hpd[:, 1]
    df["Pr(|diff|<0.2)"] = (np.absolute(left_theta - right_theta) < 0.2).mean(
        axis=0)
    float_format_dict = {
        "diff": 3,
        "Pr(|diff|<0)": 3,
        "hpd_05.5": 3,
        "hpd_94.5": 3
    }
    df = df.round(float_format_dict)


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
    parser.add_argument(
        "-m",
        "--method",
        choices=["simple", "full"],
        default="simple",
        help="(default: simple) Method to use for statistical modeling, simple"
        " is more efficient for large datasets, full applies a more powerful"
        " but slower model (warning: full is currently experimental and not "
        "recommended for use)"
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
    comps_to_idx = {comp: idx for idx, comp in enumerate(comps)}

    inclusion_total_counts = []
    for n, vals in tqdm(enumerate(matrix), desc="Loading data"):
        event_id = rows[n]
        mxes = matrix[np.isin(rows, clusters[event_id])]

        incl = vals
        excl = np.sum(mxes, axis=0)

        for i in comps:
            comp_idx = comps_to_idx[i]
            left, right = i
            left_count = incl[left]
            right_count = incl[right]
            left_total = incl[left] + excl[left]
            right_total = incl[right] + incl[right]
            data_row = (event_id, comp_idx, left_count, left_total,
                        right_count, right_total)
            inclusion_total_counts.append(data_row)

    jxn_counts = pd.DataFrame(
        inclusion_total_counts,
        columns=[
            "event_id",
            "index",
            "left_count",
            "left_total",
            "right_count",
            "right_total",
        ],
    )
    jxn_counts.dropna(axis=0)
    jxn_counts = jxn_counts[(jxn_counts.left_total != 0)
                            & (jxn_counts.right_total != 0)]

    if args.method == "full":
        simple_model(jxn_counts)
    else:
        full_model(jxn_counts)
    jxn_counts.to_csv(args.output, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser)
    args = parser.parse_args()
    run_with(args)


if __name__ == "__main__":
    main()
