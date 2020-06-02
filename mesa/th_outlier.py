#!/usr/bin/env python3
"""This script finds outlier PSI values by comparing against a set of provided
controls.
"""

import argparse
import numpy as np
import scipy.stats as sp
import pandas as pd


def add_parser(parser):
    parser.add_argument("--treatment", help="Treatment sample", required=True)
    parser.add_argument("--treatment-psi", required=True)
    parser.add_argument("--control-psi", required=True)
    parser.add_argument("-o",
                        "--output",
                        required=True,
                        help="Output file name")


def run_with(args):
    control_psi = pd.read_csv(args.control_psi, sep="\t")
    treatment_psi = pd.read_csv(args.treatment_psi, sep="\t")
    treatment = " " + args.treatment
    treatment_psi = treatment_psi[["cluster", treatment]]

    # Filter nans
    treatment_psi = treatment_psi.dropna(axis=0)

    # TODO thresh is 2 because cluster column guarantees atleast one non-nan
    # value per row, it would be better to fix this for the future
    control_psi = control_psi.dropna(thresh=2, axis=0)

    # Filter by clusters that are found in both
    treatment_psi = treatment_psi[treatment_psi["cluster"].isin(
        control_psi["cluster"])]
    control_psi = control_psi[control_psi["cluster"].isin(
        treatment_psi["cluster"])]
    only_psis = control_psi.drop(columns=["cluster"])

    iqrs = sp.iqr(only_psis, nan_policy="omit", axis=1)
    n_psis = np.count_nonzero(~np.isnan(only_psis), axis=1)

    # uplier eq: 75th percentile + (1.5 * IQR)
    upper_quantile = np.nanquantile(only_psis, 0.75, axis=1)
    uplier_threshold = upper_quantile + (1.5 * iqrs)

    # downlier eq: 25th percentile - (1.5 * IQR)
    lower_quantile = np.nanquantile(only_psis, 0.25, axis=1)
    downlier_threshold = lower_quantile - (1.5 * iqrs)

    final_df = treatment_psi["cluster"].str.split("-|:", expand=True)
    final_df = final_df.rename(columns={0: "chrom", 1: "start", 2: "end"})
    final_df[args.treatment + "_psi"] = treatment_psi[treatment]
    final_df["n_psis"] = n_psis

    final_df["dwnlier_thresh"] = downlier_threshold
    final_df["uplier_thresh"] = uplier_threshold
    final_df["dwnlier?"] = (treatment_psi[treatment] <
                            downlier_threshold).astype(int)
    final_df["uplier?"] = (treatment_psi[treatment] >
                           uplier_threshold).astype(int)

    final_df = final_df.round(3)
    final_df.to_csv(args.output, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_parser(parser)
    args = parser.parse_args()
    run_with(args)


if __name__ == "__main__":
    main()
