#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Aug 28 00:00:00 2026

@author: Andy.
"""

#%%

import numpy as np
import math
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import pandas as pd

import mesa
# print(mesa.__version__)
import networkx as nx
from enum import Enum

# %matplotlib inline
# %precision 2

import logging
logging.basicConfig(level = logging.INFO)

import elfi

#%% Harmonised British Household Panel Survey (BHPS): Waves 1-18, 1991-2009
# Wave 2, 4, 6, 8, 10, 12, 14, 16 & 18

bb_indresp = pd.read_csv("bb_indresp.csv",
                         usecols=["bb_age", "bb_netag_1", "bb_netag_2", "bb_netag_3"])

bd_indresp = pd.read_csv("bd_indresp.csv",
                         usecols=["bd_age", "bd_netag_1", "bd_netag_2", "bd_netag_3"])

bf_indresp = pd.read_csv("bf_indresp.csv",
                         usecols=["bf_age", "bf_netag_1", "bf_netag_2", "bf_netag_3"])

bh_indresp = pd.read_csv("bh_indresp.csv",
                         usecols=["bh_age", "bh_netag_1", "bh_netag_2", "bh_netag_3"])

bj_indresp = pd.read_csv("bj_indresp.csv",
                         usecols=["bj_age", "bj_netag_1", "bj_netag_2", "bj_netag_3"])

bl_indresp = pd.read_csv("bl_indresp.csv",
                         usecols=["bl_age", "bl_netag_1", "bl_netag_2", "bl_netag_3"])

bn_indresp = pd.read_csv("bn_indresp.csv",
                         usecols=["bn_age", "bn_netag_1", "bn_netag_2", "bn_netag_3"])

bp_indresp = pd.read_csv("bp_indresp.csv",
                         usecols=["bp_age", "bp_netag_1", "bp_netag_2", "bp_netag_3"])

br_indresp = pd.read_csv("br_indresp.csv",
                         usecols=["br_age", "br_netag_1", "br_netag_2", "br_netag_3"])

#%% BHPS Data Cleaning:

def bhps_clean_age(indresp, prefix):
    """
    Parameters
    ----------
    indresp : pandas.DataFrame; Dataset such as bb_indresp, etc.

    prefix : str; Column prefix, e.g. "bb" if columns are bb_age, bb_netag_1, etc.

    Returns
    -------
    pandas.DataFrame with renamed columns: age, netag_1, netag_2, netag_3.
    """
    
    # Rename wave-specific columns to common names:
    indresp_clean = indresp.rename(columns={
                                            f"{prefix}_age": "age",
                                            f"{prefix}_netag_1": "netag_1",
                                            f"{prefix}_netag_2": "netag_2",
                                            f"{prefix}_netag_3": "netag_3"
                                            }
                                   ).copy()

    netag_cols = ["netag_1", "netag_2", "netag_3"]
    
    # Keep only rows where age is between 16 and 100 inclusive:
    indresp_clean = indresp_clean[
                                  indresp_clean["age"].between(16, 100)
                                  ].reset_index(drop=True)
    
    # Drop rows where all three netag variables are negative:
    indresp_clean = indresp_clean[
                                  ~indresp_clean[netag_cols].lt(0).all(axis=1)
                                  ].reset_index(drop=True)
    
    # Replace remaining negative netag values with NaN:
    indresp_clean[netag_cols] = indresp_clean[netag_cols].mask(
                                                               indresp_clean[netag_cols] < 0,
                                                               np.nan
                                                               )
    
    # Add netag_num_d columns for the difference from netag_num to age:
    indresp_clean["netag_1_d"] = indresp_clean["netag_1"] - indresp_clean["age"]
    indresp_clean["netag_2_d"] = indresp_clean["netag_2"] - indresp_clean["age"]
    indresp_clean["netag_3_d"] = indresp_clean["netag_3"] - indresp_clean["age"]
    
    return indresp_clean

#%% 

indresp_bhps_2_age = bhps_clean_age(bb_indresp, "bb")
indresp_bhps_4_age = bhps_clean_age(bd_indresp, "bd")
indresp_bhps_6_age = bhps_clean_age(bf_indresp, "bf")
indresp_bhps_8_age = bhps_clean_age(bh_indresp, "bh")
indresp_bhps_10_age = bhps_clean_age(bj_indresp, "bj")
indresp_bhps_12_age = bhps_clean_age(bl_indresp, "bl")
indresp_bhps_14_age = bhps_clean_age(bn_indresp, "bn")
indresp_bhps_16_age = bhps_clean_age(bp_indresp, "bp")
indresp_bhps_18_age = bhps_clean_age(br_indresp, "br")

#%% UK Household Longitudinal Study: Understanding Society (UKHLS): Waves 1-15, 2009-2024
# Wave 3, 6 & 9

c_indresp = pd.read_csv("c_indresp.csv",
                        usecols=["c_age_dv", "c_closenum", "c_simage", "c_netag_1", "c_netag_2", "c_netag_3"])

f_indresp = pd.read_csv("f_indresp.csv",
                        usecols=["f_age_dv", "f_closenum", "f_simage", "f_netag_1", "f_netag_2", "f_netag_3"])

i_indresp = pd.read_csv("i_indresp.csv",
                        usecols=["i_age_dv", "i_closenum", "i_simage"])

#%% UKHLS Data Cleaning:

# print((c_indresp["c_age_dv"] >= 0).sum())
# print((f_indresp["f_age_dv"] >= 0).sum())
# print((i_indresp["i_age_dv"] >= 0).sum())

# 
def ukhls_clean_age(indresp, prefix):
    """
    Parameters
    ----------
    indresp : pandas.DataFrame; Dataset such as c_indresp, etc.

    prefix : str; Column prefix, e.g. "c" if columns are c_age, c_netag_1, etc.

    Returns
    -------
    pandas.DataFrame with renamed columns: age, netag_1, netag_2, netag_3.
    """
    
    # Extract the corresponding columns from the original dataset:
    indresp_clean = indresp[
                            [f"{prefix}_age_dv",
                             f"{prefix}_netag_1",
                             f"{prefix}_netag_2",
                             f"{prefix}_netag_3"
                             ]
                            ].copy()
    
    # Rename wave-specific columns to common names:
    indresp_clean = indresp_clean.rename(columns={
                                                  f"{prefix}_age_dv": "age",
                                                  f"{prefix}_netag_1": "netag_1",
                                                  f"{prefix}_netag_2": "netag_2",
                                                  f"{prefix}_netag_3": "netag_3"
                                                  }
                                         ).copy()

    netag_cols = ["netag_1", "netag_2", "netag_3"]
    
    # Keep only rows where age is between 16 and 100 inclusive:
    indresp_clean = indresp_clean[
                                  indresp_clean["age"].between(16, 100)
                                  ].reset_index(drop=True)
    
    # Drop rows where all three netag variables are negative:
    indresp_clean = indresp_clean[
                                  ~indresp_clean[netag_cols].lt(0).all(axis=1)
                                  ].reset_index(drop=True)
    
    # Replace remaining negative netag values with NaN:
    indresp_clean[netag_cols] = indresp_clean[netag_cols].mask(
                                                               indresp_clean[netag_cols] < 0,
                                                               np.nan
                                                               )
    
    # Add netag_num_d columns for the difference from netag_num to age:
    indresp_clean["netag_1_d"] = indresp_clean["netag_1"] - indresp_clean["age"]
    indresp_clean["netag_2_d"] = indresp_clean["netag_2"] - indresp_clean["age"]
    indresp_clean["netag_3_d"] = indresp_clean["netag_3"] - indresp_clean["age"]
    
    return indresp_clean

# 
def ukhls_clean_simage():
    
    return 

# 
def ukhls_clean_deg(indresp, prefix):
    """
    Parameters
    ----------
    indresp : pandas.DataFrame; Dataset such as c_indresp, etc.

    prefix : str; Column prefix, e.g. "c" if columns are c_age, c_netag_1, etc.

    Returns
    -------
    pandas.DataFrame with renamed columns: age, closenum.
    """
    
    # Extract the corresponding columns from the original dataset:
    indresp_clean = indresp[
                            [f"{prefix}_age_dv",
                             f"{prefix}_closenum"
                             ]
                            ].copy()
    
    # Rename wave-specific columns to common names:
    indresp_clean = indresp_clean.rename(columns={
                                                  f"{prefix}_age_dv": "age",
                                                  f"{prefix}_closenum": "closenum"
                                                  }
                                         ).copy()
    
    # Keep only rows where age is between 16 and 100 inclusive:
    indresp_clean = indresp_clean[
                                  indresp_clean["age"].between(16, 100)
                                  ].reset_index(drop=True)
    
    # Keep only rows where closenum is between 0 and 10 inclusive:
    indresp_clean = indresp_clean[
                                  indresp_clean["closenum"].between(0, 10)
                                  ].reset_index(drop=True)
    
    return indresp_clean

#%%

indresp_ukhls_3_age = ukhls_clean_age(c_indresp, "c")
indresp_ukhls_6_age = ukhls_clean_age(f_indresp, "f")

#%%

indresp_ukhls_3_deg = ukhls_clean_deg(c_indresp, "c")
indresp_ukhls_6_deg = ukhls_clean_deg(f_indresp, "f")
indresp_ukhls_9_deg = ukhls_clean_deg(i_indresp, "i")

#%%

# ----------

#%% Degree:

indresp_deg = pd.concat(
                        [
                         indresp_ukhls_3_deg.assign(edition = "ukhls wave 3"),
                         indresp_ukhls_6_deg.assign(edition = "ukhls wave 6"),
                         indresp_ukhls_9_deg.assign(edition = "ukhls wave 9")
                         ],
                        ignore_index = True
                        )
indresp_deg = indresp_deg[["edition"] + [col for col in indresp_deg.columns if col != "edition"]]

print(indresp_deg["age"].value_counts().sort_index())
print(indresp_deg["closenum"].mean())

#%% Degree Distributions:

deg_dist = (
            indresp_deg["closenum"]
            .value_counts(normalize=True)
            .sort_index()
            .rename_axis("degree")
            .reset_index(name="probability")
            )

print(deg_dist.to_string(index=False))
print((deg_dist["degree"] * deg_dist["probability"]).sum())


FHS_deg_dist = pd.DataFrame({
                             "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             "probability": [0.094, 0.254, 0.186, 0.138, 0.099,
                                             0.071, 0.049, 0.032, 0.022, 0.021, 0.034]
                             })

# print(FHS_deg_dist["probability"].sum())
print((FHS_deg_dist["degree"] * FHS_deg_dist["probability"]).sum())

#%% Degree Distribution by Age:

deg_dist_by_age = {
                   age: (
                         subdf["closenum"]
                         .value_counts(normalize=True)
                         .sort_index()
                         .rename_axis("degree")
                         .reset_index(name="probability")
                         )
                   for age, subdf in indresp_deg.groupby("age")
                   }

avg_deg_by_age = (
                  pd.DataFrame(
                               {
                                age: (dist["degree"] * dist["probability"]).sum()
                                for age, dist in deg_dist_by_age.items()
                                }.items(),
                               columns=["age", "average degree"]
                               )
                  .sort_values("age")
                  .reset_index(drop=True)
                  )

# print(avg_deg_by_age.to_string(index=False))

#%% Degree Distribution - Smoothing



#%%

#
plt.figure(figsize=(9, 6), dpi=1000)

plt.bar(
    avg_deg_by_age["age"],
    avg_deg_by_age["average degree"],
    color=plt.cm.cividis(
        mpl.colors.Normalize(
            vmin=avg_deg_by_age["average degree"].min(),
            vmax=avg_deg_by_age["average degree"].max()
        )(avg_deg_by_age["average degree"])
    ),
    edgecolor="black"
)

plt.xlabel("Age")
plt.ylabel("Average degree <k>")
plt.title("Average degree by age")

plt.xticks([16] + list(range(20, 101, 5)))

plt.colorbar(
    mpl.cm.ScalarMappable(
        cmap=plt.cm.cividis,
        norm=mpl.colors.Normalize(
            vmin=avg_deg_by_age["average degree"].min(),
            vmax=avg_deg_by_age["average degree"].max()
        )
    ),
    ax=plt.gca()
).set_label("Average degree <k>")

plt.tight_layout()
plt.show()

#
plt.figure(figsize = (9, 6), dpi = 1000)

plt.bar(
        deg_dist["degree"],
        deg_dist["probability"],
        label="Understanding Society: <k> = 4.2767",
        color="red",
        edgecolor="black",
        alpha=0.7
        )

plt.bar(
        FHS_deg_dist["degree"],
        FHS_deg_dist["probability"],
        label="Framingham Heart Study: <k> = 3.0140",
        color="gray",
        edgecolor="black",
        alpha=0.7
        )

plt.xlabel("Degree K")
plt.ylabel("P(k)")
plt.title("Degree Distribution Comparison")

plt.xticks(range(0, 11))

plt.legend()
plt.tight_layout()
plt.show()

#%%

# ----------

#%% Age:

indresp_age = pd.concat(
                        [
                         indresp_bhps_2_age.assign(edition = "bhps wave 2"),
                         indresp_bhps_4_age.assign(edition = "bhps wave 4"),
                         indresp_bhps_6_age.assign(edition = "bhps wave 6"),
                         indresp_bhps_8_age.assign(edition = "bhps wave 8"),
                         indresp_bhps_10_age.assign(edition = "bhps wave 10"),
                         indresp_bhps_12_age.assign(edition = "bhps wave 12"),
                         indresp_bhps_14_age.assign(edition = "bhps wave 14"),
                         indresp_bhps_16_age.assign(edition = "bhps wave 16"),
                         indresp_bhps_18_age.assign(edition = "bhps wave 18"),
                         indresp_ukhls_3_age.assign(edition = "ukhls wave 3"),
                         indresp_ukhls_6_age.assign(edition = "ukhls wave 6"),
                         ],
                        ignore_index = True
                        )
indresp_age = indresp_age[["edition"] + [col for col in indresp_age.columns if col != "edition"]]

print(indresp_age["age"].value_counts().sort_index())

# with pd.option_context("display.max_rows", None):
#     print(indresp_age["age"].value_counts().sort_index())

# print(indresp_age["age"].min())
# print(indresp_age["age"].max())
# print(indresp_age[indresp_age["age"] == 100])
    
# print(indresp_age["netag_1"].min()) # 0
# print(indresp_age["netag_1"].max()) # 100

# print(indresp_age["netag_2"].min()) # 0
# print(indresp_age["netag_2"].max()) # 100

# print(indresp_age["netag_3"].min()) # 0
# print(indresp_age["netag_3"].max()) # 103: ONLY 2

# print(indresp_age[indresp_age["netag_1"] == 0])
# print(indresp_age[indresp_age["netag_2"] == 0])
# print(indresp_age[indresp_age["netag_3"] == 0])

#%%

netag_by_age = (
                indresp_age
                .groupby("age")[["netag_1", "netag_2", "netag_3"]]
                .apply(lambda x: x.stack().dropna().tolist())
                .to_dict()
                )

netag_hist_by_age = {
                     ego_age: (
                               pd.Series(friend_ages)
                               .value_counts(normalize=False)
                               .sort_index()
                               .rename_axis("age")
                               .reset_index(name = "count")
                               )
                     for ego_age, friend_ages in netag_by_age.items()
                     }

netag_dist_by_age = {
                     ego_age: (
                               pd.Series(friend_ages)
                               .value_counts(normalize=True)
                               .sort_index()
                               .rename_axis("age")
                               .reset_index(name = "probability")
                               )
                     for ego_age, friend_ages in netag_by_age.items()
                     }

#%% Draft Smoothing:

def partial_smooth_from_hist(
    hist_df,
    min_age=0,
    max_age=100,
    bandwidth=1.0,
    missing_scale=1.0
    ):
    
    all_ages = np.arange(min_age, max_age + 1)

    # Clean and prepare histogram
    h = hist_df.copy()

    h["age"] = pd.to_numeric(h["age"], errors="coerce")
    h["count"] = pd.to_numeric(h["count"], errors="coerce")

    h = h.dropna(subset=["age", "count"])
    h["age"] = h["age"].round().astype(int)

    h = h[h["age"].between(min_age, max_age)]

    # Make sure we have all ages 0-100
    observed_counts = (
        h.groupby("age")["count"]
        .sum()
        .reindex(all_ages, fill_value=0)
        .sort_index()
        .astype(float)
    )

    if observed_counts.sum() == 0:
        raise ValueError("No valid age counts found.")

    # Gaussian smoothing from observed counts
    distance_matrix = all_ages[:, None] - all_ages[None, :]

    kernel = np.exp(
        -0.5 * (distance_matrix / bandwidth) ** 2
    )

    # Normalise each observed age's Gaussian contribution
    kernel = kernel / kernel.sum(axis=0, keepdims=True)

    smoothed_counts = kernel @ observed_counts.to_numpy()

    smoothed_counts = pd.Series(
        smoothed_counts,
        index=all_ages
    )

    # Partial smoothing:
    # if observed count > 0, keep observed count
    # if observed count == 0, use inferred smoothed count
    partial_counts = observed_counts.copy()

    missing_mask = observed_counts == 0

    partial_counts[missing_mask] = (
        missing_scale * smoothed_counts[missing_mask]
    )

    # Normalise after partial smoothing
    probability = partial_counts / partial_counts.sum()

    result = pd.DataFrame({
        "age": all_ages,
        "observed_count": observed_counts.values,
        "smoothed_count": smoothed_counts.values,
        "partial_count": partial_counts.values,
        "probability": probability.values,
        "was_observed": observed_counts.values > 0
    })

    return result

netag_dist_by_age_smoothed = {
                              ego_age: partial_smooth_from_hist(
                                                                hist_df,
                                                                min_age=0,
                                                                max_age=100,
                                                                bandwidth=0.5,
                                                                missing_scale=1.0
                                                                )
                              for ego_age, hist_df in netag_hist_by_age.items()
                              }

netag_dist_by_age_empirical = {
                               ego_age: (
                                         df[df["age"].between(16, 100)][["age", "probability"]]
                                         .assign(probability=lambda x: x["probability"] / x["probability"].sum())
                                         .reset_index(drop=True)
                                         )
                               for ego_age, df in netag_dist_by_age_smoothed.items()
                               if 16 <= ego_age <= 100
                               }

#%% Friend's Age Distribution - Smoothing



#%%

# ----------

#%% Network construction and Monte Carlo settings

MIN_AGE = 16
MAX_AGE = 100

AGE_VALUES = np.arange(MIN_AGE, MAX_AGE + 1)

NETWORK_SEED = 123

# Choose which degree distribution to use for network construction:
# Options: "FHS" or "UKHLS"
# DEGREE_DIST_FOR_NETWORK = "FHS"
DEGREE_DIST_FOR_NETWORK = "UKHLS"

# Friend-age plots for these ego ages
SELECTED_EGO_AGES = [16, 20, 25, 30, 40, 50, 60, 70, 80, 90]

#%% Load ONS age distribution

ons_age_dist = pd.read_excel(
    "ONS - UK Data for Population by Age (1974-2024).xlsx",
    index_col=0
)

ons_age_dist = (
    ons_age_dist.loc[1974]
    .rename_axis("age")
    .reset_index(name="probability")
)

#%% Helper functions for distributions

def clean_distribution(dist_df, value_col, min_value=None, max_value=None):

    out = dist_df[[value_col, "probability"]].copy()

    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out["probability"] = pd.to_numeric(out["probability"], errors="coerce")

    out = out.dropna()
    out[value_col] = out[value_col].astype(int)

    out = out[out["probability"] >= 0]

    if min_value is not None:
        out = out[out[value_col] >= min_value]

    if max_value is not None:
        out = out[out[value_col] <= max_value]

    out = (
        out.groupby(value_col, as_index=False)["probability"]
        .sum()
        .sort_values(value_col)
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(f"{value_col} distribution is empty after cleaning.")

    if out["probability"].sum() <= 0:
        raise ValueError(f"{value_col} distribution probabilities sum to zero.")

    out["probability"] = out["probability"] / out["probability"].sum()

    return out


def make_sequence_from_distribution(dist_df, value_col, N, rng):

    raw_counts = dist_df["probability"].to_numpy() * N

    counts = np.floor(raw_counts).astype(int)

    remainder = N - counts.sum()

    if remainder > 0:
        fractional_part = raw_counts - counts
        add_to = np.argsort(fractional_part)[-remainder:]
        counts[add_to] += 1

    sequence = np.repeat(
        dist_df[value_col].astype(int).to_numpy(),
        counts
    )

    rng.shuffle(sequence)

    assigned_dist = dist_df.copy()
    assigned_dist["assigned_count"] = counts
    assigned_dist["assigned_probability"] = counts / N

    return sequence, assigned_dist

#%% Choose age and degree distributions for network construction

network_age_dist = clean_distribution(
    ons_age_dist,
    value_col="age",
    min_value=MIN_AGE,
    max_value=MAX_AGE
)

if DEGREE_DIST_FOR_NETWORK == "FHS":
    network_degree_dist = FHS_deg_dist.copy()

elif DEGREE_DIST_FOR_NETWORK == "UKHLS":
    network_degree_dist = deg_dist.copy()

else:
    raise ValueError("DEGREE_DIST_FOR_NETWORK must be either 'FHS' or 'UKHLS'.")

network_degree_dist = clean_distribution(
    network_degree_dist,
    value_col="degree",
    min_value=0,
    max_value=None
)

print("Age distribution sum:", network_age_dist["probability"].sum())
print("Degree distribution sum:", network_degree_dist["probability"].sum())

#%% Base-network convergence settings

# NETWORK_SIZES_TO_TEST = [10000, 50000, 100000, 500000, 1000000]

NETWORK_SIZES_TO_TEST = [1000, 5000, 10000, 50000, 100000]

SELECTED_AGES_TO_CHECK = SELECTED_EGO_AGES

PLOT_DPI = 300

#%% Expected targets

# Expected degree distribution: P(k)
network_degree_target = (
    network_degree_dist
    .groupby("degree")["probability"]
    .sum()
    .sort_index()
)

network_degree_target = network_degree_target / network_degree_target.sum()

ALL_DEGREES = np.arange(
    int(network_degree_target.index.min()),
    int(network_degree_target.index.max()) + 1
)

network_degree_target = network_degree_target.reindex(
    ALL_DEGREES,
    fill_value=0
)

target_average_degree = np.sum(
    network_degree_target.index.to_numpy() * network_degree_target.to_numpy()
)

# Expected friend-age distribution: approximately the overall age distribution
network_age_target = (
    network_age_dist
    .groupby("age")["probability"]
    .sum()
    .reindex(AGE_VALUES, fill_value=0)
    .sort_index()
)

network_age_target = network_age_target / network_age_target.sum()

print("Target average degree:", target_average_degree)

#%% Construct base network for given N

def construct_base_network_for_N(N_test, seed):

    rng = np.random.default_rng(seed)

    node_ages, _ = make_sequence_from_distribution(
        network_age_dist,
        value_col="age",
        N=N_test,
        rng=rng
    )

    degree_sequence_array, _ = make_sequence_from_distribution(
        network_degree_dist,
        value_col="degree",
        N=N_test,
        rng=rng
    )

    degree_sequence_array = degree_sequence_array.astype(int)

    # Undirected graph requires even total degree
    if degree_sequence_array.sum() % 2 == 1:

        candidates = np.where(degree_sequence_array > 0)[0]

        if len(candidates) == 0:
            raise ValueError("Odd degree sum but no positive degree to adjust.")

        chosen = rng.choice(candidates)
        degree_sequence_array[chosen] -= 1

    degree_sequence = degree_sequence_array.tolist()

    if not nx.is_graphical(degree_sequence, method="eg"):
        raise ValueError(f"Degree sequence is not graphical for N = {N_test}.")

    G_test = nx.random_degree_sequence_graph(
        degree_sequence,
        tries=1000,
        seed=int(seed)
    )

    node_age_test = {
        node: int(age)
        for node, age in zip(sorted(G_test.nodes()), node_ages)
    }

    nx.set_node_attributes(G_test, node_age_test, "age")

    return G_test, node_age_test

#%% Functions for convergence diagnostics

def get_nodes_dataframe(G_test, node_age_test):

    return pd.DataFrame({
        "node": list(G_test.nodes()),
        "age": [node_age_test[node] for node in G_test.nodes()],
        "degree": [G_test.degree[node] for node in G_test.nodes()]
    })


def degree_distribution_by_selected_ages(nodes_df, N_test):

    rows = []

    for ego_age in SELECTED_AGES_TO_CHECK:

        sub = nodes_df[nodes_df["age"] == ego_age]

        counts = (
            sub["degree"]
            .value_counts()
            .reindex(ALL_DEGREES, fill_value=0)
            .sort_index()
        )

        if counts.sum() > 0:
            probs = counts / counts.sum()
        else:
            probs = counts.astype(float)

        for degree, probability in probs.items():

            rows.append({
                "N": N_test,
                "ego_age": ego_age,
                "degree": degree,
                "probability": probability,
                "n_nodes_age": len(sub)
            })

    return pd.DataFrame(rows)


def average_degree_by_selected_ages(nodes_df, N_test):

    rows = []

    for ego_age in SELECTED_AGES_TO_CHECK:

        sub = nodes_df[nodes_df["age"] == ego_age]

        rows.append({
            "N": N_test,
            "ego_age": ego_age,
            "average_degree": sub["degree"].mean() if len(sub) > 0 else np.nan,
            "n_nodes_age": len(sub)
        })

    return pd.DataFrame(rows)


def friend_age_distribution_by_selected_ages(G_test, node_age_test, N_test):

    age_to_idx_local = {
        age: i
        for i, age in enumerate(AGE_VALUES)
    }

    counts = np.zeros(
        (len(AGE_VALUES), len(AGE_VALUES)),
        dtype=float
    )

    for u, v in G_test.edges():

        age_u = int(node_age_test[u])
        age_v = int(node_age_test[v])

        i = age_to_idx_local.get(age_u)
        j = age_to_idx_local.get(age_v)

        if i is not None and j is not None:
            counts[i, j] += 1
            counts[j, i] += 1

    row_sums = counts.sum(axis=1, keepdims=True)

    probs = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0
    )

    rows = []

    for ego_age in SELECTED_AGES_TO_CHECK:

        i = age_to_idx_local[ego_age]

        for j, friend_age in enumerate(AGE_VALUES):

            rows.append({
                "N": N_test,
                "ego_age": ego_age,
                "friend_age": friend_age,
                "probability": probs[i, j],
                "n_ego_stubs": counts[i, :].sum()
            })

    return pd.DataFrame(rows)

#%% Generate base networks and calculate diagnostics

base_convergence_networks = {}

degree_dist_by_age_base_allN = []
avg_degree_by_age_base_allN = []
friend_age_dist_by_age_base_allN = []

for i, N_test in enumerate(NETWORK_SIZES_TO_TEST):

    print(f"Constructing base network for N = {N_test}")

    seed_i = NETWORK_SEED + 1000 + i

    G_test, node_age_test = construct_base_network_for_N(
        N_test=N_test,
        seed=seed_i
    )

    base_convergence_networks[N_test] = {
        "G": G_test,
        "node_age": node_age_test
    }

    nodes_df_test = get_nodes_dataframe(
        G_test,
        node_age_test
    )

    degree_dist_by_age_base_allN.append(
        degree_distribution_by_selected_ages(
            nodes_df_test,
            N_test
        )
    )

    avg_degree_by_age_base_allN.append(
        average_degree_by_selected_ages(
            nodes_df_test,
            N_test
        )
    )

    friend_age_dist_by_age_base_allN.append(
        friend_age_distribution_by_selected_ages(
            G_test,
            node_age_test,
            N_test
        )
    )


degree_dist_by_age_base_allN = pd.concat(
    degree_dist_by_age_base_allN,
    ignore_index=True
)

avg_degree_by_age_base_allN = pd.concat(
    avg_degree_by_age_base_allN,
    ignore_index=True
)

friend_age_dist_by_age_base_allN = pd.concat(
    friend_age_dist_by_age_base_allN,
    ignore_index=True
)

###############################################################################

#%% Degree distribution by age: each age, all N

for ego_age in SELECTED_AGES_TO_CHECK:

    plt.figure(figsize=(9, 6), dpi=PLOT_DPI)

    for N_test in NETWORK_SIZES_TO_TEST:

        sub = degree_dist_by_age_base_allN[
            (degree_dist_by_age_base_allN["N"] == N_test)
            & (degree_dist_by_age_base_allN["ego_age"] == ego_age)
        ]

        n_age = int(sub["n_nodes_age"].iloc[0]) if len(sub) > 0 else 0

        plt.plot(
            sub["degree"],
            sub["probability"],
            marker="o",
            linewidth=1.8,
            label=f"N={N_test}, n_age={n_age}"
        )

    plt.plot(
        network_degree_target.index,
        network_degree_target.values,
        color="black",
        linewidth=2.8,
        linestyle="--",
        label="Expected target P(k)"
    )

    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.title(f"Degree distribution for age {ego_age}: convergence to target")
    plt.xticks(ALL_DEGREES)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Degree distribution by age: each N, all selected ages

for N_test in NETWORK_SIZES_TO_TEST:

    plt.figure(figsize=(9, 6), dpi=PLOT_DPI)

    for ego_age in SELECTED_AGES_TO_CHECK:

        sub = degree_dist_by_age_base_allN[
            (degree_dist_by_age_base_allN["N"] == N_test)
            & (degree_dist_by_age_base_allN["ego_age"] == ego_age)
        ]

        n_age = int(sub["n_nodes_age"].iloc[0]) if len(sub) > 0 else 0

        plt.plot(
            sub["degree"],
            sub["probability"],
            marker="o",
            linewidth=1.5,
            label=f"age {ego_age}, n={n_age}"
        )

    plt.plot(
        network_degree_target.index,
        network_degree_target.values,
        color="black",
        linewidth=2.8,
        linestyle="--",
        label="Expected target P(k)"
    )

    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.title(f"Degree distribution by selected ages, N={N_test}")
    plt.xticks(ALL_DEGREES)
    plt.legend()
    plt.tight_layout()
    plt.show()

###############################################################################

#%% Average degree by age: each age, all N

for ego_age in SELECTED_AGES_TO_CHECK:

    sub = avg_degree_by_age_base_allN[
        avg_degree_by_age_base_allN["ego_age"] == ego_age
    ].sort_values("N")

    plt.figure(figsize=(9, 6), dpi=PLOT_DPI)

    plt.plot(
        sub["N"],
        sub["average_degree"],
        marker="o",
        linewidth=2,
        label=f"Base network age {ego_age}"
    )

    plt.axhline(
        target_average_degree,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label=f"Expected target = {target_average_degree:.3f}"
    )

    for _, row in sub.iterrows():
        plt.text(
            row["N"],
            row["average_degree"],
            f"n={int(row['n_nodes_age'])}",
            fontsize=8,
            ha="center",
            va="bottom"
        )

    plt.xscale("log")
    plt.xlabel("Network size N")
    plt.ylabel("Average degree")
    plt.title(f"Average degree for age {ego_age}: convergence to target")
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Average degree by age: each N, all selected ages

for N_test in NETWORK_SIZES_TO_TEST:

    sub = avg_degree_by_age_base_allN[
        avg_degree_by_age_base_allN["N"] == N_test
    ].copy()

    sub = sub.sort_values("ego_age")

    plt.figure(figsize=(9, 6), dpi=PLOT_DPI)

    plt.bar(
        sub["ego_age"],
        sub["average_degree"],
        color="steelblue",
        edgecolor="black",
        alpha=0.8,
        label="Base network"
    )

    plt.axhline(
        target_average_degree,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label=f"Expected target = {target_average_degree:.3f}"
    )

    plt.xlabel("Age")
    plt.ylabel("Average degree")
    plt.title(f"Average degree by selected ages, N={N_test}")
    plt.xticks(SELECTED_AGES_TO_CHECK)
    plt.legend()
    plt.tight_layout()
    plt.show()

###############################################################################

#%% Friend-age distribution by ego age: each ego age, all N

for ego_age in SELECTED_AGES_TO_CHECK:

    plt.figure(figsize=(9, 6), dpi=PLOT_DPI)

    for N_test in NETWORK_SIZES_TO_TEST:

        sub = friend_age_dist_by_age_base_allN[
            (friend_age_dist_by_age_base_allN["N"] == N_test)
            & (friend_age_dist_by_age_base_allN["ego_age"] == ego_age)
        ]

        n_stubs = int(sub["n_ego_stubs"].iloc[0]) if len(sub) > 0 else 0

        plt.plot(
            sub["friend_age"],
            sub["probability"],
            linewidth=1.5,
            label=f"N={N_test}, stubs={n_stubs}"
        )

    plt.plot(
        network_age_target.index,
        network_age_target.values,
        color="black",
        linewidth=2.8,
        linestyle="--",
        label="Expected target age distribution"
    )

    plt.xlabel("Friend age")
    plt.ylabel("Probability")
    plt.title(f"Friend-age distribution for ego age {ego_age}: convergence to target")
    plt.xticks([16] + list(range(20, 101, 10)))
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Friend-age distribution by ego age: each N, all selected ego ages

for N_test in NETWORK_SIZES_TO_TEST:

    plt.figure(figsize=(9, 6), dpi=PLOT_DPI)

    for ego_age in SELECTED_AGES_TO_CHECK:

        sub = friend_age_dist_by_age_base_allN[
            (friend_age_dist_by_age_base_allN["N"] == N_test)
            & (friend_age_dist_by_age_base_allN["ego_age"] == ego_age)
        ]

        n_stubs = int(sub["n_ego_stubs"].iloc[0]) if len(sub) > 0 else 0

        plt.plot(
            sub["friend_age"],
            sub["probability"],
            linewidth=1.4,
            label=f"ego age {ego_age}, stubs={n_stubs}"
        )

    plt.plot(
        network_age_target.index,
        network_age_target.values,
        color="black",
        linewidth=2.8,
        linestyle="--",
        label="Expected target age distribution"
    )

    plt.xlabel("Friend age")
    plt.ylabel("Probability")
    plt.title(f"Friend-age distribution by selected ego ages, N={N_test}")
    plt.xticks([16] + list(range(20, 101, 10)))
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Numerical convergence errors

degree_target_df = (
    network_degree_target
    .rename_axis("degree")
    .reset_index(name="target_probability")
)

degree_dist_error_rows = []

for (N_test, ego_age), sub in (
    degree_dist_by_age_base_allN
    .merge(degree_target_df, on="degree", how="left")
    .groupby(["N", "ego_age"])
):

    degree_dist_error_rows.append({
        "N": N_test,
        "ego_age": ego_age,
        "n_nodes_age": sub["n_nodes_age"].iloc[0],
        "mae": np.mean(np.abs(sub["probability"] - sub["target_probability"])),
        "mse": np.mean((sub["probability"] - sub["target_probability"]) ** 2),
        "tv": 0.5 * np.abs(sub["probability"] - sub["target_probability"]).sum()
    })

degree_dist_error_by_age_N = pd.DataFrame(degree_dist_error_rows)


avg_degree_error_by_age_N = avg_degree_by_age_base_allN.copy()

avg_degree_error_by_age_N["target_average_degree"] = target_average_degree

avg_degree_error_by_age_N["absolute_error"] = (
    avg_degree_error_by_age_N["average_degree"]
    - avg_degree_error_by_age_N["target_average_degree"]
).abs()


age_target_df = (
    network_age_target
    .rename_axis("friend_age")
    .reset_index(name="target_probability")
)

friend_age_error_rows = []

for (N_test, ego_age), sub in (
    friend_age_dist_by_age_base_allN
    .merge(age_target_df, on="friend_age", how="left")
    .groupby(["N", "ego_age"])
):

    friend_age_error_rows.append({
        "N": N_test,
        "ego_age": ego_age,
        "n_ego_stubs": sub["n_ego_stubs"].iloc[0],
        "mae": np.mean(np.abs(sub["probability"] - sub["target_probability"])),
        "mse": np.mean((sub["probability"] - sub["target_probability"]) ** 2),
        "tv": 0.5 * np.abs(sub["probability"] - sub["target_probability"]).sum()
    })

friend_age_error_by_age_N = pd.DataFrame(friend_age_error_rows)

print("\nDegree distribution convergence errors:")
print(degree_dist_error_by_age_N.to_string(index=False))

print("\nAverage degree convergence errors:")
print(avg_degree_error_by_age_N.to_string(index=False))

print("\nFriend-age distribution convergence errors:")
print(friend_age_error_by_age_N.to_string(index=False))

#%% Store convergence outputs

base_convergence_outputs = {
    "base_networks": base_convergence_networks,
    "degree_dist_by_age": degree_dist_by_age_base_allN,
    "avg_degree_by_age": avg_degree_by_age_base_allN,
    "friend_age_dist_by_age": friend_age_dist_by_age_base_allN,
    "degree_dist_error_by_age_N": degree_dist_error_by_age_N,
    "avg_degree_error_by_age_N": avg_degree_error_by_age_N,
    "friend_age_error_by_age_N": friend_age_error_by_age_N,
    "network_degree_target": network_degree_target,
    "target_average_degree": target_average_degree,
    "network_age_target": network_age_target
}

#%%

# Degree distribution by age   -> network_degree_target
# Average degree by age        -> target_average_degree
# Friend-age distribution      -> network_age_target












