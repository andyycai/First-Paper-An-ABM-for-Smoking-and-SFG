#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jul 24 00:00:00 2026

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

N = 10000

NETWORK_SEED = 123
MONTE_CARLO_SEED = 123

N_ITER = 2_000_000
PRINT_EVERY = 1_000

# Choose which degree distribution to use for network construction:
# Options: "FHS" or "UKHLS"
DEGREE_DIST_FOR_NETWORK = "FHS"

# Loss functions to test
LOSS_FUNCTIONS_TO_RUN = ["mse", "mae", "tv", "cdf_mae"]

# Metric used to choose the best resulting network
# Options: "mse", "mae", "tv", "cdf_mae"
COMMON_SELECTION_METRIC = "tv"

# Friend-age plots for these ego ages
SELECTED_EGO_AGES = [16, 20, 25, 30, 40, 50, 60, 70, 80, 90]

#%% Build empirical target friend-age matrix

targetP = pd.DataFrame(
    0.0,
    index=AGE_VALUES,
    columns=AGE_VALUES
)

targetP.index.name = "ego_age"
targetP.columns.name = "friend_age"

for ego_age, df in netag_dist_by_age_empirical.items():

    ego_age = int(ego_age)

    if MIN_AGE <= ego_age <= MAX_AGE:

        tmp = df[["age", "probability"]].copy()

        tmp["age"] = pd.to_numeric(tmp["age"], errors="coerce")
        tmp["probability"] = pd.to_numeric(tmp["probability"], errors="coerce")

        tmp = tmp.dropna()
        tmp["age"] = tmp["age"].astype(int)

        tmp = tmp[tmp["age"].between(MIN_AGE, MAX_AGE)]

        probs = (
            tmp.groupby("age")["probability"]
            .sum()
            .reindex(AGE_VALUES, fill_value=0)
            .astype(float)
        )

        if probs.sum() > 0:
            targetP.loc[ego_age, :] = probs / probs.sum()

target_array = targetP.to_numpy(dtype=float)

target_rows = targetP.sum(axis=1).to_numpy() > 0

age_to_idx = {
    age: i
    for i, age in enumerate(AGE_VALUES)
}

print("Number of ego ages with empirical target:", target_rows.sum())

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

#%% Helper function to collect network information

def collect_network_info(
    name,
    G_in,
    node_age_in,
    simP=None,
    history=None,
    counts=None,
    initial_loss=None,
    best_loss=None
):

    node_list = sorted(G_in.nodes())

    nodes_info = pd.DataFrame({
        "node": node_list,
        "age": [node_age_in[node] for node in node_list],
        "degree": [G_in.degree[node] for node in node_list]
    })

    edges_info = pd.DataFrame([
        {
            "node_1": u,
            "node_2": v,
            "age_1": node_age_in[u],
            "age_2": node_age_in[v],
            "degree_1": G_in.degree[u],
            "degree_2": G_in.degree[v]
        }
        for u, v in G_in.edges()
    ])

    summary_info = pd.DataFrame({
        "item": [
            "name",
            "number_of_nodes",
            "number_of_edges",
            "average_degree",
            "initial_loss",
            "best_loss"
        ],
        "value": [
            name,
            G_in.number_of_nodes(),
            G_in.number_of_edges(),
            np.mean([d for _, d in G_in.degree()]),
            initial_loss,
            best_loss
        ]
    })

    return {
        "name": name,
        "G": G_in.copy(),
        "node_age": node_age_in.copy(),
        "nodes": nodes_info,
        "edges": edges_info,
        "summary": summary_info,
        "simP": None if simP is None else simP.copy(),
        "history": None if history is None else history.copy(),
        "counts": None if counts is None else counts.copy()
    }

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

#%% Construct base network

rng_network = np.random.default_rng(NETWORK_SEED)

node_ages, age_dist_assigned = make_sequence_from_distribution(
    network_age_dist,
    value_col="age",
    N=N,
    rng=rng_network
)

degree_sequence_array, degree_dist_assigned = make_sequence_from_distribution(
    network_degree_dist,
    value_col="degree",
    N=N,
    rng=rng_network
)

degree_sequence_array = degree_sequence_array.astype(int)

if degree_sequence_array.sum() % 2 == 1:

    candidates = np.where(degree_sequence_array > 0)[0]

    if len(candidates) == 0:
        raise ValueError("Odd degree sum but no positive degree to adjust.")

    chosen = rng_network.choice(candidates)
    degree_sequence_array[chosen] -= 1

degree_sequence = degree_sequence_array.tolist()

if not nx.is_graphical(degree_sequence, method="eg"):
    raise ValueError("Degree sequence is not graphical.")

G = nx.random_degree_sequence_graph(
    degree_sequence,
    tries=1000,
    seed=NETWORK_SEED
)

node_age = {
    node: int(age)
    for node, age in zip(sorted(G.nodes()), node_ages)
}

nx.set_node_attributes(G, node_age, "age")

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Average degree:", np.mean([d for _, d in G.degree()]))
print("Self-loops:", nx.number_of_selfloops(G))

print(
    "Degree sequence preserved:",
    sorted([d for _, d in G.degree()]) == sorted(degree_sequence)
)

base_G = G.copy()
base_node_age = node_age.copy()

base_network_info = collect_network_info(
    name="base_network",
    G_in=base_G,
    node_age_in=base_node_age
)

#%% Check initial age and degree distributions

sim_age_dist = (
    pd.Series(base_node_age)
    .value_counts(normalize=True)
    .sort_index()
    .rename_axis("age")
    .reset_index(name="sim_probability")
)

age_check = (
    network_age_dist
    .rename(columns={"probability": "target_probability"})
    .merge(sim_age_dist, on="age", how="outer")
    .fillna(0)
    .sort_values("age")
    .reset_index(drop=True)
)

print("\nInitial age distribution check:")
print(age_check.to_string(index=False))


sim_degree_dist = (
    pd.Series(dict(base_G.degree()))
    .value_counts(normalize=True)
    .sort_index()
    .rename_axis("degree")
    .reset_index(name="sim_probability")
)

degree_check = (
    network_degree_dist
    .rename(columns={"probability": "target_probability"})
    .merge(sim_degree_dist, on="degree", how="outer")
    .fillna(0)
    .sort_values("degree")
    .reset_index(drop=True)
)

print("\nInitial degree distribution check:")
print(degree_check.to_string(index=False))

#%% Monte Carlo helper functions

def counts_to_probs(counts):

    row_sums = counts.sum(axis=1, keepdims=True)

    probs = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0
    )

    return probs


def initialise_friend_age_counts(G_this, node_age_this):

    counts = np.zeros(
        (len(AGE_VALUES), len(AGE_VALUES)),
        dtype=float
    )

    for u, v in G_this.edges():

        age_u = int(node_age_this[u])
        age_v = int(node_age_this[v])

        i = age_to_idx.get(age_u)
        j = age_to_idx.get(age_v)

        if i is not None and j is not None:
            counts[i, j] += 1
            counts[j, i] += 1

    return counts


def add_edge_contribution(counts, age_x, age_y, amount):

    i = age_to_idx.get(int(age_x))
    j = age_to_idx.get(int(age_y))

    if i is not None and j is not None:
        counts[i, j] += amount
        counts[j, i] += amount


def proposed_counts_after_swap(G_this, node_age_this, counts, u, v):

    new_counts = counts.copy()

    affected_edges = set()

    for nbr in G_this.neighbors(u):
        affected_edges.add(tuple(sorted((u, nbr))))

    for nbr in G_this.neighbors(v):
        affected_edges.add(tuple(sorted((v, nbr))))

    for x, y in affected_edges:
        add_edge_contribution(
            new_counts,
            node_age_this[x],
            node_age_this[y],
            -1
        )

    for x, y in affected_edges:

        if x == u:
            age_x = node_age_this[v]
        elif x == v:
            age_x = node_age_this[u]
        else:
            age_x = node_age_this[x]

        if y == u:
            age_y = node_age_this[v]
        elif y == v:
            age_y = node_age_this[u]
        else:
            age_y = node_age_this[y]

        add_edge_contribution(
            new_counts,
            age_x,
            age_y,
            1
        )

    return new_counts

#%% Loss functions

def loss_mse(counts):

    simP = counts_to_probs(counts)

    diff = simP[target_rows, :] - target_array[target_rows, :]

    return np.mean(diff ** 2)


def loss_mae(counts):

    simP = counts_to_probs(counts)

    diff = simP[target_rows, :] - target_array[target_rows, :]

    return np.mean(np.abs(diff))


def loss_tv(counts):

    simP = counts_to_probs(counts)

    diff = simP[target_rows, :] - target_array[target_rows, :]

    row_tv = 0.5 * np.abs(diff).sum(axis=1)

    return np.mean(row_tv)


def loss_cdf_mae(counts):

    simP = counts_to_probs(counts)

    diff = simP[target_rows, :] - target_array[target_rows, :]

    cdf_diff = np.cumsum(diff, axis=1)

    return np.mean(np.abs(cdf_diff))


loss_functions = {
    "mse": loss_mse,
    "mae": loss_mae,
    "tv": loss_tv,
    "cdf_mae": loss_cdf_mae
}

#%% Monte Carlo runner

def run_age_swapping_monte_carlo(
    method_name,
    base_G,
    base_node_age,
    loss_function,
    n_iter=N_ITER,
    seed=MONTE_CARLO_SEED,
    temp_factor=0.05,
    cooling=0.99995,
    print_every=PRINT_EVERY
):

    print(f"\nRunning Monte Carlo with loss function: {method_name}")

    G_this = base_G.copy()
    node_age_this = base_node_age.copy()

    rng_propose = np.random.default_rng(seed)
    rng_accept = np.random.default_rng(seed + 1)

    counts = initialise_friend_age_counts(G_this, node_age_this)

    current_loss = loss_function(counts)
    initial_loss = current_loss

    best_loss = current_loss
    best_counts = counts.copy()
    best_node_age = node_age_this.copy()

    temperature = max(current_loss * temp_factor, 1e-9)

    nodes = np.array(list(G_this.nodes()))

    history = []

    accepted = 0
    accepted_better = 0
    accepted_equal = 0
    accepted_worse = 0
    rejected = 0

    print("Initial loss:", current_loss)

    for it in range(1, n_iter + 1):

        u, v = rng_propose.choice(nodes, size=2, replace=False)

        if node_age_this[u] != node_age_this[v]:

            new_counts = proposed_counts_after_swap(
                G_this,
                node_age_this,
                counts,
                u,
                v
            )

            new_loss = loss_function(new_counts)

            delta = new_loss - current_loss

            if delta <= 0:
                accept = True
            else:
                accept = rng_accept.random() < np.exp(
                    -delta / max(temperature, 1e-12)
                )

            if accept:

                counts = new_counts
                node_age_this[u], node_age_this[v] = node_age_this[v], node_age_this[u]
                current_loss = new_loss
                accepted += 1

                if delta < 0:
                    accepted_better += 1
                elif delta == 0:
                    accepted_equal += 1
                else:
                    accepted_worse += 1

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_counts = counts.copy()
                    best_node_age = node_age_this.copy()

            else:
                rejected += 1

        temperature *= cooling

        if it % print_every == 0:

            history.append({
                "method": method_name,
                "iteration": it,
                "current_loss": current_loss,
                "best_loss": best_loss,
                "temperature": temperature,
                "accepted_swaps": accepted,
                "accepted_better": accepted_better,
                "accepted_equal": accepted_equal,
                "accepted_worse": accepted_worse,
                "rejected": rejected
            })

            print(
                f"Iteration {it}, "
                f"current loss = {current_loss:.8f}, "
                f"best loss = {best_loss:.8f}, "
                f"accepted swaps = {accepted}, "
                f"worse accepted = {accepted_worse}"
            )

    nx.set_node_attributes(G_this, best_node_age, "age")

    simP = pd.DataFrame(
        counts_to_probs(best_counts),
        index=AGE_VALUES,
        columns=AGE_VALUES
    )

    simP.index.name = "ego_age"
    simP.columns.name = "friend_age"

    return {
        "method": method_name,
        "G": G_this,
        "node_age": best_node_age,
        "counts": best_counts,
        "simP": simP,
        "history": pd.DataFrame(history),
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "accepted_swaps": accepted,
        "accepted_better": accepted_better,
        "accepted_equal": accepted_equal,
        "accepted_worse": accepted_worse,
        "rejected": rejected
    }

#%% Run Monte Carlo for each loss function

results = {}

for method_name in LOSS_FUNCTIONS_TO_RUN:

    results[method_name] = run_age_swapping_monte_carlo(
        method_name=method_name,
        base_G=base_G,
        base_node_age=base_node_age,
        loss_function=loss_functions[method_name],
        n_iter=N_ITER,
        seed=MONTE_CARLO_SEED,
        temp_factor=0.05,
        cooling=0.99995,
        print_every=PRINT_EVERY
    )

#%% Store resulting network information as variables

resulting_network_info = {
    method_name: collect_network_info(
        name=method_name,
        G_in=result["G"],
        node_age_in=result["node_age"],
        simP=result["simP"],
        history=result["history"],
        counts=result["counts"],
        initial_loss=result["initial_loss"],
        best_loss=result["best_loss"]
    )
    for method_name, result in results.items()
}

#%% Evaluate fit using common metrics

def evaluate_fit(simP):

    sim_array = simP.loc[AGE_VALUES, AGE_VALUES].to_numpy(dtype=float)

    diff = sim_array[target_rows, :] - target_array[target_rows, :]

    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    tv = np.mean(0.5 * np.abs(diff).sum(axis=1))

    cdf_diff = np.cumsum(diff, axis=1)
    cdf_mae = np.mean(np.abs(cdf_diff))

    return {
        "mse": mse,
        "mae": mae,
        "tv": tv,
        "cdf_mae": cdf_mae
    }


fit_summary = pd.DataFrame([
    {
        "method": method_name,
        "optimised_loss_value": result["best_loss"],
        "accepted_swaps": result["accepted_swaps"],
        "accepted_worse": result["accepted_worse"],
        **evaluate_fit(result["simP"])
    }
    for method_name, result in results.items()
])

for metric in ["mse", "mae", "tv", "cdf_mae"]:
    fit_summary[f"{metric}_rank"] = fit_summary[metric].rank(method="min")

fit_summary["average_rank"] = fit_summary[
    ["mse_rank", "mae_rank", "tv_rank", "cdf_mae_rank"]
].mean(axis=1)

fit_summary = fit_summary.sort_values(COMMON_SELECTION_METRIC).reset_index(drop=True)

print("\nFit summary:")
print(fit_summary.to_string(index=False))

best_method = fit_summary.iloc[0]["method"]
best_result = results[best_method]
best_network_info = resulting_network_info[best_method]

print(f"\nBest method by {COMMON_SELECTION_METRIC}: {best_method}")

#%% Row-by-row fit by ego age

def evaluate_fit_by_age(simP):

    sim_array = simP.loc[AGE_VALUES, AGE_VALUES].to_numpy(dtype=float)

    diff = sim_array - target_array

    rows = []

    for i, ego_age in enumerate(AGE_VALUES):

        if target_rows[i]:

            row_diff = diff[i, :]

            rows.append({
                "ego_age": ego_age,
                "mse": np.mean(row_diff ** 2),
                "mae": np.mean(np.abs(row_diff)),
                "tv": 0.5 * np.abs(row_diff).sum(),
                "cdf_mae": np.mean(np.abs(np.cumsum(row_diff)))
            })

    return pd.DataFrame(rows)


row_fit_all = []

for method_name, result in results.items():

    tmp = evaluate_fit_by_age(result["simP"])
    tmp["method"] = method_name

    row_fit_all.append(tmp)

row_fit_all = pd.concat(row_fit_all, ignore_index=True)

print("\nWorst-fitting ego ages by method:")
print(
    row_fit_all
    .sort_values(["method", COMMON_SELECTION_METRIC], ascending=[True, False])
    .groupby("method")
    .head(10)
    .to_string(index=False)
)

#%% Plot friend-age distributions for selected ego ages

friend_age_comparisons = {}

for ego_age in SELECTED_EGO_AGES:

    if ego_age not in targetP.index:
        print(f"Skipping age {ego_age}: not in targetP.")
        continue

    if targetP.loc[ego_age].sum() == 0:
        print(f"Skipping age {ego_age}: empirical target row is empty.")
        continue

    compare_df = pd.DataFrame({
        "friend_age": AGE_VALUES,
        "empirical": targetP.loc[ego_age].to_numpy()
    })

    for method_name, result in results.items():
        compare_df[method_name] = result["simP"].loc[ego_age].to_numpy()

    friend_age_comparisons[ego_age] = compare_df

    plt.figure(figsize=(9, 6), dpi=1000)

    plt.plot(
        compare_df["friend_age"],
        compare_df["empirical"],
        label="Empirical",
        linewidth=2.5,
        color="gray"
    )

    for method_name in results:
        plt.plot(
            compare_df["friend_age"],
            compare_df[method_name],
            label=method_name,
            linewidth=1.8
        )

    plt.xlabel("Friend age")
    plt.ylabel("Probability")
    plt.title(f"Friend-age distribution for ego age {ego_age}")
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Plot Monte Carlo convergence

history_list = []

for method_name, result in results.items():

    history = result["history"]

    if not history.empty:
        history_list.append(history.assign(method=method_name))

    plt.figure(figsize=(9, 6), dpi=1000)

    plt.plot(
        history["iteration"],
        history["current_loss"],
        label="Current loss",
        alpha=0.7
    )

    plt.plot(
        history["iteration"],
        history["best_loss"],
        label="Best loss",
        linewidth=2
    )

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Monte Carlo convergence: {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

history_all = pd.concat(history_list, ignore_index=True)

#%% Degree distribution by age check

def degree_by_age_check(result, method_name):

    G_this = result["G"]
    node_age_this = result["node_age"]

    nodes_final = pd.DataFrame({
        "node": list(G_this.nodes()),
        "age": [node_age_this[node] for node in G_this.nodes()],
        "degree": [G_this.degree[node] for node in G_this.nodes()]
    })

    nodes_final = nodes_final[
        nodes_final["age"].between(MIN_AGE, MAX_AGE)
    ].copy()

    nodes_final["age"] = nodes_final["age"].astype(int)
    nodes_final["degree"] = nodes_final["degree"].astype(int)

    emp_degree_data = indresp_deg[["age", "closenum"]].copy()

    emp_degree_data["age"] = pd.to_numeric(
        emp_degree_data["age"],
        errors="coerce"
    )

    emp_degree_data["closenum"] = pd.to_numeric(
        emp_degree_data["closenum"],
        errors="coerce"
    )

    emp_degree_data = emp_degree_data.dropna(subset=["age", "closenum"])

    emp_degree_data = emp_degree_data[
        emp_degree_data["age"].between(MIN_AGE, MAX_AGE)
    ].copy()

    emp_degree_data["age"] = emp_degree_data["age"].astype(int)
    emp_degree_data["degree"] = emp_degree_data["closenum"].astype(int)

    degree_min = 0

    degree_max = max(
        nodes_final["degree"].max(),
        emp_degree_data["degree"].max()
    )

    all_degrees = np.arange(degree_min, degree_max + 1)

    full_index = pd.MultiIndex.from_product(
        [AGE_VALUES, all_degrees],
        names=["age", "degree"]
    )

    emp_degree_by_age = (
        emp_degree_data
        .groupby(["age", "degree"])
        .size()
        .reindex(full_index, fill_value=0)
        .rename("emp_count")
        .reset_index()
    )

    emp_totals = emp_degree_by_age.groupby("age")["emp_count"].transform("sum")

    emp_degree_by_age["emp_probability"] = np.where(
        emp_totals > 0,
        emp_degree_by_age["emp_count"] / emp_totals,
        0
    )

    sim_degree_by_age = (
        nodes_final
        .groupby(["age", "degree"])
        .size()
        .reindex(full_index, fill_value=0)
        .rename("sim_count")
        .reset_index()
    )

    sim_totals = sim_degree_by_age.groupby("age")["sim_count"].transform("sum")

    sim_degree_by_age["sim_probability"] = np.where(
        sim_totals > 0,
        sim_degree_by_age["sim_count"] / sim_totals,
        0
    )

    degree_by_age_compare = (
        emp_degree_by_age
        .merge(
            sim_degree_by_age,
            on=["age", "degree"],
            how="outer"
        )
        .fillna(0)
        .sort_values(["age", "degree"])
        .reset_index(drop=True)
    )

    rows = []

    for ego_age, subdf in degree_by_age_compare.groupby("age"):

        rows.append({
            "method": method_name,
            "age": ego_age,
            "emp_n": subdf["emp_count"].sum(),
            "sim_n": subdf["sim_count"].sum(),
            "mae": np.mean(
                np.abs(subdf["sim_probability"] - subdf["emp_probability"])
            ),
            "mse": np.mean(
                (subdf["sim_probability"] - subdf["emp_probability"]) ** 2
            ),
            "tv": 0.5 * np.abs(
                subdf["sim_probability"] - subdf["emp_probability"]
            ).sum(),
            "emp_average_degree": np.sum(
                subdf["degree"] * subdf["emp_probability"]
            ),
            "sim_average_degree": np.sum(
                subdf["degree"] * subdf["sim_probability"]
            )
        })

    degree_by_age_fit = pd.DataFrame(rows)

    return degree_by_age_compare, degree_by_age_fit

#%% Run degree-by-age check

degree_by_age_outputs = {}
degree_by_age_fit_all = []

for method_name, result in results.items():

    compare_df, fit_df = degree_by_age_check(
        result=result,
        method_name=method_name
    )

    degree_by_age_outputs[method_name] = {
        "compare": compare_df,
        "fit": fit_df
    }

    degree_by_age_fit_all.append(fit_df)

degree_by_age_fit_all = pd.concat(
    degree_by_age_fit_all,
    ignore_index=True
)

print("\nWorst degree-by-age fits by method:")
print(
    degree_by_age_fit_all
    .sort_values(["method", "tv"], ascending=[True, False])
    .groupby("method")
    .head(10)
    .to_string(index=False)
)

#%% Bar plot: average degree by age for best network

best_degree_by_age_fit = degree_by_age_outputs[best_method]["fit"]

x = np.arange(len(best_degree_by_age_fit))
width = 0.4

plt.figure(figsize=(12, 6), dpi=1000)

plt.bar(
    x - width / 2,
    best_degree_by_age_fit["emp_average_degree"],
    width=width,
    label="Empirical",
    color="gray",
    edgecolor="black",
    alpha=0.8
)

plt.bar(
    x + width / 2,
    best_degree_by_age_fit["sim_average_degree"],
    width=width,
    label=f"Simulated best network: {best_method}",
    color="steelblue",
    edgecolor="black",
    alpha=0.8
)

plt.xlabel("Age")
plt.ylabel("Average degree")
plt.title(f"Average degree by age: empirical vs best network ({best_method})")

# Show ticks at 16, 20, 25, ..., 100
tick_ages = [16] + list(range(20, 101, 5))

tick_positions = [
    best_degree_by_age_fit.index[
        best_degree_by_age_fit["age"] == age
    ][0]
    for age in tick_ages
    if age in best_degree_by_age_fit["age"].values
]

tick_labels = [
    age
    for age in tick_ages
    if age in best_degree_by_age_fit["age"].values
]

plt.xticks(tick_positions, tick_labels)

plt.legend()
plt.tight_layout()
plt.show()

#%% Plot degree distribution by selected ages: empirical vs base vs best network

# -----------------------------
# Prepare empirical degree data
# -----------------------------

emp_degree_data = indresp_deg[["age", "closenum"]].copy()

emp_degree_data["age"] = pd.to_numeric(
    emp_degree_data["age"],
    errors="coerce"
)

emp_degree_data["closenum"] = pd.to_numeric(
    emp_degree_data["closenum"],
    errors="coerce"
)

emp_degree_data = emp_degree_data.dropna(subset=["age", "closenum"])

emp_degree_data = emp_degree_data[
    emp_degree_data["age"].between(MIN_AGE, MAX_AGE)
].copy()

emp_degree_data["age"] = emp_degree_data["age"].astype(int)
emp_degree_data["degree"] = emp_degree_data["closenum"].astype(int)


# -----------------------------
# Prepare base network node data
# -----------------------------

base_nodes_degree = pd.DataFrame({
    "node": list(base_G.nodes()),
    "age": [base_node_age[node] for node in base_G.nodes()],
    "degree": [base_G.degree[node] for node in base_G.nodes()]
})

base_nodes_degree["age"] = base_nodes_degree["age"].astype(int)
base_nodes_degree["degree"] = base_nodes_degree["degree"].astype(int)


# -----------------------------
# Prepare best network node data
# -----------------------------

best_G = results[best_method]["G"]
best_node_age = results[best_method]["node_age"]

best_nodes_degree = pd.DataFrame({
    "node": list(best_G.nodes()),
    "age": [best_node_age[node] for node in best_G.nodes()],
    "degree": [best_G.degree[node] for node in best_G.nodes()]
})

best_nodes_degree["age"] = best_nodes_degree["age"].astype(int)
best_nodes_degree["degree"] = best_nodes_degree["degree"].astype(int)


# -----------------------------
# Degree range
# -----------------------------

max_degree = int(max(
    emp_degree_data["degree"].max(),
    base_nodes_degree["degree"].max(),
    best_nodes_degree["degree"].max()
))

all_degrees = np.arange(0, max_degree + 1)


# -----------------------------
# Function: P(degree | age)
# -----------------------------

def degree_distribution_for_age(df, ego_age, all_degrees):
    counts = (
        df.loc[df["age"] == ego_age, "degree"]
        .value_counts()
        .reindex(all_degrees, fill_value=0)
        .sort_index()
    )

    if counts.sum() > 0:
        probs = counts / counts.sum()
    else:
        probs = counts.astype(float)

    return probs


# -----------------------------
# Plot selected ages
# -----------------------------

degree_dist_comparisons_selected = {}

for ego_age in SELECTED_EGO_AGES:

    compare_df = pd.DataFrame({
        "degree": all_degrees,
        "empirical": degree_distribution_for_age(
            emp_degree_data,
            ego_age,
            all_degrees
        ).to_numpy(),
        "base_network": degree_distribution_for_age(
            base_nodes_degree,
            ego_age,
            all_degrees
        ).to_numpy(),
        f"best_network_{best_method}": degree_distribution_for_age(
            best_nodes_degree,
            ego_age,
            all_degrees
        ).to_numpy()
    })

    degree_dist_comparisons_selected[ego_age] = compare_df

    if compare_df.drop(columns="degree").sum().sum() == 0:
        print(f"Skipping age {ego_age}: no empirical or simulated degree data.")
        continue

    x = np.arange(len(all_degrees))
    width = 0.25

    plt.figure(figsize=(9, 6), dpi=1000)

    plt.bar(
        x - width,
        compare_df["empirical"],
        width=width,
        label="Empirical",
        color="black",
        edgecolor="black",
        alpha=0.8
    )

    plt.bar(
        x,
        compare_df["base_network"],
        width=width,
        label="Base network",
        color="gray",
        edgecolor="black",
        alpha=0.8
    )

    plt.bar(
        x + width,
        compare_df[f"best_network_{best_method}"],
        width=width,
        label=f"Best network ({best_method})",
        color="steelblue",
        edgecolor="black",
        alpha=0.8
    )

    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.title(f"Degree distribution for age {ego_age}")

    plt.xticks(x, all_degrees)

    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Plot friend-age distributions for selected ages: empirical vs MSE only

friend_age_comparisons_mse = {}

for ego_age in SELECTED_EGO_AGES:

    if ego_age not in targetP.index:
        print(f"Skipping age {ego_age}: not in targetP.")
        continue

    if targetP.loc[ego_age].sum() == 0:
        print(f"Skipping age {ego_age}: empirical target row is empty.")
        continue

    compare_df = pd.DataFrame({
        "friend_age": AGE_VALUES,
        "empirical": targetP.loc[ego_age].to_numpy(),
        "mse": results["mse"]["simP"].loc[ego_age].to_numpy()
    })

    friend_age_comparisons_mse[ego_age] = compare_df

    plt.figure(figsize=(9, 6), dpi=1000)

    plt.plot(
        compare_df["friend_age"],
        compare_df["empirical"],
        label="Empirical",
        linewidth=2.5,
        color="black"
    )

    plt.plot(
        compare_df["friend_age"],
        compare_df["mse"],
        label="Simulated network, MSE",
        linewidth=2,
        color="steelblue"
    )

    plt.xlabel("Friend age")
    plt.ylabel("Probability")
    plt.title(f"Friend-age distribution for ego age {ego_age}: empirical vs MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Store all outputs as variables

all_network_outputs = {
    "base_network": base_network_info,
    "resulting_networks": resulting_network_info,
    "best_method": best_method,
    "best_network": best_network_info,
    "fit_summary": fit_summary,
    "row_fit_all": row_fit_all,
    "history_all": history_all,
    "friend_age_comparisons": friend_age_comparisons,
    "degree_by_age_outputs": degree_by_age_outputs,
    "degree_by_age_fit_all": degree_by_age_fit_all
}

#%% 



















