#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jul 22 00:00:00 2026

@author: Andy.
"""

#%%

import numpy as np
import math
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
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

#%% UK Household Longitudinal Study: Understanding Society (UKHLS): Waves 1-15, 2009-2024
# Wave 3 & 6

c_indresp = pd.read_csv("c_indresp.csv",
                        usecols=["c_age_dv", "c_netag_1", "c_netag_2", "c_netag_3"])

f_indresp = pd.read_csv("f_indresp.csv",
                        usecols=["f_age_dv", "f_netag_1", "f_netag_2", "f_netag_3"])

#%% Data Cleaning

indresp_1992 = bb_indresp.rename(columns={
                                          "bb_age": "age",
                                          "bb_netag_1": "netag_1",
                                          "bb_netag_2": "netag_2",
                                          "bb_netag_3": "netag_3"
                                          })
indresp_1992 = indresp_1992[indresp_1992["age"] >= 0].reset_index(drop=True)
indresp_1992 = indresp_1992[
                            ~indresp_1992[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_1992[["netag_1", "netag_2", "netag_3"]] = indresp_1992[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_1992[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_1994 = bd_indresp.rename(columns={
                                          "bd_age": "age",
                                          "bd_netag_1": "netag_1",
                                          "bd_netag_2": "netag_2",
                                          "bd_netag_3": "netag_3"
                                          })
indresp_1994 = indresp_1994[indresp_1994["age"] >= 0].reset_index(drop=True)
indresp_1994 = indresp_1994[
                            ~indresp_1994[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_1994[["netag_1", "netag_2", "netag_3"]] = indresp_1994[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_1994[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_1996 = bf_indresp.rename(columns={
                                          "bf_age": "age",
                                          "bf_netag_1": "netag_1",
                                          "bf_netag_2": "netag_2",
                                          "bf_netag_3": "netag_3"
                                          })
indresp_1996 = indresp_1996[indresp_1996["age"] >= 0].reset_index(drop=True)
indresp_1996 = indresp_1996[
                            ~indresp_1996[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_1996[["netag_1", "netag_2", "netag_3"]] = indresp_1996[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_1996[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_1998 = bh_indresp.rename(columns={
                                          "bh_age": "age",
                                          "bh_netag_1": "netag_1",
                                          "bh_netag_2": "netag_2",
                                          "bh_netag_3": "netag_3"
                                          })
indresp_1998 = indresp_1998[indresp_1998["age"] >= 0].reset_index(drop=True)
indresp_1998 = indresp_1998[
                            ~indresp_1998[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_1998[["netag_1", "netag_2", "netag_3"]] = indresp_1998[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_1998[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_2000 = bj_indresp.rename(columns={
                                          "bj_age": "age",
                                          "bj_netag_1": "netag_1",
                                          "bj_netag_2": "netag_2",
                                          "bj_netag_3": "netag_3"
                                          })
indresp_2000 = indresp_2000[indresp_2000["age"] >= 0].reset_index(drop=True)
indresp_2000 = indresp_2000[
                            ~indresp_2000[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_2000[["netag_1", "netag_2", "netag_3"]] = indresp_2000[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_2000[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_2002 = bl_indresp.rename(columns={
                                          "bl_age": "age",
                                          "bl_netag_1": "netag_1",
                                          "bl_netag_2": "netag_2",
                                          "bl_netag_3": "netag_3"
                                          })
indresp_2002 = indresp_2002[indresp_2002["age"] >= 0].reset_index(drop=True)
indresp_2002 = indresp_2002[
                            ~indresp_2002[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_2002[["netag_1", "netag_2", "netag_3"]] = indresp_2002[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_2002[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_2004 = bn_indresp.rename(columns={
                                          "bn_age": "age",
                                          "bn_netag_1": "netag_1",
                                          "bn_netag_2": "netag_2",
                                          "bn_netag_3": "netag_3"
                                          })
indresp_2004 = indresp_2004[indresp_2004["age"] >= 0].reset_index(drop=True)
indresp_2004 = indresp_2004[
                            ~indresp_2004[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_2004[["netag_1", "netag_2", "netag_3"]] = indresp_2004[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_2004[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_2006 = bp_indresp.rename(columns={
                                          "bp_age": "age",
                                          "bp_netag_1": "netag_1",
                                          "bp_netag_2": "netag_2",
                                          "bp_netag_3": "netag_3"
                                          })
indresp_2006 = indresp_2006[indresp_2006["age"] >= 0].reset_index(drop=True)
indresp_2006 = indresp_2006[
                            ~indresp_2006[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_2006[["netag_1", "netag_2", "netag_3"]] = indresp_2006[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_2006[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_2008 = br_indresp.rename(columns={
                                          "br_age": "age",
                                          "br_netag_1": "netag_1",
                                          "br_netag_2": "netag_2",
                                          "br_netag_3": "netag_3"
                                          })
indresp_2008 = indresp_2008[indresp_2008["age"] >= 0].reset_index(drop=True)
indresp_2008 = indresp_2008[
                            ~indresp_2008[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_2008[["netag_1", "netag_2", "netag_3"]] = indresp_2008[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_2008[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_2011 = c_indresp.rename(columns={
                                         "c_age_dv": "age",
                                         "c_netag_1": "netag_1",
                                         "c_netag_2": "netag_2",
                                         "c_netag_3": "netag_3"
                                         })
indresp_2011 = indresp_2011[indresp_2011["age"] >= 0].reset_index(drop=True)
indresp_2011 = indresp_2011[
                            ~indresp_2011[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_2011[["netag_1", "netag_2", "netag_3"]] = indresp_2011[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_2011[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)


indresp_2014 = f_indresp.rename(columns={
                                         "f_age_dv": "age",
                                         "f_netag_1": "netag_1",
                                         "f_netag_2": "netag_2",
                                         "f_netag_3": "netag_3"
                                         })
indresp_2014 = indresp_2014[indresp_2014["age"] >= 0].reset_index(drop=True)
indresp_2014 = indresp_2014[
                            ~indresp_2014[["netag_1", "netag_2", "netag_3"]].lt(0).all(axis = 1)
                            ].reset_index(drop = True)
indresp_2014[["netag_1", "netag_2", "netag_3"]] = indresp_2014[["netag_1", "netag_2", "netag_3"]].mask(
                                                      indresp_2014[["netag_1", "netag_2", "netag_3"]] < 0,
                                                      np.nan)

#%%

indresp = pd.concat(
                    [
                     indresp_1992.assign(year = 1992),
                     indresp_1994.assign(year = 1994),
                     indresp_1996.assign(year = 1996),
                     indresp_1998.assign(year = 1998),
                     indresp_2000.assign(year = 2000),
                     indresp_2002.assign(year = 2002),
                     indresp_2004.assign(year = 2004),
                     indresp_2006.assign(year = 2006),
                     indresp_2008.assign(year = 2008),
                     indresp_2011.assign(year = 2011),
                     indresp_2014.assign(year = 2014)
                    ],
                    ignore_index = True
                    )

indresp = indresp[["year"] + [col for col in indresp.columns if col != "year"]]

indresp["netag_1_d"] = indresp["netag_1"] - indresp["age"]
indresp["netag_2_d"] = indresp["netag_2"] - indresp["age"]
indresp["netag_3_d"] = indresp["netag_3"] - indresp["age"]

#%%

print(indresp["age"].value_counts().sort_index())

with pd.option_context("display.max_rows", None):
    print(indresp["age"].value_counts().sort_index())

# print(indresp["age"].min())
# print(indresp["age"].max())
    
# print(indresp["netag_1"].min())
# print(indresp["netag_1"].max())

# print(indresp["netag_2"].min())
# print(indresp["netag_2"].max())

# print(indresp["netag_3"].min())
# print(indresp["netag_3"].max())

# print(indresp[indresp["age"] == 100])
# print(indresp[indresp["age"] == 102])

# print(indresp[indresp["netag_1"] == 0])
# print(indresp[indresp["netag_2"] == 0])
# print(indresp[indresp["netag_3"] == 0])

#%%

indresp_copy = indresp.copy()
indresp_copy["age_group"] = pd.cut(
                                   indresp_copy["age"],
                                   bins = [15, 20, 30, 40, 50, 60, 70, 80, 90, np.inf],
                                   labels = [
                                             "15-19",
                                             "20-29",
                                             "30-39",
                                             "40-49",
                                             "50-59",
                                             "60-69",
                                             "70-79",
                                             "80-89",
                                             "90+"
                                             ],
                                   right = False
                                   )
indresp_by_age_group = {
                        label: indresp_copy[indresp_copy["age_group"] == label].copy()
                        for label in ["15-19", "20-29", "30-39", "40-49", "50-59",
                                      "60-69", "70-79", "80-89", "90+"]
                        }
netag_d_by_age_group = {
                        label: subdf[["netag_1_d", "netag_2_d", "netag_3_d"]].stack().dropna().tolist()
                        for label, subdf in indresp_by_age_group.items()
                        }

# print(min(netag_d_by_age_group["90+"]))
# print(max(netag_d_by_age_group["90+"]))

# print(min(netag_d_by_age_group["15-19"]))
# print(max(netag_d_by_age_group["15-19"]))

dist_netag_d_by_age_group = {
                             ego_age: (
                                       pd.Series(friend_ages)
                                       .value_counts(normalize=True)
                                       .sort_index()
                                       .rename_axis("age group")
                                       .reset_index(name = "probability")
                                       )
                             for ego_age, friend_ages in netag_d_by_age_group.items()
                             }

#%%

plt.figure(figsize = (9, 6), dpi = 1000)

palette = sns.color_palette("Spectral", 9)

sns.kdeplot(netag_d_by_age_group["15-19"], label = "Age 15-19", color=palette[0], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["20-29"], label = "Age 20-29", color=palette[1], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["30-39"], label = "Age 30-39", color=palette[2], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["40-49"], label = "Age 40-49", color=palette[3], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["50-59"], label = "Age 50-59", color=palette[4], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["60-69"], label = "Age 60-69", color=palette[5], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["70-79"], label = "Age 70-79", color=palette[6], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["80-89"], label = "Age 80-89", color=palette[7], fill=True, linewidth=2, alpha=0.3)
sns.kdeplot(netag_d_by_age_group["90+"], label = "Age 90+", color=palette[8], fill=True, linewidth=2, alpha=0.3)

plt.xlabel("Age difference with bestfriend")
plt.ylabel("Probability density")
plt.title("Distribution of age difference with bestfriend by age groups")
plt.legend()
plt.tight_layout()
plt.show()

#%%

indresp_by_age = {
                  age: subdf.copy()
                  for age, subdf in indresp_copy.groupby("age")
                  }

netag_d_by_age = (
                  indresp_copy
                  .groupby("age")[["netag_1_d", "netag_2_d", "netag_3_d"]]
                  .apply(lambda x: x.stack().dropna().tolist())
                  .to_dict()
                  )

netag_by_age = (
                indresp_copy
                .groupby("age")[["netag_1", "netag_2", "netag_3"]]
                .apply(lambda x: x.stack().dropna().tolist())
                .to_dict()
                )

dist_netag_by_age = {
                     ego_age: (
                               pd.Series(friend_ages)
                               .value_counts(normalize=True)
                               .sort_index()
                               .rename_axis("age")
                               .reset_index(name = "probability")
                               )
                     for ego_age, friend_ages in netag_by_age.items()
                     }

#%%


hist_netag_by_age = {
                     ego_age: (
                               pd.Series(friend_ages)
                               .value_counts(normalize=False)
                               .sort_index()
                               .rename_axis("age")
                               .reset_index(name = "count")
                               )
                     for ego_age, friend_ages in netag_by_age.items()
                     }


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


smoothed_dist_netag_by_age = {
                              ego_age: partial_smooth_from_hist(
                                                                hist_df,
                                                                min_age=0,
                                                                max_age=100,
                                                                bandwidth=0.5,
                                                                missing_scale=1.0
                                                                )
                              for ego_age, hist_df in hist_netag_by_age.items()
                              }


empirical_dist_netag_by_age = {
                               ego_age: (
                                         df[df["age"].between(16, 100)][["age", "probability"]]
                                         .assign(probability=lambda x: x["probability"] / x["probability"].sum())
                                         .reset_index(drop=True)
                                         )
                               for ego_age, df in smoothed_dist_netag_by_age.items()
                               if 16 <= ego_age <= 100
                               }


#%%

# -------

#%% Network Construction:

age_values = list(range(16, 101))

targetP = pd.DataFrame(
    0.0,
    index=age_values,
    columns=age_values
)

targetP.index.name = "ego_age"
targetP.columns.name = "friend_age"

for ego_age, df in empirical_dist_netag_by_age.items():
    ego_age = int(ego_age)

    if 16 <= ego_age <= 100:
        probs = (
            df.set_index("age")["probability"]
            .reindex(age_values, fill_value=0)
            .astype(float)
        )

        if probs.sum() > 0:
            targetP.loc[ego_age, :] = probs / probs.sum()
            
#%%

age_dist = pd.read_excel("ONS - UK Data for Population by Age (1974-2024).xlsx",
                         index_col = 0) # Great Britain
age_dist = (
            age_dist.loc[1974]
            .rename_axis("age")
            .reset_index(name="probability")
            )

#%% Network Construction:

rng = np.random.default_rng(123)

N = 10000

age_dist = age_dist[["age", "probability"]].copy()
degree_dist = degree_dist[["degree", "probability"]].copy() # FHS/...

age_dist["age"] = pd.to_numeric(age_dist["age"], errors="coerce")
age_dist["probability"] = pd.to_numeric(age_dist["probability"], errors="coerce")

degree_dist["degree"] = pd.to_numeric(degree_dist["degree"], errors="coerce")
degree_dist["probability"] = pd.to_numeric(degree_dist["probability"], errors="coerce")

age_dist = age_dist.dropna()
degree_dist = degree_dist.dropna()

age_dist["age"] = age_dist["age"].astype(int)
degree_dist["degree"] = degree_dist["degree"].astype(int)

age_dist = age_dist[age_dist["probability"] >= 0]
degree_dist = degree_dist[degree_dist["probability"] >= 0]

age_dist["probability"] = age_dist["probability"] / age_dist["probability"].sum()
degree_dist["probability"] = degree_dist["probability"] / degree_dist["probability"].sum()

# print("Age probability sum:", age_dist["probability"].sum())
# print("Degree probability sum:", degree_dist["probability"].sum())

age_dist = age_dist.sort_values("age").copy()

raw_age_counts = age_dist["probability"].to_numpy() * N
age_counts = np.floor(raw_age_counts).astype(int)

remainder = N - age_counts.sum()

if remainder > 0:
    age_counts[
        np.argsort(raw_age_counts - age_counts)[-remainder:]
    ] += 1

node_ages = np.repeat(
    age_dist["age"].astype(int).to_numpy(),
    age_counts
)

rng.shuffle(node_ages)

# print(len(node_ages))

degree_dist = degree_dist.sort_values("degree").copy()

raw_degree_counts = degree_dist["probability"].to_numpy() * N
degree_counts = np.floor(raw_degree_counts).astype(int)

remainder = N - degree_counts.sum()

if remainder > 0:
    degree_counts[
        np.argsort(raw_degree_counts - degree_counts)[-remainder:]
    ] += 1

degree_sequence = np.repeat(
    degree_dist["degree"].astype(int).to_numpy(),
    degree_counts
)

rng.shuffle(degree_sequence)

# print(len(degree_sequence))
# print("Mean degree:", degree_sequence.mean())

degree_sequence = degree_sequence.astype(int).tolist()

if sum(degree_sequence) % 2 == 1:
    raise ValueError("Sum of degree sequence is odd. It must be even.")

if not nx.is_graphical(degree_sequence, method="eg"):
    raise ValueError("Degree sequence is not graphical.")

G = nx.random_degree_sequence_graph(
    degree_sequence,
    tries=100,
    seed=123
)

# Assign ages to nodes
node_age = {
    node: int(age)
    for node, age in zip(sorted(G.nodes()), node_ages)
}

nx.set_node_attributes(G, node_age, "age")

# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())
# print("Average degree:", np.mean([d for _, d in G.degree()]))
# print("Self-loops:", nx.number_of_selfloops(G))

# print(
#       "Degree sequence preserved:",
#       sorted([d for _, d in G.degree()]) == sorted(degree_sequence)
#       )

sim_age_dist = (
    pd.Series(node_age)
    .value_counts(normalize=True)
    .sort_index()
    .rename_axis("age")
    .reset_index(name="sim_probability")
)

age_check = (
    age_dist
    .rename(columns={"probability": "target_probability"})
    .merge(sim_age_dist, on="age", how="outer")
    .fillna(0)
    .sort_values("age")
    .reset_index(drop=True)
)

# print(age_check.to_string(index=False))

sim_degree_dist = (
    pd.Series(dict(G.degree()))
    .value_counts(normalize=True)
    .sort_index()
    .rename_axis("degree")
    .reset_index(name="sim_probability")
)

degree_check = (
    degree_dist
    .rename(columns={"probability": "target_probability"})
    .merge(sim_degree_dist, on="degree", how="outer")
    .fillna(0)
    .sort_values("degree")
    .reset_index(drop=True)
)

# print(degree_check.to_string(index=False))

nodes_df = pd.DataFrame({
    "node": list(G.nodes()),
    "age": [node_age[node] for node in G.nodes()],
    "degree": [G.degree[node] for node in G.nodes()]
})

print(nodes_df.head())

#%% Monte Carlo:

#

age_values = np.arange(16, 101)

# If targetP accidentally has ego_age as a column, set it as index
if "ego_age" in targetP.columns:
    targetP = targetP.set_index("ego_age")

targetP.index = targetP.index.astype(int)
targetP.columns = targetP.columns.astype(int)

targetP = targetP.loc[age_values, age_values]

target_array = targetP.to_numpy(dtype=float)

target_rows = targetP.sum(axis=1).to_numpy() > 0

age_to_idx = {
    age: i
    for i, age in enumerate(age_values)
}

#

def counts_to_probs(counts):
    row_sums = counts.sum(axis=1, keepdims=True)

    probs = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums > 0
    )

    return probs

def initialise_friend_age_counts(G, node_age):
    counts = np.zeros(
        (len(age_values), len(age_values)),
        dtype=float
    )

    for u, v in G.edges():
        age_u = int(node_age[u])
        age_v = int(node_age[v])

        i = age_to_idx.get(age_u)
        j = age_to_idx.get(age_v)

        if i is not None and j is not None:
            counts[i, j] += 1
            counts[j, i] += 1

    return counts

###############################################################################
def loss_from_counts(counts):
    simP = counts_to_probs(counts)

    diff = simP[target_rows, :] - target_array[target_rows, :]

    # return np.mean(diff ** 2)
    return np.mean(np.abs(diff))

###############################################################################

#

def add_edge_contribution(counts, age_x, age_y, amount):
    i = age_to_idx.get(int(age_x))
    j = age_to_idx.get(int(age_y))

    if i is not None and j is not None:
        counts[i, j] += amount
        counts[j, i] += amount

def proposed_counts_after_swap(counts, u, v):
    new_counts = counts.copy()

    affected_edges = set()

    for nbr in G.neighbors(u):
        affected_edges.add(tuple(sorted((u, nbr))))

    for nbr in G.neighbors(v):
        affected_edges.add(tuple(sorted((v, nbr))))

    # Remove old contributions
    for x, y in affected_edges:
        add_edge_contribution(
            new_counts,
            node_age[x],
            node_age[y],
            -1
        )

    # Add new contributions after swapping ages of u and v
    for x, y in affected_edges:

        if x == u:
            age_x = node_age[v]
        elif x == v:
            age_x = node_age[u]
        else:
            age_x = node_age[x]

        if y == u:
            age_y = node_age[v]
        elif y == v:
            age_y = node_age[u]
        else:
            age_y = node_age[y]

        add_edge_contribution(
            new_counts,
            age_x,
            age_y,
            1
        )

    return new_counts

#

counts = initialise_friend_age_counts(G, node_age)

current_loss = loss_from_counts(counts)

best_loss = current_loss
best_counts = counts.copy()
best_node_age = node_age.copy()

print("Initial loss:", current_loss)

#

nodes = np.array(list(G.nodes()))

n_iter = 2000000

temperature = max(current_loss * 0.05, 1e-9)
cooling = 0.99995

history = []
accepted = 0

for it in range(1, n_iter + 1):

    u, v = rng.choice(nodes, size=2, replace=False)

    if node_age[u] == node_age[v]:
        continue

    new_counts = proposed_counts_after_swap(counts, u, v)
    new_loss = loss_from_counts(new_counts)

    delta = new_loss - current_loss

    if delta <= 0:
        accept = True
    else:
        accept = rng.random() < np.exp(
            -delta / max(temperature, 1e-12)
        )

    if accept:
        counts = new_counts
        node_age[u], node_age[v] = node_age[v], node_age[u]
        current_loss = new_loss
        accepted += 1

        if current_loss < best_loss:
            best_loss = current_loss
            best_counts = counts.copy()
            best_node_age = node_age.copy()

    temperature *= cooling

    if it % 1000 == 0:
        history.append({
            "iteration": it,
            "current_loss": current_loss,
            "best_loss": best_loss,
            "temperature": temperature,
            "accepted_swaps": accepted
        })

        print(
            f"Iteration {it}, "
            f"current loss = {current_loss:.8f}, "
            f"best loss = {best_loss:.8f}, "
            f"accepted swaps = {accepted}"
        )

history = pd.DataFrame(history)

#

node_age = best_node_age.copy()
counts = best_counts.copy()

nx.set_node_attributes(G, node_age, "age")

print("Best loss:", best_loss)

#

finalP = counts_to_probs(counts)

simP = pd.DataFrame(
    finalP,
    index=age_values,
    columns=age_values
)

simP.index.name = "ego_age"
simP.columns.name = "friend_age"

# simP.loc[16]
# targetP.loc[16]


#

sim_dist_netag_by_age = {
    ego_age: pd.DataFrame({
        "age": age_values,
        "probability": simP.loc[ego_age].to_numpy()
    })
    for ego_age in age_values
}

# print(sim_dist_netag_by_age[16].to_string(index=False))

# 

compare_age_16 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[16].to_numpy(),
    "simulated_probability": simP.loc[16].to_numpy()
})

# print(compare_age_16.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_16["age"],
    compare_age_16["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_16["age"],
    compare_age_16["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 16")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_20 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[20].to_numpy(),
    "simulated_probability": simP.loc[20].to_numpy()
})

# print(compare_age_20.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_20["age"],
    compare_age_20["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_20["age"],
    compare_age_20["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 20")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_25 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[25].to_numpy(),
    "simulated_probability": simP.loc[25].to_numpy()
})

# print(compare_age_25.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_25["age"],
    compare_age_25["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_25["age"],
    compare_age_25["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 25")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_30 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[30].to_numpy(),
    "simulated_probability": simP.loc[30].to_numpy()
})

# print(compare_age_30.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_30["age"],
    compare_age_30["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_30["age"],
    compare_age_30["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 30")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_40 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[40].to_numpy(),
    "simulated_probability": simP.loc[40].to_numpy()
})

# print(compare_age_40.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_40["age"],
    compare_age_40["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_40["age"],
    compare_age_40["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 40")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_50 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[50].to_numpy(),
    "simulated_probability": simP.loc[50].to_numpy()
})

# print(compare_age_50.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_50["age"],
    compare_age_50["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_50["age"],
    compare_age_50["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 50")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_60 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[60].to_numpy(),
    "simulated_probability": simP.loc[60].to_numpy()
})

# print(compare_age_60.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_60["age"],
    compare_age_60["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_60["age"],
    compare_age_60["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 60")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_70 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[70].to_numpy(),
    "simulated_probability": simP.loc[70].to_numpy()
})

# print(compare_age_70.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_70["age"],
    compare_age_70["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_70["age"],
    compare_age_70["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 70")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_80 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[80].to_numpy(),
    "simulated_probability": simP.loc[80].to_numpy()
})

# print(compare_age_80.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_80["age"],
    compare_age_80["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_80["age"],
    compare_age_80["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 80")
plt.legend()
plt.tight_layout()
plt.show()

#

compare_age_90 = pd.DataFrame({
    "age": age_values,
    "empirical_probability": targetP.loc[90].to_numpy(),
    "simulated_probability": simP.loc[90].to_numpy()
})

# print(compare_age_90.to_string(index=False))


plt.figure(figsize=(9, 6), dpi=1000)

plt.plot(
    compare_age_90["age"],
    compare_age_90["empirical_probability"],
    label="Empirical",
    linewidth=2
)

plt.plot(
    compare_age_90["age"],
    compare_age_90["simulated_probability"],
    label="Simulated",
    linewidth=2
)

plt.xlabel("Friend age")
plt.ylabel("Probability")
plt.title("Friend-age distribution for ego age 90")
plt.legend()
plt.tight_layout()
plt.show()

# 



#

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
plt.title("Age-swapping Monte Carlo convergence")
plt.legend()
plt.tight_layout()
plt.show()

#

#%%
































