#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat Sep 12 00:00:00 2026

@author: Andy.
"""

#%%

import pandas as pd

#%%

mx_male = pd.read_excel("Mortality mx - UK Male.xlsx", index_col = 0)

mx_female = pd.read_excel("Mortality mx - UK Female.xlsx", index_col = 0)

#%%

qx_male = pd.read_excel("Mortality qx - UK Male.xlsx", index_col = 0)

qx_female = pd.read_excel("Mortality qx - UK Female.xlsx", index_col = 0)

#%%

pe_male = pd.read_excel("Population Estimates - UK Male.xlsx", index_col = 0)

pe_female = pd.read_excel("Population Estimates - UK Female.xlsx", index_col = 0)

pe_pop = pd.read_excel("Population Estimates - UK Population.xlsx", index_col = 0)

#%%

mx_male.head()
mx_female.head()

qx_male.head()
qx_female.head()

pe_male.head()
pe_female.head()
pe_pop.head()

#%%

import numpy as np
import pandas as pd
import warnings

YEARS = pd.Index(range(1974, 2024), name="year")
AGES = [str(age) for age in range(100)] + ["100+"]
POP_COLUMNS = AGES + ["All Ages"]


def normalise_column(column):
    """Handle labels stored as integers, floats, or strings."""
    label = str(column).strip()

    # For example: "1.0" -> "1"
    if label.endswith(".0") and label[:-2].isdigit():
        label = label[:-2]

    return label


def prepare_table(df, columns, name):
    """
    Use the existing year index.
    Select the required years/columns and validate numeric values.
    """
    out = df.copy()

    out.columns = [normalise_column(c) for c in out.columns]
    out.index = pd.Index(
        pd.to_numeric(out.index, errors="raise"),
        name="year"
    )

    if out.index.has_duplicates or out.columns.has_duplicates:
        raise ValueError(f"{name}: duplicate years or columns.")

    missing_years = YEARS.difference(out.index)
    missing_columns = pd.Index(columns).difference(out.columns)

    if len(missing_years) or len(missing_columns):
        raise ValueError(
            f"{name}: missing years {missing_years.tolist()}; "
            f"missing columns {missing_columns.tolist()}."
        )

    # Select by labels, not by column position.
    # This also removes Gender and any other unused columns.
    out = out.loc[YEARS, columns].copy()
    out.index = YEARS
    out = out.apply(pd.to_numeric, errors="raise").astype(float)

    if not np.isfinite(out.to_numpy()).all():
        raise ValueError(f"{name}: missing or infinite values.")

    if (out < 0).any().any():
        raise ValueError(f"{name}: negative values.")

    return out


# Central mortality rates
m_m = prepare_table(mx_male, AGES, "mx_male")
m_f = prepare_table(mx_female, AGES, "mx_female")

# Mortality probabilities
q_m = prepare_table(qx_male, AGES, "qx_male")
q_f = prepare_table(qx_female, AGES, "qx_female")

for name, table in [("qx_male", q_m), ("qx_female", q_f)]:
    if (table > 1).any().any():
        raise ValueError(f"{name}: qx must be between 0 and 1.")

# Population estimates
p_m = prepare_table(pe_male, POP_COLUMNS, "pe_male")
p_f = prepare_table(pe_female, POP_COLUMNS, "pe_female")
p_all = prepare_table(pe_pop, POP_COLUMNS, "pe_pop")

for name, table in [
    ("pe_male", p_m),
    ("pe_female", p_f),
    ("pe_pop", p_all)
]:
    if (table["All Ages"] <= 0).any():
        raise ValueError(f"{name}: All Ages population must be positive.")


#%%

population_diagnostics = pd.DataFrame(index=YEARS)

for label, table in [
    ("male", p_m),
    ("female", p_f),
    ("persons", p_all)
]:
    age_sum = table[AGES].sum(axis=1)
    reported_total = table["All Ages"]
    gap = age_sum - reported_total

    population_diagnostics[f"{label}_age_sum"] = age_sum
    population_diagnostics[f"{label}_reported_total"] = reported_total
    population_diagnostics[f"{label}_gap"] = gap
    population_diagnostics[f"{label}_gap_pct"] = (
        100 * gap / reported_total
    )

# A separate check on published totals, not estimated age sums.
population_diagnostics["published_sex_total_gap"] = (
    p_m["All Ages"] + p_f["All Ages"] - p_all["All Ages"]
)

if (
    population_diagnostics["published_sex_total_gap"].abs() > 1
).any():
    warnings.warn(
        "Published male + female All Ages totals differ from the "
        "published Persons total by more than one person. "
        "Check population_diagnostics and confirm population coverage.",
        stacklevel=2
    )

print("Maximum absolute age-sum discrepancy (% of reported population):")
print(
    population_diagnostics[
        ["male_gap_pct", "female_gap_pct", "persons_gap_pct"]
    ].abs().max()
)


#%%

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1. Population by age and sex
# ------------------------------------------------------------

male_population = p_m.loc[YEARS, AGES]
female_population = p_f.loc[YEARS, AGES]

combined_population = male_population + female_population # Or: p_all[AGES]

# Undefined combined rates if an age group has zero population.
age_denominator = combined_population.where(
    combined_population > 0
)


# ------------------------------------------------------------
# 2. Estimated deaths by age and sex
# ------------------------------------------------------------
# Age 0:
# Uses the supplied life-table-derived m0 as a proxy for the
# infant deaths-to-population rate.
#
# 100+:
# Uses the original age-100 mx as a proxy for everyone aged 100+.

estimated_deaths_male_by_age = (
    m_m.loc[YEARS, AGES] * male_population
)

estimated_deaths_female_by_age = (
    m_f.loc[YEARS, AGES] * female_population
)

estimated_deaths_all_by_age = (
    estimated_deaths_male_by_age
    + estimated_deaths_female_by_age
)


# ------------------------------------------------------------
# 3. All-persons age-specific mx
# ------------------------------------------------------------

mx_all = estimated_deaths_all_by_age / age_denominator

mx_all.index.name = "year"
mx_all.columns.name = "age"


# ------------------------------------------------------------
# 4. Modelled all-persons qx, using one formula at every age
# ------------------------------------------------------------
# Age 0: modelled infant probability.
# Ages 1–99: ONS one-year age-interval conversion.
# 100+: one-year probability proxy, NOT an open-ended life-table q.

qx_all = 2.0 * mx_all / (2.0 + mx_all)

invalid_qx = (qx_all < 0) | (qx_all > 1)

if invalid_qx.any().any():
    raise ValueError(
        "The conversion produced qx outside [0, 1]. "
        "Check mx units and the conversion assumptions. "
        "Do not simply clip the results."
    )

qx_all.index.name = "year"
qx_all.columns.name = "age"


# ------------------------------------------------------------
# 5. Total estimated deaths across ALL ages
# ------------------------------------------------------------

deaths_male = estimated_deaths_male_by_age.sum(axis=1)
deaths_female = estimated_deaths_female_by_age.sum(axis=1)
deaths_all = deaths_male + deaths_female


# ------------------------------------------------------------
# 6. Yearly crude mortality estimates
# ------------------------------------------------------------
# Use reported All Ages totals as denominators.

yearly_mortality = pd.DataFrame(
    {
        "population_all": p_all.loc[YEARS, "All Ages"],

        "estimated_deaths_male": deaths_male,
        "estimated_deaths_female": deaths_female,
        "estimated_deaths_all": deaths_all,

        "cdr_male_per_1000": (
            1000 * deaths_male / p_m.loc[YEARS, "All Ages"]
        ),
        "cdr_female_per_1000": (
            1000 * deaths_female / p_f.loc[YEARS, "All Ages"]
        ),
        "cdr_all_per_1000": (
            1000 * deaths_all / p_all.loc[YEARS, "All Ages"]
        ),

        # Keep boundary-age contributions visible.
        "estimated_deaths_age_0": (
            estimated_deaths_all_by_age["0"]
        ),
        "estimated_deaths_age_100plus": (
            estimated_deaths_all_by_age["100+"]
        ),
    },
    index=YEARS
)

# Distinguish earlier imputation from centred three-year rates.
yearly_mortality["rate_status"] = np.where(
    YEARS.to_numpy() <= 1980,
    "imputed_constant_rates",
    "centred_three_year_rates"
)


# ------------------------------------------------------------
# 7. Record how to interpret the age-specific outputs
# ------------------------------------------------------------

rate_notes = pd.DataFrame(index=pd.Index(AGES, name="age"))

rate_notes["interpretation"] = (
    "Population-weighted mx; qx from the ONS conversion"
)

rate_notes.loc["0", "interpretation"] = (
    "Supplied life-table m0 used as a deaths/exposure proxy; "
    "combined q0 is modelled"
)

rate_notes.loc["100+", "interpretation"] = (
    "Original age-100 mx applied to all people aged 100+; "
    "qx is a modelled one-year probability proxy"
)


# Inspect results.
print("\nAll-persons mx:")
print(mx_all.head())

print("\nModelled all-persons qx:")
print(qx_all.head())

print("\nYearly crude mortality estimates:")
print(
    yearly_mortality[
        [
            "estimated_deaths_all",
            "cdr_all_per_1000",
            "rate_status",
        ]
    ].head(10)
)


#%%

with pd.ExcelWriter("derived_mortality_1974_2023.xlsx") as writer:

    mx_all.to_excel(
        writer,
        sheet_name="mx_all"
    )

    qx_all.to_excel(
        writer,
        sheet_name="qx_all"
    )

    yearly_mortality.to_excel(
        writer,
        sheet_name="yearly_mortality"
    )

    rate_notes.to_excel(
        writer,
        sheet_name="rate_notes"
    )

    population_diagnostics.to_excel(
        writer,
        sheet_name="population_diagnostics"
    )
        
    age_denominator.to_excel(
        writer,
        sheet_name="population_estimates"
    )


#%% DOUBLE CHECK !!!

a = p_m[AGES] + p_f[AGES] - p_all[AGES]
aa = (p_m[AGES] + p_f[AGES] - p_all[AGES]) / p_all[AGES]

b = p_m[AGES].sum(axis=1) - p_m["All Ages"]
bb = p_f[AGES].sum(axis=1) - p_f["All Ages"]
bbb = p_all[AGES].sum(axis=1) - p_all["All Ages"]
bbbb = p_m[AGES].sum(axis=1) + p_f[AGES].sum(axis=1) - p_all[AGES].sum(axis=1)
bbbbb = p_m[AGES].sum(axis=1) + p_f[AGES].sum(axis=1) - p_all["All Ages"]






