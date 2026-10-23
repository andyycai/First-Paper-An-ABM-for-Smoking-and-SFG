#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Sep 13 00:00:00 2026

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

#%%

# -----------------------------------------------------------------------------

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Replace these example names with your actual DataFrame and column.
cbr_per_1000 = pd.read_excel("UK Crude Birth Rate.xlsx", index_col = 0)
cbr_per_1000.head()

#%%

PROJECTION_YEARS = pd.Index(range(1974, 2024), name="year")
TRANSITION_YEARS = PROJECTION_YEARS[:-1]

AGES = [str(age) for age in range(100)] + ["100+"]

# Observed population counts by age.
# These use your existing, prepared population table.
observed_population = (
    p_all.loc[PROJECTION_YEARS, AGES].astype(float).copy()
)

# Your previously calculated all-persons qx, including the
# documented infant and age-100-as-100+ approximations.
q_projection = (
    qx_all.loc[TRANSITION_YEARS, AGES].astype(float).copy()
)

# Extract the birth-rate column as a Series.
# Your original DataFrame remains unchanged.
birth_rate = cbr_per_1000["Crude Birth per 1000"].copy()

# Ensure the existing year index is numeric.
birth_rate.index = pd.Index(
    pd.to_numeric(birth_rate.index, errors="raise"),
    name="year"
)

if birth_rate.index.has_duplicates:
    raise ValueError("Crude birth rates contain duplicate years.")

# Convert values to numeric.
birth_rate = pd.to_numeric(birth_rate, errors="raise")

# Select the rates needed for the projection transitions.
# Convert births per 1,000 population to births per person-year.
birth_rate = birth_rate.reindex(TRANSITION_YEARS) / 1000.0

print(birth_rate.head())

# Validate inputs.
for name, table in [
    ("observed_population", observed_population),
    ("q_projection", q_projection),
    ("birth_rate", birth_rate),
]:
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError(f"{name}: missing or infinite values.")

if (observed_population < 0).any().any():
    raise ValueError("Population counts cannot be negative.")

if (observed_population.sum(axis=1) <= 0).any():
    raise ValueError("Annual total populations must be positive.")

if ((q_projection < 0) | (q_projection > 1)).any().any():
    raise ValueError("qx must be between 0 and 1.")

if (birth_rate < 0).any():
    raise ValueError("Crude birth rates cannot be negative.")

#%%

# Rows = population snapshots; columns = age groups.
simulated_population = pd.DataFrame(
    np.nan,
    index=PROJECTION_YEARS,
    columns=AGES
)

# Initialise ONLY from the observed 1974 age counts.
simulated_population.loc[1974] = observed_population.loc[1974]

flow_records = []

for year in TRANSITION_YEARS:

    current = simulated_population.loc[year].to_numpy(dtype=float)
    q = q_projection.loc[year].to_numpy(dtype=float)

    start_total = current.sum()

    # --------------------------------------------------------
    # A. Survival when moving between single-year age groups
    # --------------------------------------------------------
    # q[:-1] corresponds to ages 0–99.
    # q[1:] corresponds to ages 1–99 and the age-100 proxy.

    survival_to_next_age = (
        (1.0 - q[:-1])
        * (1.0 - 0.5 * q[1:])
        / (1.0 - 0.5 * q[:-1])
    )

    next_population = np.zeros(len(AGES), dtype=float)

    # Survivors aged 0–98 move into groups 1–99.
    next_population[1:100] = (
        current[:99] * survival_to_next_age[:99]
    )

    # Open-ended 100+ group:
    # survivors entering from 99 plus survivors already in 100+.
    # Existing 100+ residents use the age-100 probability proxy.
    next_population[100] = (
        current[99] * survival_to_next_age[99]
        + current[100] * (1.0 - q[100])
    )

    survivors_existing_population = next_population.sum()

    # --------------------------------------------------------
    # B. Births and survival of newborns
    # --------------------------------------------------------

    newborn_survival = 1.0 - 0.5 * q[0]
    b = birth_rate.loc[year]  # births per person-year

    # Let B = births and r = newborn survival.
    # Ending population = existing survivors + B*r.
    # B = b * (starting + ending population) / 2.
    # Solving gives the expression below.
    denominator = 2.0 - b * newborn_survival

    if denominator <= 0:
        raise ValueError(
            f"{year}: invalid birth-rate calculation. Check units."
        )

    births = (
        b * (start_total + survivors_existing_population)
        / denominator
    )

    next_population[0] = births * newborn_survival

    # Store next year's population.
    simulated_population.loc[year + 1] = next_population

    # --------------------------------------------------------
    # C. Record demographic flows
    # --------------------------------------------------------

    deaths_existing_population = (
        start_total - survivors_existing_population
    )

    deaths_among_newborns = births * (1.0 - newborn_survival)

    deaths = (
        deaths_existing_population + deaths_among_newborns
    )

    end_total = next_population.sum()
    average_population = 0.5 * (start_total + end_total)

    # Accounting check: no migration enters the calculation.
    if not np.isclose(
        end_total,
        start_total + births - deaths,
        rtol=1e-10,
        atol=1e-6
    ):
        raise RuntimeError(f"{year}: population balance failed.")

    flow_records.append(
        {
            "from_year": year,
            "to_year": year + 1,
            "start_population": start_total,
            "births": births,
            "deaths": deaths,
            "end_population": end_total,
            "model_cbr_per_1000": 1000 * births / average_population,
            "model_cdr_per_1000": 1000 * deaths / average_population,
        }
    )

projection_flows = (
    pd.DataFrame(flow_records)
    .set_index("from_year")
)

simulated_population.columns.name = "age"

print(simulated_population.head())
print(projection_flows.head())

#%%

observed_total = observed_population.sum(axis=1)
simulated_total = simulated_population.sum(axis=1)

# Age shares, expressed as fractions summing to 1.
observed_age_share = observed_population.div(
    observed_total, axis=0
)

simulated_age_share = simulated_population.div(
    simulated_total, axis=0
)

# Signed differences in people.
population_gap_by_age = (
    simulated_population - observed_population
)

# Signed differences in percentage points.
age_share_gap_pp = (
    100 * (simulated_age_share - observed_age_share)
)

comparison_summary = pd.DataFrame(
    {
        "observed_population_age_sum": observed_total,
        "simulated_population": simulated_total,
        "population_gap": simulated_total - observed_total,
        "population_gap_pct": (
            100 * (simulated_total - observed_total) / observed_total
        ),

        # Distribution discrepancy:
        # zero means identical age distributions.
        # Ranges from 0 to 100 percentage points.
        "age_distribution_total_variation_pp": (
            0.5 * age_share_gap_pp.abs().sum(axis=1)
        ),
    }
)

print(comparison_summary.loc[[1974, 1980, 1990, 2000, 2010, 2023]])

# Detailed comparison at every age in 2023.
comparison_2023 = pd.DataFrame(
    {
        "observed_population": observed_population.loc[2023],
        "simulated_population": simulated_population.loc[2023],
        "observed_share_pct": 100 * observed_age_share.loc[2023],
        "simulated_share_pct": 100 * simulated_age_share.loc[2023],
        "share_gap_pp": age_share_gap_pp.loc[2023],
    }
)

print(comparison_2023.to_string())

#%%

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Total population over time.
axes[0].plot(
    PROJECTION_YEARS,
    observed_total / 1_000_000,
    label="Observed"
)

axes[0].plot(
    PROJECTION_YEARS,
    simulated_total / 1_000_000,
    linestyle="--",
    label="Births-and-deaths-only"
)

axes[0].set_title("Total population")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Population (millions)")
axes[0].legend()
axes[0].grid(alpha=0.25)

# Age distribution in the final year.
age_positions = np.arange(len(AGES))

axes[1].plot(
    age_positions,
    100 * observed_age_share.loc[2023],
    label="Observed"
)

axes[1].plot(
    age_positions,
    100 * simulated_age_share.loc[2023],
    linestyle="--",
    label="Births-and-deaths-only"
)

axes[1].set_title("Age distribution in 2023")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Share of population (%)")
axes[1].set_xticks([0, 20, 40, 60, 80, 100])
axes[1].set_xticklabels(["0", "20", "40", "60", "80", "100+"])
axes[1].legend()
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()

#%%

# -----------------------------------------------------------------------------

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nm = pd.read_excel("UK Net Migration.xlsx", index_col = 0)

#%%

years = pd.Index(range(1974, 2024), name="year")

# Extract the column without modifying your original DataFrame.
net_migration = pd.to_numeric(
    nm["Net Migration (1000)"],
    errors="raise"
).copy()

# Use the existing year index.
net_migration.index = pd.Index(
    pd.to_numeric(net_migration.index, errors="raise"),
    name="year"
)

if net_migration.index.has_duplicates:
    raise ValueError("Migration data contain duplicate years.")

# Convert thousands of people into people.
net_migration = net_migration.reindex(years) * 1000.0

# Published observed population totals for the crude rate denominator.
population_for_rate = p_all["All Ages"].reindex(years)

# Existing population totals from your projection/comparison code.
observed = observed_total.reindex(years)
simulated = simulated_total.reindex(years)

# Check inputs.
for name, series in [
    ("net_migration", net_migration),
    ("population_for_rate", population_for_rate),
    ("observed_total", observed),
    ("simulated_total", simulated),
]:
    if not np.isfinite(series.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: missing or non-finite values.")

if (population_for_rate <= 0).any():
    raise ValueError("Population denominators must be positive.")

#%%

# Annual crude net migration rate per 1,000 population.
crude_net_migration_rate = (
    1000.0 * net_migration / population_for_rate
)

# Approximate migration between successive mid-year snapshots.
# For example, the value at 1975 represents:
# 0.5 * migration_1974 + 0.5 * migration_1975.
migration_between_snapshots = (
    0.5 * (net_migration.shift(1) + net_migration)
)

# The 1974 snapshot is the baseline: nothing accumulated yet.
migration_between_snapshots.loc[1974] = 0.0

cumulative_net_migration = migration_between_snapshots.cumsum()

# Use observed minus simulated so positive values indicate
# that the observed population exceeds the closed projection.
population_gap = observed - simulated

migration_comparison = pd.DataFrame(
    {
        "annual_net_migration_people": net_migration,
        "crude_net_migration_rate_per_1000": crude_net_migration_rate,
        "cumulative_net_migration_people": cumulative_net_migration,
        "observed_minus_simulated_people": population_gap,
        "gap_minus_cumulative_migration_people": (
            population_gap - cumulative_net_migration
        ),
    },
    index=years
)

print(migration_comparison.loc[[1974, 1975, 1980, 2000, 2023]])

#%%

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ------------------------------------------------------------
# Plot 1: annual crude net migration rate
# ------------------------------------------------------------

axes[0].plot(
    years,
    crude_net_migration_rate,
    color="tab:blue",
    linewidth=2
)

axes[0].axhline(
    0,
    color="black",
    linewidth=0.8,
    linestyle=":"
)

axes[0].set_title("Annual crude net migration rate")
axes[0].set_xlabel("Calendar year (assumed)")
axes[0].set_ylabel("Net migrants per 1,000 population")
axes[0].grid(alpha=0.25)


# ------------------------------------------------------------
# Plot 2: cumulative migration versus population gap
# ------------------------------------------------------------

axes[1].plot(
    years,
    cumulative_net_migration / 1_000_000,
    label="Cumulative net migration",
    color="tab:blue",
    linewidth=2
)

axes[1].plot(
    years,
    population_gap / 1_000_000,
    label="Observed − simulated population",
    color="tab:orange",
    linewidth=2,
    linestyle="--"
)

axes[1].axhline(
    0,
    color="black",
    linewidth=0.8,
    linestyle=":"
)

axes[1].set_title("Population gap and migration since mid-1974")
axes[1].set_xlabel("Mid-year population snapshot")
axes[1].set_ylabel("People (millions)")
axes[1].legend()
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()

# Optional:
# fig.savefig("migration_comparison.png", dpi=300, bbox_inches="tight")

