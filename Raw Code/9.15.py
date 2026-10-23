#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 15 00:00:00 2026

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

cdr_per_1000 = pd.read_excel("UK Crude Mortality Rate.xlsx", index_col = 0)
cdr_per_1000.head()

#%%

import pandas as pd
import matplotlib.pyplot as plt

# Work on a copy so the original DataFrame remains unchanged.
cdr_plot = cdr_per_1000.copy()

# Ensure years and mortality rates are numeric.
cdr_plot.index = pd.Index(
    pd.to_numeric(cdr_plot.index, errors="raise"),
    name="year"
)

columns = [
    "Crude Death per 1000 (WB)",
    "Crude Death per 1000 (ONS)"
]

cdr_plot[columns] = cdr_plot[columns].apply(
    pd.to_numeric,
    errors="raise"
)

cdr_plot = cdr_plot.sort_index()

# Create the comparison plot.
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    cdr_plot.index,
    cdr_plot["Crude Death per 1000 (WB)"],
    label="World Bank (WB)",
    color="tab:blue",
    linewidth=2
)

ax.plot(
    cdr_plot.index,
    cdr_plot["Crude Death per 1000 (ONS)"],
    label="Estimated from ONS life tables",
    color="tab:orange",
    linestyle="--",
    linewidth=2
)

# Identify years using your imputed mortality schedules.
ax.axvspan(
    1974, 1980,
    color="grey",
    alpha=0.15,
    label="Imputed mortality schedules (1974–1980)"
)

ax.set_title("UK crude mortality rates: World Bank and ONS-derived estimates")
ax.set_xlabel("Year")
ax.set_ylabel("Deaths per 1,000 population")

ax.set_xlim(cdr_plot.index.min(), cdr_plot.index.max())
ax.grid(alpha=0.25)
ax.legend(frameon=False)

plt.tight_layout()

# Optional: save the figure.
# fig.savefig("UK_crude_mortality_comparison.png",
#             dpi=300, bbox_inches="tight")

plt.show()

#%%

# -----------------------------------------------------------------------------

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nm = pd.read_excel("UK Net Migration.xlsx", index_col = 0)

#%%

#%%

# ============================================================
# 1. PREPARE INPUTS FOR POPULATION ACCOUNTING
# ============================================================

# Choose the mortality series used in the reconstruction.
# CDR_COLUMN = "Crude Death per 1000 (WB)"

# To use your ONS-derived mortality estimates instead:
CDR_COLUMN = "Crude Death per 1000 (ONS)"


def align_accounting_series(series, name):
    """Return a numeric Series aligned to the existing YEARS index."""
    out = pd.to_numeric(series, errors="raise").copy()

    out.index = pd.Index(
        pd.to_numeric(out.index, errors="raise"),
        name="year"
    )

    if out.index.has_duplicates:
        raise ValueError(f"{name}: duplicate years.")

    out = out.reindex(YEARS).astype(float)

    if not np.isfinite(out.to_numpy()).all():
        raise ValueError(
            f"{name}: missing or non-finite values for 1974–2023."
        )

    return out.rename(name)


accounting_inputs = pd.DataFrame(
    {
        "population": align_accounting_series(
            p_all["All Ages"],
            "population"
        ),

        "birth_rate_per_1000": align_accounting_series(
            cbr_per_1000["Crude Birth per 1000"],
            "birth_rate_per_1000"
        ),

        "death_rate_per_1000": align_accounting_series(
            cdr_per_1000[CDR_COLUMN],
            "death_rate_per_1000"
        ),

        # Convert migration from thousands into people.
        "net_migration": (
            1000.0 * align_accounting_series(
                nm["Net Migration (1000)"],
                "net_migration"
            )
        ),
    },
    index=YEARS
)

if (accounting_inputs["population"] <= 0).any():
    raise ValueError("Population must be positive.")

if (
    accounting_inputs[
        ["birth_rate_per_1000", "death_rate_per_1000"]
    ] < 0
).any().any():
    raise ValueError("Birth and death rates cannot be negative.")


#%%
# ============================================================
# 2. ESTIMATE CALENDAR-YEAR BIRTHS AND DEATHS
# ============================================================
# Counts = rate per 1,000 * observed population / 1,000

accounting_inputs["births"] = (
    accounting_inputs["birth_rate_per_1000"]
    * accounting_inputs["population"]
    / 1000.0
)

accounting_inputs["deaths"] = (
    accounting_inputs["death_rate_per_1000"]
    * accounting_inputs["population"]
    / 1000.0
)


#%%
# ============================================================
# 3. APPROXIMATE FLOWS BETWEEN MID-YEAR SNAPSHOTS
# ============================================================
# For the interval mid-(year - 1) to mid-year:
#
# interval flow = 0.5 * previous calendar-year flow
#                 + 0.5 * current calendar-year flow
#
# This assumes uniform flows within each calendar year.

flow_columns = ["births", "deaths", "net_migration"]

interval_flows = (
    accounting_inputs[flow_columns].shift(1)
    + accounting_inputs[flow_columns]
) / 2.0

# Index represents the ENDING year.
# There is no 1974 reconstruction because 1973 inputs are absent.
population_accounting = interval_flows.loc[1975:2023].copy()

population_accounting["previous_observed_population"] = (
    accounting_inputs["population"].shift(1)
)

population_accounting["observed_population"] = (
    accounting_inputs["population"]
)


#%%
# ============================================================
# 4. RECONSTRUCT THE POPULATION
# ============================================================
# Every interval starts from the previous OBSERVED population.
# The reconstructed population is NOT carried into the next step.

population_accounting["reconstructed_change"] = (
    population_accounting["births"]
    + population_accounting["net_migration"]
    - population_accounting["deaths"]
)

population_accounting["reconstructed_population"] = (
    population_accounting["previous_observed_population"]
    + population_accounting["reconstructed_change"]
)

population_accounting["observed_change"] = (
    population_accounting["observed_population"]
    - population_accounting["previous_observed_population"]
)

population_accounting["observed_minus_reconstructed"] = (
    population_accounting["observed_population"]
    - population_accounting["reconstructed_population"]
)

population_accounting["residual_pct_observed"] = (
    100.0
    * population_accounting["observed_minus_reconstructed"]
    / population_accounting["observed_population"]
)


#%%
# ============================================================
# 5. PREPARE THE STACKED COMPONENTS
# ============================================================

plot_years = population_accounting.index.to_numpy()

interval_births = population_accounting["births"]
interval_deaths = population_accounting["deaths"]
interval_migration = population_accounting["net_migration"]

# Separate positive and negative NET migration.
# These are not gross immigration and gross emigration.
positive_net_migration = interval_migration.clip(lower=0)
negative_net_migration = (-interval_migration).clip(lower=0)

# Subtract deaths and negative net migration before stacking
# the positive additions.
#
# This is an accounting remainder, not an independently
# estimated population of cohort survivors.
population_after_losses = (
    population_accounting["previous_observed_population"]
    - interval_deaths
    - negative_net_migration
)

if (population_after_losses < 0).any():
    raise ValueError(
        "Negative accounting baseline: check population and flow units."
    )

# Check: the top of the stack must equal reconstructed population.
stack_total = (
    population_after_losses
    + interval_births
    + positive_net_migration
)

if not np.allclose(
    stack_total,
    population_accounting["reconstructed_population"],
    rtol=1e-10,
    atol=1e-6
):
    raise RuntimeError("Stacked components do not reconcile.")


#%%
# ============================================================
# 6. CREATE THE TWO-PANEL FIGURE
# ============================================================

fig_accounting, axes = plt.subplots(
    2, 1,
    figsize=(14, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [1.3, 1]}
)

# ------------------------------------------------------------
# UPPER PANEL: stacked reconstructed population
# ------------------------------------------------------------

axes[0].stackplot(
    plot_years,
    population_after_losses.to_numpy() / 1_000_000,
    interval_births.to_numpy() / 1_000_000,
    positive_net_migration.to_numpy() / 1_000_000,
    labels=[
        "Previous population less deaths and negative net migration",
        "Births",
        "Positive net migration"
    ],
    colors=["#d9d9d9", "#f2c14e", "#81b29a"],
    alpha=0.85
)

axes[0].plot(
    plot_years,
    population_accounting["observed_population"].to_numpy()
    / 1_000_000,
    color="black",
    linewidth=2,
    label="Observed population"
)

axes[0].plot(
    plot_years,
    population_accounting["reconstructed_population"].to_numpy()
    / 1_000_000,
    color="tab:blue",
    linewidth=1.8,
    linestyle="--",
    label="Reconstructed population: top of stack"
)

axes[0].set_title("Observed and reconstructed UK population")
axes[0].set_ylabel("Population (millions)")
axes[0].set_ylim(bottom=0)
axes[0].legend(loc="lower left", fontsize=9, frameon=False)
axes[0].grid(axis="y", alpha=0.2)


# ------------------------------------------------------------
# LOWER PANEL: signed demographic contributions
# ------------------------------------------------------------

# Positive additions.
axes[1].bar(
    plot_years,
    interval_births.to_numpy() / 1000,
    color="#f2c14e",
    label="Births"
)

axes[1].bar(
    plot_years,
    positive_net_migration.to_numpy() / 1000,
    bottom=interval_births.to_numpy() / 1000,
    color="#81b29a",
    label="Positive net migration"
)

# Negative contributions.
axes[1].bar(
    plot_years,
    -interval_deaths.to_numpy() / 1000,
    color="#e07a5f",
    label="Deaths"
)

axes[1].bar(
    plot_years,
    -negative_net_migration.to_numpy() / 1000,
    bottom=-interval_deaths.to_numpy() / 1000,
    color="#8e7dbe",
    label="Negative net migration"
)

# Observed and reconstructed net population changes.
axes[1].plot(
    plot_years,
    population_accounting["observed_change"].to_numpy() / 1000,
    color="black",
    linewidth=2,
    label="Observed population change"
)

axes[1].plot(
    plot_years,
    population_accounting["reconstructed_change"].to_numpy() / 1000,
    color="tab:blue",
    linewidth=1.8,
    linestyle="--",
    label="Births + net migration − deaths"
)

axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Contributions between consecutive mid-year snapshots")
axes[1].set_xlabel("Ending year of the interval")
axes[1].set_ylabel("People (thousands)")
axes[1].legend(ncol=2, fontsize=9, frameon=False)
axes[1].grid(axis="y", alpha=0.2)
axes[1].set_xlim(plot_years.min() - 0.5, plot_years.max() + 0.5)

fig_accounting.suptitle(
    f"Population accounting using {CDR_COLUMN}",
    fontsize=13
)

fig_accounting.tight_layout(rect=[0, 0, 1, 0.96])

# Optional figure export:
# fig_accounting.savefig(
#     "UK_population_accounting.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.show()


#%%
# ============================================================
# 7. INSPECT AND OPTIONALLY SAVE RESULTS
# ============================================================

summary_columns = [
    "observed_population",
    "reconstructed_population",
    "observed_minus_reconstructed",
    "residual_pct_observed"
]

print(
    population_accounting.loc[
        [1975, 1980, 1990, 2000, 2010, 2023],
        summary_columns
    ].round(3).to_string()
)

# Optional Excel export:
# with pd.ExcelWriter("UK_population_accounting.xlsx") as writer:
#     accounting_inputs.to_excel(
#         writer, sheet_name="annual_inputs"
#     )
#     population_accounting.to_excel(
#         writer, sheet_name="population_accounting"
#     )
