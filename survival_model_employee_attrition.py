
#%% Introduction
"""
===============================================================================
DISCRETE-TIME SURVIVAL ANALYSIS — Employee Attrition
===============================================================================

Goal: estimate *when* employees leave a company and *what factors* accelerate
or delay their departure.  We mirror every analytical step from the child-
mortality demo but swap in the IBM HR Employee Attrition dataset so the
concepts land in a more everyday context.

What is "discrete-time survival analysis"?
------------------------------------------
Ordinary logistic regression asks "did the event happen — yes or no?"
Survival analysis asks "WHEN did the event happen, and what if we never saw
it happen (censoring)?"  In discrete time we chop the timeline into periods
(here: years at the company) and model the probability of the event in each
period, conditional on having survived to that period.

Key vocabulary
--------------
  - **Event**: the thing we're waiting for (here: employee quits).
  - **Censoring**: the employee is still working when data collection ends,
    so we know they survived *at least* this long, but not how much longer
    they would stay.  Censoring is the whole reason we need survival models
    instead of plain regression.
  - **Hazard  h(t)**: probability of quitting in year t, given that the
    employee has survived (stayed) through year t-1.
  - **Survival  S(t)**: probability of still being employed past year t.
    S(t) = product of (1 - h(1)) * (1 - h(2)) * ... * (1 - h(t)).
  - **Cumulative hazard  H(t)**: running total of the period hazards.  It
    measures accumulated risk over time.

Why the complementary log-log (cloglog) link?
----------------------------------------------
The cloglog link   log(-log(1 - h)) = Xβ   is special: it is the exact
discrete-time counterpart of the continuous-time Cox proportional-hazards
model.  Coefficients have a proportional-hazards interpretation — a one-unit
change in a covariate multiplies the *underlying continuous hazard* by
exp(β).  Logit works too but gives an odds-ratio interpretation instead.

Pipeline overview
-----------------
  1.  Load one-row-per-employee data.
  2.  Expand to person-period format (one row per employee-year).
  3.  Aggregate identical covariate patterns into binomial counts.
  4.  Fit a binomial cloglog model with Bayesian MCMC (Bambi + NumPyro).
  5.  Extract posterior survival curves, compare subgroups.
  6.  Re-fit with a smooth spline baseline, and with a Poisson formulation.
  7.  Compare models with LOO cross-validation.
"""

#%% Imports
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#%% ── 1. LOAD & PREPARE DATA ────────────────────────────────────────────────
"""
The IBM HR dataset has 1 470 employees with 35 attributes.
Key columns for us:
  - YearsAtCompany : how long the employee has been at the company (our time axis)
  - Attrition      : "Yes" / "No" (our event indicator)
  - Department, Gender, OverTime, JobRole : covariates

Mapping to survival concepts:
  subject  →  employee
  time     →  YearsAtCompany  (integer years, already discrete)
  event    →  Attrition == "Yes"  (1 = quit, 0 = still employed / censored)
"""

df = pd.read_csv("data/emp_attrition.csv")

# Recode to numeric event indicator
df["event"] = (df["Attrition"] == "Yes").astype(int)

# YearsAtCompany == 0 means "less than 1 year"; bump to 1 so every employee
# contributes at least one person-period row.
df["duration"] = df["YearsAtCompany"].clip(lower=1)

# Simplify OverTime to a 0/1 flag
df["overtime"] = (df["OverTime"] == "Yes").astype(int)

# Quick look at the data
print(f"Employees           : {len(df):,}")
print(f"Attrition rate      : {df['event'].mean():.1%}")
print(f"Median tenure (yrs) : {df['duration'].median():.0f}")
print()
print(df[["duration", "event"]].describe().round(2))
print()
print("Department counts:")
print(df["Department"].value_counts())
print()
print("Gender counts:")
print(df["Gender"].value_counts())


#%% ── 2. PERSON-PERIOD EXPANSION ────────────────────────────────────────────
"""
WHY do we need this?

Our raw data has one row per employee:
    Employee 42:  duration=5, event=1  (quit in year 5)

But a discrete-time survival model needs to see *each year the employee was
at risk*.  We expand Employee 42 into 5 rows:

    Employee 42, period=1, event=0   (survived year 1)
    Employee 42, period=2, event=0   (survived year 2)
    Employee 42, period=3, event=0   (survived year 3)
    Employee 42, period=4, event=0   (survived year 4)
    Employee 42, period=5, event=1   ← quit here

A censored employee (event=0) gets all zeros — they survived every observed
period, and we simply don't know what happens after.

This is the fundamental trick: we've turned a time-to-event problem into a
sequence of binary outcomes that a standard GLM can handle.
"""


def create_person_period_data(df, id_col, time_col, event_col, covariates):
    """
    Expand one-row-per-subject survival data into person-period format.

    Each subject gets one row for every discrete time period they were
    observed.  The event column is 1 only in the final period *if* the
    event actually occurred (otherwise 0 = censored).
    """
    records = []
    for _, row in df.iterrows():
        duration = int(np.ceil(row[time_col]))
        for t in range(1, duration + 1):
            is_final = t == duration
            event_in_period = 1 if (is_final and row[event_col] == 1) else 0
            record = {id_col: row[id_col], "period": t, "event": event_in_period}
            for cov in covariates:
                record[cov] = row[cov]
            records.append(record)
    return pd.DataFrame(records)


df_long = create_person_period_data(
    df,
    id_col="EmployeeNumber",
    time_col="duration",
    event_col="event",
    covariates=["Gender", "Department", "overtime", "JobRole"],
)

print(f"Person-period rows : {len(df_long):,}")
print(f"Total quit events  : {df_long['event'].sum():,}")


#%% ── 3. BINOMIAL AGGREGATION ───────────────────────────────────────────────
"""
WHY aggregate?

After expansion we might have hundreds of thousands of rows, but many are
identical in their covariates.  For example, all male Sales employees with
no overtime in period 3 share the same covariate pattern.  Instead of
feeding the model 200 separate Bernoulli rows, we collapse them into one
binomial observation:  events=12, at_risk=200.

The likelihood is identical (product of Bernoullis = Binomial) but the
dataset shrinks dramatically, making MCMC much faster.
"""


def aggregate_person_period(df_long, group_cols, event_col="event"):
    """
    Collapse person-period rows with identical covariates into binomial
    counts:  events (sum of 1s) and at_risk (number of rows).
    """
    return (
        df_long.groupby(group_cols)
        .agg(events=(event_col, "sum"), at_risk=(event_col, "count"))
        .reset_index()
    )


df_binomial = aggregate_person_period(
    df_long,
    group_cols=["period", "Gender", "Department", "overtime"],
)

print(f"Aggregated rows    : {len(df_binomial):,}")
print(f"vs person-period   : {len(df_long):,}")
print(f"Compression ratio  : {len(df_long) / len(df_binomial):.1f}x")

# Set ordered categoricals so Bambi picks sensible reference levels
df_binomial["Gender"] = pd.Categorical(
    df_binomial["Gender"], categories=["Male", "Female"], ordered=True
)
df_binomial["Department"] = pd.Categorical(
    df_binomial["Department"],
    categories=["Research & Development", "Sales", "Human Resources"],
    ordered=True,
)
df_binomial.head(10)


#%% ── 4. FIT THE CLOGLOG MODEL ──────────────────────────────────────────────
"""
Model specification
-------------------
    p(events, at_risk) ~ Gender + Department + overtime
                         + period + scale(...)

  - p(events, at_risk) : binomial response — events out of at_risk trials.
  - Gender, Department : categorical covariates (reference-coded).
  - overtime           : binary covariate.
  - period             : treated as *categorical* (`categorical="period"`),
      giving a separate intercept for each year — this is a fully
      nonparametric baseline hazard (no smoothness assumed).
  - family="binomial", link="cloglog" : the discrete-time proportional
      hazards model.

Interpretation of coefficients:
  exp(β) is a *hazard ratio*.  For example, if β_overtime = 0.5, then
  employees who work overtime have exp(0.5) ≈ 1.65× the baseline hazard
  of quitting in any given year.
"""

# Cap period at 20 to avoid sparse cells in the tail
df_model = df_binomial[df_binomial["period"] <= 20].copy()

model_cloglog = bmb.Model(
    "p(events, at_risk) ~ Gender + Department + overtime + period",
    data=df_model,
    family="binomial",
    link="cloglog",
    categorical="period",
)

results_cloglog = model_cloglog.fit(
    chains=4,
    draws=1000,
    random_seed=42,
    inference_method="numpyro",
    idata_kwargs={"log_likelihood": True},
)

# Forest plot of the period (baseline hazard) parameters
az.plot_forest(results_cloglog, var_names="period", combined=True, r_hat=True)
plt.title("Baseline hazard parameters (one per year)")
plt.tight_layout()
plt.show()


#%% ── 5. DERIVE SURVIVAL CURVES ─────────────────────────────────────────────
"""
How survival curves are computed from the model
------------------------------------------------
The model gives us posterior draws of h(t) — the hazard probability in each
period.  From those draws we compute:

  Survival:         S(t) = ∏_{k=1}^{t} (1 - h(k))
  Cumulative hazard: H(t) = Σ_{k=1}^{t} h(k)

We do this *per posterior draw* to propagate uncertainty, then summarize
with means and 94% credible intervals (3rd–97th percentile).

The function below works for both the binomial/cloglog model (draws are
hazard probabilities) and the Poisson model (draws are rates, clipped to
[0,1] for coherence).
"""


def derive_survival_curves(model, results, pred_df, model_type="cloglog"):
    """
    From a fitted Bambi model, compute posterior hazard, cumulative hazard,
    and survival curves with uncertainty bands.
    """
    predictions = model.predict(results, data=pred_df, inplace=False)

    var_name = "p" if model_type == "cloglog" else "mu"
    hazard = predictions.posterior[var_name]

    if model_type == "poisson":
        hazard = hazard.clip(0, 1)

    cumulative_hazard = hazard.cumsum("__obs__")
    survival = (1 - hazard).cumprod("__obs__")

    hazard_mean = hazard.mean(("chain", "draw"))
    cum_hazard_mean = cumulative_hazard.mean(("chain", "draw"))
    survival_mean = survival.mean(("chain", "draw"))

    # 94% credible intervals
    hazard_hdi = hazard.quantile((0.03, 0.97), dim=("chain", "draw"))
    cum_hazard_hdi = cumulative_hazard.quantile((0.03, 0.97), dim=("chain", "draw"))
    survival_hdi = survival.quantile((0.03, 0.97), dim=("chain", "draw"))

    return {
        "hazards": hazard_mean,
        "cum_hazards": cum_hazard_mean,
        "survival": survival_mean,
        "hazards_hdi": hazard_hdi,
        "cum_hazards_hdi": cum_hazard_hdi,
        "survival_hdi": survival_hdi,
    }


#%% ── 6. PLOTTING UTILITY ───────────────────────────────────────────────────

def plot_survival_curves(scenarios, title="Survival Analysis", show_hdi=True):
    """
    Three-panel plot: hazard h(t), cumulative hazard H(t), survival S(t).
    Each scenario is one line (e.g., a department or gender).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(scenarios), 2)))

    for i, sc in enumerate(scenarios):
        periods = np.arange(1, len(sc["hazards"]) + 1)

        # ── Hazard ──
        axes[0].step(periods, sc["hazards"], where="mid",
                     label=sc["label"], color=colors[i], linewidth=2)
        if show_hdi and "hazards_hdi" in sc:
            axes[0].fill_between(periods, sc["hazards_hdi"][0],
                                 sc["hazards_hdi"][1],
                                 color=colors[i], alpha=0.15, step="mid")

        # ── Cumulative hazard ──
        axes[1].step(periods, sc["cum_hazards"], where="mid",
                     label=sc["label"], color=colors[i], linewidth=2)
        if show_hdi and "cum_hazards_hdi" in sc:
            axes[1].fill_between(periods, sc["cum_hazards_hdi"][0],
                                 sc["cum_hazards_hdi"][1],
                                 color=colors[i], alpha=0.15, step="mid")

        # ── Survival (prepend S(0) = 1.0) ──
        surv_plot = np.insert(sc["survival"], 0, 1.0)
        time_plot = np.insert(periods, 0, 0)
        axes[2].step(time_plot, surv_plot, where="post",
                     label=sc["label"], color=colors[i], linewidth=2)
        if show_hdi and "survival_hdi" in sc:
            lo = np.insert(sc["survival_hdi"][0], 0, 1.0)
            hi = np.insert(sc["survival_hdi"][1], 0, 1.0)
            axes[2].fill_between(time_plot, lo, hi,
                                 color=colors[i], alpha=0.15, step="post")

    axes[0].set_title("Hazard $h(t)$", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("P(Quit in Year $t$ | Stayed to $t$)")

    axes[1].set_title("Cumulative Hazard $H(t)$", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Accumulated Risk")

    axes[2].set_title("Survival $S(t)$", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("P(Still Employed Past Year $t$)")
    axes[2].set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xlabel("Years at Company")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


#%% ── 7. SURVIVAL CURVES BY DEPARTMENT ───────────────────────────────────────
"""
We create a "prediction dataframe" for each scenario: hold all covariates
constant except the one we want to compare.  Here we loop over departments,
fixing Gender="Male", overtime=0.  The model then predicts the hazard at
each period for that covariate profile.
"""

t_range = np.arange(1, 21)  # years 1–20


def make_pred_df(department, gender="Male", overtime=0, periods=t_range):
    """Build a prediction dataframe for one covariate scenario."""
    return pd.DataFrame({
        "period": periods,
        "Gender": [gender] * len(periods),
        "Department": [department] * len(periods),
        "overtime": [overtime] * len(periods),
        "at_risk": [1] * len(periods),  # per-person prediction
    })


scenarios_dept = []
for dept in ["Research & Development", "Sales", "Human Resources"]:
    pred_df = make_pred_df(dept)
    curves = derive_survival_curves(model_cloglog, results_cloglog, pred_df)
    curves["label"] = dept
    scenarios_dept.append(curves)

plot_survival_curves(
    scenarios_dept,
    title="Employee Retention by Department (Male, No Overtime)",
)


#%% ── 8. SURVIVAL CURVES BY GENDER ──────────────────────────────────────────

scenarios_gender = []
for gender in ["Male", "Female"]:
    pred_df = make_pred_df("Research & Development", gender=gender)
    curves = derive_survival_curves(model_cloglog, results_cloglog, pred_df)
    curves["label"] = gender
    scenarios_gender.append(curves)

plot_survival_curves(
    scenarios_gender,
    title="Employee Retention by Gender (R&D, No Overtime)",
)


#%% ── 9. EFFECT OF OVERTIME ─────────────────────────────────────────────────
"""
Overtime is the attrition analogue of "illegitimacy" in the child-mortality
example — a binary risk factor we expect to strongly predict the event.
"""

scenarios_ot = []
for ot in [0, 1]:
    pred_df = make_pred_df("Research & Development", overtime=ot)
    curves = derive_survival_curves(model_cloglog, results_cloglog, pred_df)
    curves["label"] = "Overtime" if ot else "No Overtime"
    scenarios_ot.append(curves)

plot_survival_curves(
    scenarios_ot,
    title="Employee Retention: Overtime vs No Overtime (R&D, Male)",
)


#%% ── 10. SPLINE BASELINE HAZARD ────────────────────────────────────────────
"""
The categorical-period model estimates a free parameter for every year —
flexible but noisy, especially in later years with few employees.

A B-spline baseline  bs(period, df=5)  forces the hazard to be a smooth
function of time using 5 basis functions instead of 20 free parameters.
This trades flexibility for stability: we assume the baseline hazard
changes smoothly, which is reasonable for attrition (no reason to expect
the year-7 hazard to be wildly different from year-6).
"""

model_spline = bmb.Model(
    "p(events, at_risk) ~ Gender + Department + overtime + bs(period, df=5)",
    data=df_model,
    family="binomial",
    link="cloglog",
)

results_spline = model_spline.fit(
    draws=1000,
    chains=4,
    random_seed=2535,
    inference_method="numpyro",
    idata_kwargs={"log_likelihood": True},
)

az.summary(results_spline, var_names=["bs(period, df=5)"])

#%% Spline model: survival by department

scenarios_spline = []
for dept in ["Research & Development", "Sales", "Human Resources"]:
    pred_df = make_pred_df(dept)
    curves = derive_survival_curves(model_spline, results_spline, pred_df)
    curves["label"] = dept
    scenarios_spline.append(curves)

plot_survival_curves(
    scenarios_spline,
    title="Employee Retention by Department (Spline Baseline)",
)


#%% ── 11. POISSON ALTERNATIVE ───────────────────────────────────────────────
"""
Why a Poisson model for binary data?
-------------------------------------
There's a classical equivalence: when events are rare, a Poisson regression
on event counts with an offset for log(at_risk) gives nearly identical
results to a binomial model.  The log link of the Poisson directly models
the log-hazard rate.

This is useful because:
  - It extends naturally to models with time-varying exposures.
  - Software for Poisson regression is universal.
  - The offset term handles varying exposure (at_risk) cleanly.

We fit both and compare with LOO (leave-one-out cross-validation) to see
which the data prefer.
"""

model_poisson = bmb.Model(
    "events ~ Gender + Department + overtime + period + offset(np.log(at_risk))",
    data=df_model,
    family="poisson",
    categorical="period",
)

results_poisson = model_poisson.fit(
    draws=1000,
    chains=4,
    random_seed=42,
    inference_method="numpyro",
    idata_kwargs={"log_likelihood": True},
)


#%% ── 12. MODEL COMPARISON (LOO) ────────────────────────────────────────────
"""
LOO-CV estimates out-of-sample predictive accuracy.  Lower ELPD (expected
log pointwise predictive density) is worse.  The model with the highest
ELPD is preferred.  The "weight" column gives approximate model-averaging
weights.
"""

comparison = az.compare(
    {"Cloglog": results_cloglog, "Poisson": results_poisson},
    ic="loo",
)
print(comparison)


#%% ── 13. CLOGLOG vs POISSON SURVIVAL CURVES ────────────────────────────────
"""
Even when two models have similar LOO scores, their implied survival curves
can diverge — especially in the tails where data is sparse.  Plotting them
together is a good sanity check.
"""

pred_baseline = make_pred_df(department="Research & Development", gender="Male")

curves_clog = derive_survival_curves(model_cloglog, results_cloglog, pred_baseline)
curves_clog["label"] = "Cloglog (Binomial)"

curves_pois = derive_survival_curves(
    model_poisson, results_poisson, pred_baseline, model_type="poisson"
)
curves_pois["label"] = "Poisson"

plot_survival_curves(
    [curves_clog, curves_pois],
    title="Cloglog vs Poisson: R&D Male, No Overtime",
)
