
#%% import descirption data
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, erf
"""
-Case study: historical child mortality
Now let's apply these methods to real data. We'll analyze historical child mortality data, examining how survival varied by social class, sex, and birth year. We draw this example and inspiration from Goran Brostrom's Event History Analysis with R. This is a compelling example because the Swedish official statistics data captures a number of demographic attributes for each person that allows us to assess the risk of childhood death across various strata of the population. It also allows us to showcase how to prepare the data to make a computationally demanding sample more directly amenable to study.

-About the data
This dataset comes from historical parish records in 19th-century Sweden (roughly 1850–1890), where churches maintained detailed registers of births, deaths, and family circumstances. Child mortality was common during this period because of infectious disease, malnutrition, or limited medical care meant that a substantial fraction of children did not survive to age 15. The data lets us examine how survival varied across social strata at a time when class differences in living conditions were stark.

1. Data prep (lines 17-29) — Load one-row-per-child survival data with columns: id, exit (age at death/censoring), event (1=died, 0=censored), plus covariates (sex, socBranch, m.age, illeg, birthdate). Bin birth year into decades.

2. Person-period expansion (lines 32-86) — The key transformation. Each child gets exploded into one row per year they were alive. If a child died at age 5, they get rows for periods 1–5, with event=1 only in period 5. This is how you turn survival data into something a GLM can eat — each row is a Bernoulli trial: "did the child die in this period, given they survived to it?"

3. Binomial aggregation (lines 89-127) — Since many children share the same covariate pattern (e.g., male + worker + legitimate + 1860s), you can collapse identical person-period rows into counts: events (how many died) and at_risk (how many were observed). This shrinks the dataset dramatically and makes MCMC feasible.

4. Cloglog model (lines 130-145) — Fits a binomial GLM with a complementary log-log link. Why cloglog? It's the discrete-time analogue of the Cox proportional hazards model. The cloglog link means log(-log(1-h(t))) = X*beta, which is exactly the grouped continuous-time hazard. period is treated as categorical = fully nonparametric baseline hazard (one parameter per year of age).

5. Derive survival curves (lines 149-205) — Takes posterior draws of the hazard h(t), then computes:

Cumulative hazard: H(t) = sum of h(t)
Survival: S(t) = product of (1 - h(t)) across periods
Uncertainty bands come from doing this per posterior draw, then taking quantiles
6. Plot by social class & sex (lines 296-335) — Creates counterfactual prediction DataFrames (hold everything constant, vary one covariate), derives survival curves, plots.

7. Spline baseline (lines 338-366) — Replaces the 15 categorical period dummies with bs(period, df=4) — a smooth B-spline baseline hazard. Fewer parameters, smoother curves, but assumes the baseline hazard is smooth.

8. Poisson alternative (lines 369-402) — Fits the same model but as Poisson regression on event counts with offset(log(at_risk)). This is mathematically equivalent to the binomial model when events are rare (the Poisson-binomial equivalence for grouped survival data). Compares them with LOO.
"""
#%% data
child = pd.read_csv("data/child_raw.csv")
child["birth_year"] = pd.to_datetime(child["birthdate"]).dt.year
# child["birth_month"] = pd.to_datetime(child["birthdate"]).dt.month
# child["birth_day"] = pd.to_datetime(child["birthdate"]).dt.day
child[["exit", "event"]].describe()
child["sex"].value_counts()
child["socBranch"].value_counts()
print(f"Percent All records children that died: {child['event'].mean():.1%}")
child.head()
#%% Bin birth year into decades for more meaningful aggregation
child["birth_decade"] = (child["birth_year"] // 10) * 10
child.head()

#%% create_person_period_data
def create_person_period_data(df, id_col, time_col, event_col, covariates):
    """
    Transform survival data to person-period format.
    Parameters
    ----------
    df : DataFrame
        Original survival data (one row per subject)
    id_col : str
        Column name for subject identifier
    time_col : str
        Column name for observed time (event or censoring)
    event_col : str
        Column name for event indicator (1=event, 0=censored)
    covariates : list
        Column names for covariates to carry forward

    Returns
    -------
    DataFrame in person-period format
    """
    records = []

    for _, row in df.iterrows():
        duration = int(np.ceil(row[time_col]))

        for t in range(1, duration + 1):
            # Event only in final period if event occurred
            is_final = t == duration
            event_in_period = 1 if (is_final and row[event_col] == 1) else 0

            record = {
                id_col: row[id_col], 
                "period": t, 
                "event": event_in_period
                }

            # Carry forward covariates
            for cov in covariates:
                record[cov] = row[cov]

            records.append(record)

    return pd.DataFrame(records)
# Reuse the general-purpose person-period transformation
#%% transform data
df_long = create_person_period_data(
    child,
    id_col="id",
    time_col="exit",
    event_col="event",
    covariates=["sex", "socBranch", "m.age", "illeg", "birth_decade"],
)
# df_long.to_csv("data/personPeriodChild.csv")
print(f"Person-period rows: {len(df_long):,}")
print(f"Total events: {df_long['event'].sum():,}")

#%% aggregate_person_period(
def aggregate_person_period(df_long, group_cols, event_col="event"):
    """
    Aggregate person-period data into binomial counts per covariate stratum.

    Parameters
    ----------
    df_long : DataFrame
        Person-period data with one row per person-period
    group_cols : list
        Columns defining the strata to aggregate over
    event_col : str
        Name of the binary event column

    Returns
    -------
    DataFrame with 'events' (sum) and 'at_risk' (count) per stratum
    """
    return (
        df_long.groupby(group_cols)
        .agg(events=(event_col, "sum"), at_risk=(event_col, "count"))
        .reset_index()
    )


# Aggregate — using birth_decade instead of birth_year for real compression
df_binomial = aggregate_person_period(
    df_long, 
    group_cols=["period", "sex", "socBranch", "illeg", "birth_decade"]
)
# df_binomial.to_csv("data/binomialDF.csv")
print(f"Aggregated dataset: {len(df_binomial):,} rows")
print(f"compared to {len(df_long):,} person-period rows")


df_binomial["sex"] = pd.Categorical(df_binomial["sex"], categories=["male", "female"], ordered=True)
df_binomial["socBranch"] = pd.Categorical(
    df_binomial["socBranch"], categories=["official", "business", "farming", "worker"], ordered=True
)
df_binomial.head()

#%% Fit binomial model with cloglog link
model_child = bmb.Model(
    "p(events, at_risk) ~ sex + socBranch + period + scale(birth_decade)",
    data=df_binomial,
    family="binomial",
    link="cloglog",
    categorical="period",
)

results_child = model_child.fit(
    chains=4,
    random_seed=2320,
    inference_method="numpyro",
    idata_kwargs={"log_likelihood": True},
)

az.plot_forest(results_child, var_names="period", combined=True, r_hat=True);


#%% derive_survival_curves
def derive_survival_curves(model, results, pred_df, model_type="cloglog"):
    """
    Derive hazard, cumulative hazard, and survival from a fitted Bambi model,
    with uncertainty bands from the posterior distribution.

    Uses model.predict() to obtain posterior draws of the mean parameter
    (hazard probability for cloglog, rate for Poisson) at each time period.
    Survival and cumulative hazard are computed from these draws via
    cumulative product and cumulative sum. Credible intervals are obtained
    directly from quantiles across the posterior draws.

    Parameters
    ----------
    model : bmb.Model
        Fitted Bambi model
    results : az.InferenceData
        MCMC results
    pred_df : DataFrame
        Prediction data with covariate values for each period
    model_type : str, one of 'cloglog' or 'poisson'
        Controls how posterior draws are processed.
        - 'cloglog': draws are hazard probabilities, used directly.
        - 'poisson': draws are rates. We clip at 1 because rates > 1
          are incoherent for per-period survival probabilities.
    """
    predictions = model.predict(results, data=pred_df, inplace=False)

    if model_type == "cloglog":
        var_name = "p"
    else:
        var_name = "mu"

    hazard = predictions.posterior[var_name]
    cumulative_hazard = hazard.cumsum("__obs__")
    survival_p = (1 - hazard).cumprod("__obs__")

    # For Poisson models, clip counts at 1 to recover binary interpretation
    if model_type == "poisson":
        hazard = hazard.clip(0, 1)

    hazard_mean = hazard.mean(("chain", "draw"))
    cumulative_hazard_mean = cumulative_hazard.mean(("chain", "draw"))
    survival_p_mean = survival_p.mean(("chain", "draw"))

    # Credible intervals 94%
    hazards_hdi = hazard.quantile((0.03, 0.97), dim=("chain", "draw"))
    cum_hazards_hdi = cumulative_hazard.quantile((0.03, 0.97), dim=("chain", "draw"))
    survival_hdi = survival_p.quantile((0.03, 0.97), dim=("chain", "draw"))

    return {
        "hazards": hazard_mean,
        "cum_hazards": cumulative_hazard_mean,
        "survival": survival_p_mean,
        "hazards_hdi": hazards_hdi,
        "survival_hdi": survival_hdi,
        "cum_hazards_hdi": cum_hazards_hdi,
    }


def plot_survival_curves(scenarios, title="Survival Analysis", show_hdi=True):
    """
    Plot hazard, cumulative hazard, and survival side-by-side,
    with optional HDI bands.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(scenarios), 2)))

    for i, scenario in enumerate(scenarios):
        periods = np.arange(1, len(scenario["hazards"]) + 1)

        # Hazard
        axes[0].step(
            periods,
            scenario["hazards"],
            where="mid",
            label=scenario["label"],
            color=colors[i],
            linewidth=2,
        )
        if show_hdi and "hazards_hdi" in scenario:
            axes[0].fill_between(
                periods,
                scenario["hazards_hdi"][0],
                scenario["hazards_hdi"][1],
                color=colors[i],
                alpha=0.15,
                step="mid",
            )

        # Cumulative hazard
        axes[1].step(
            periods,
            scenario["cum_hazards"],
            where="mid",
            label=scenario["label"],
            color=colors[i],
            linewidth=2,
        )
        if show_hdi and "cum_hazards_hdi" in scenario:
            axes[1].fill_between(
                periods,
                scenario["cum_hazards_hdi"][0],
                scenario["cum_hazards_hdi"][1],
                color=colors[i],
                alpha=0.15,
                step="mid",
            )

        # Survival (prepend 1.0 at time 0)
        surv_plot = np.insert(scenario["survival"], 0, 1.0)
        time_plot = np.insert(periods, 0, 0)
        axes[2].step(
            time_plot,
            surv_plot,
            where="post",
            label=scenario["label"],
            color=colors[i],
            linewidth=2,
        )
        if show_hdi and "survival_hdi" in scenario:
            surv_lo = np.insert(scenario["survival_hdi"][0], 0, 1.0)
            surv_hi = np.insert(scenario["survival_hdi"][1], 0, 1.0)
            axes[2].fill_between(
                time_plot, surv_lo, surv_hi, color=colors[i], alpha=0.15, step="post"
            )

    axes[0].set_title("Hazard $h(t)$", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("P(Event in Period | Survived)")

    axes[1].set_title("Cumulative Hazard $H(t)$", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Accumulated Risk")

    axes[2].set_title("Survival $S(t)$", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("P(Survival Past Time $t$)")
    axes[2].set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xlabel("Age (Years)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


#%% "Estimated survival curves by father's social class, for males born in the reference decade"
# Define periods matching the child data
t_range = np.arange(1, 16)


# Create prediction scenarios for different social classes
def make_pred_df(socBranch, sex="male", periods=t_range):
    return pd.DataFrame(
        {
            "period": periods,
            "sex": [sex] * len(periods),
            "socBranch": [socBranch] * len(periods),
            "birth_decade": [1864] * len(periods),
            "at_risk": [1]
            * len(
                periods
            ),  #  gives you the per-person rate, which is needed for hazard probability
        }
    )


scenarios = []
for soc_class in ["official", "business", "farming", "worker"]:
    pred_df = make_pred_df(soc_class)
    curves = derive_survival_curves(model_child, results_child, pred_df)
    curves["label"] = soc_class.capitalize()
    scenarios.append(curves)

plot_survival_curves(scenarios, title="Child Mortality by Father's Social Class")


#%% "Estimated survival curves by sex"
scenarios_sex = []
for sex in ["male", "female"]:
    pred_df = make_pred_df("official", sex=sex)
    curves = derive_survival_curves(model_child, results_child, pred_df)
    curves["label"] = sex.capitalize()
    scenarios_sex.append(curves)

plot_survival_curves(scenarios_sex, title="Child Mortality by Sex (Official Class)")


#%% Model with spline baseline hazard
model_spline = bmb.Model(
    "p(events, at_risk) ~ sex + socBranch + bs(period, df=4) + scale(birth_decade)",
    data=df_binomial,
    family="binomial",
    link="cloglog",
)

results_spline = model_spline.fit(
    draws=1000,
    chains=4,
    inference_method="numpyro",
    random_seed=2535,
    idata_kwargs={"log_likelihood": True},
)


az.summary(results_spline, var_names=["bs(period, df=4)"])


#%% "Survival curves from the spline baseline model, compared across social classes"
scenarios_spline = []
for soc_class in ["official", "business", "farming", "worker"]:
    pred_df = make_pred_df(soc_class)
    curves = derive_survival_curves(model_spline, results_spline, pred_df)
    curves["label"] = soc_class.capitalize()
    scenarios_spline.append(curves)

plot_survival_curves(scenarios_spline, title="Child Mortality by Social Class (Spline Baseline)")


model_poisson = bmb.Model(
    "events ~ sex + socBranch + period + scale(birth_decade) + offset(log(at_risk))",
    data=df_binomial,
    family="poisson",
    categorical="period",
)

results_poisson = model_poisson.fit(
    draws=1000,
    chains=4,
    random_seed=42,
    idata_kwargs={"log_likelihood": True},
    inference_method="numpyro",
)


comparison = az.compare({"Cloglog": results_child, "Poisson": results_poisson}, ic="loo")
comparison

#%% "Survival curves derived from the Poisson rate model, compared to the binomial cloglog model"
# Compare Poisson and cloglog survival curves for the baseline profile
pred_baseline = make_pred_df(socBranch="official", sex="male")

curves_clog = derive_survival_curves(model_child, results_child, pred_baseline)
curves_clog["label"] = "Cloglog (Binomial)"

curves_pois = derive_survival_curves(
    model_poisson, results_poisson, pred_baseline, model_type="poisson"
)
curves_pois["label"] = "Poisson"

plot_survival_curves(
    [curves_clog, curves_pois], title="Cloglog vs Poisson: Survival for Official-Class Males"
)