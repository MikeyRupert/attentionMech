# Discrete-Time Survival Analysis

Bayesian discrete-time survival models using [Bambi](https://bambinos.github.io/bambi/) + [NumPyro](https://num.pyro.ai/). Two case studies walk through the same analytical pipeline on different datasets, building intuition for how survival analysis works in practice.

## What's in here

### `survial_model_discrete_time.py` — Child Mortality (19th-century Sweden)

The original case study, drawn from Goran Brostrom's *Event History Analysis with R*. Uses historical parish records (~1850–1890) to model childhood survival to age 15.

- **Event**: child death before age 15
- **Time axis**: age in years (1–15)
- **Covariates**: father's social class, sex, legitimacy, birth decade
- **Data**: `data/child_raw.csv`

### `survival_model_employee_attrition.py` — Employee Attrition (IBM HR)

A new demo using the [IBM HR Employee Attrition](https://github.com/IBM/employee-attrition-aif360) dataset. Same pipeline, more accessible domain, with expanded explanations of every concept.

- **Event**: employee voluntarily quits
- **Time axis**: years at company (1–20)
- **Covariates**: department, gender, overtime
- **Data**: `data/emp_attrition.csv`

## Pipeline (both files follow the same steps)

1. **Load** one-row-per-subject survival data
2. **Person-period expansion** — explode each subject into one row per time period they were at risk
3. **Binomial aggregation** — collapse identical covariate patterns into (events, at_risk) counts for faster MCMC
4. **Cloglog model** — binomial GLM with complementary log-log link (discrete-time proportional hazards)
5. **Derive survival curves** — compute hazard h(t), cumulative hazard H(t), and survival S(t) from posterior draws
6. **Compare subgroups** — plot survival curves by covariate (social class / department / gender / overtime)
7. **Spline baseline** — replace categorical period effects with smooth B-spline basis
8. **Poisson alternative** — fit equivalent model via Poisson regression with offset
9. **Model comparison** — LOO cross-validation to compare cloglog vs Poisson

## Key concepts

| Term | Meaning |
|---|---|
| **Hazard h(t)** | P(event in period t \| survived to t) |
| **Survival S(t)** | P(still alive/employed past time t) = product of (1 - h(k)) for k = 1..t |
| **Cumulative hazard H(t)** | Running sum of h(t) — measures accumulated risk |
| **Censoring** | Subject observed for some time but event never happened — we know they survived *at least* this long |
| **Cloglog link** | log(-log(1 - h)) = Xb — the discrete-time analogue of Cox proportional hazards |

## Setup

```bash
# Requires Python >= 3.11
uv sync
```

Both files are structured as cell-based scripts (`#%%` markers) for use with VS Code's interactive Python / Jupyter execution or any editor that supports cell mode.
