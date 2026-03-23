# MIT License
# Copyright (c) 2026 Tiraphap Fakthong
# See LICENSE file for full license text.
"""
=============================================================================
EC627 Microeconometrics - Chapter 6
Panel Data: Linear and Nonlinear Models
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

Datasets: Chap6_panel.xlsx (PSID), Chap6_npanel.xlsx (RAND HIE)
This script replicates "Chap 6 panel.do" and "Chap 6 npanel.do" in Python.
Designed to run on Google Colab.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib linearmodels openpyxl

=============================================================================
OVERVIEW OF KEY TOPICS COVERED IN THIS SCRIPT
=============================================================================

This script covers the core econometric methods for panel (longitudinal) data,
which tracks the same individuals (or firms, countries, etc.) over multiple
time periods. Panel data is one of the most powerful data structures in
applied microeconometrics because it allows researchers to control for
unobserved heterogeneity -- characteristics of individuals that are not
measured in the data but may be correlated with the regressors.

PART A: Linear Panel Data Models (using PSID wage data)
-------------------------------------------------------
  1. Panel Summary Statistics (xtsum decomposition)
     - Decomposing total variation into "between" (cross-sectional) and
       "within" (time-series) components.
     - Understanding how much variation exists across individuals vs.
       within individuals over time.

  2. Time-Series Plots of Panel Data
     - Visualizing individual trajectories ("spaghetti plots") to see
       heterogeneity in levels (between variation) and trends (within
       variation).

  3. Pooled OLS Estimator
     - Treats the panel as a single cross-section, ignoring the panel
       structure entirely.
     - Requires cluster-robust standard errors to account for within-
       individual correlation.
     - Biased if unobserved individual effects (alpha_i) are correlated
       with regressors.

  4. Between Estimator (BE)
     - Runs OLS on group means (one observation per individual).
     - Uses only cross-sectional ("between") variation.
     - Captures long-run or permanent relationships.
     - Inefficient and potentially biased (cannot control for alpha_i).

  5. Fixed Effects / Within Estimator (FE)
     - Demeans all variables by subtracting individual-specific means.
     - This "within transformation" eliminates all time-invariant
       unobserved heterogeneity (alpha_i).
     - Key consequence: time-invariant regressors (like education in PSID)
       are also eliminated and their coefficients cannot be estimated.
     - Consistent even when alpha_i is correlated with X.
     - Equivalent to including individual dummy variables (LSDV).

  6. Random Effects Estimator (RE)
     - Assumes alpha_i is UNCORRELATED with the regressors X.
     - Uses a GLS transformation that is a weighted average of within and
       between variation.
     - More efficient than FE when its assumptions hold.
     - Can estimate coefficients on time-invariant variables (e.g., ed).
     - Biased and inconsistent if alpha_i is correlated with X.

  7. First Differences Estimator (FD)
     - Subtracts previous-period values: Delta(y_it) = y_it - y_{i,t-1}.
     - Like FE, eliminates time-invariant alpha_i.
     - Loses one time period per individual.
     - Equivalent to FE when T = 2; generally different for T > 2.
     - More robust to certain types of serial correlation in errors.

  8. Comparison Table of All Estimators
     - Side-by-side display for comparing coefficient magnitudes and signs
       across all estimation approaches.

  9. Hausman Test (FE vs. RE)
     - Tests H0: alpha_i is uncorrelated with X (RE is consistent)
       vs. H1: alpha_i is correlated with X (only FE is consistent).
     - Under H0, both FE and RE are consistent, but RE is efficient.
     - Under H1, only FE is consistent, so FE should be used.
     - Test statistic: H = (b_FE - b_RE)' [V(b_FE) - V(b_RE)]^{-1} (b_FE - b_RE)
       which follows a chi-squared distribution under H0.

PART B: Nonlinear Panel Data Models (using RAND HIE data)
----------------------------------------------------------
  10. Pooled Logit for Binary Outcomes
      - Models the probability of a positive medical expenditure (dmdu = 1).
      - Uses cluster-robust SEs to account for repeated observations.
      - Average Marginal Effects (AME) convert log-odds coefficients
        into probability-scale effects.

  11. Poisson and Negative Binomial for Count Outcomes
      - Models the number of medical utilization events (mdu).
      - Poisson assumes E[y|X] = Var[y|X] (equidispersion).
      - Negative Binomial relaxes this by allowing overdispersion
        (Var > Mean), which is extremely common in health data.
      - Comparison of Poisson vs. NB coefficients and standard errors.

MAPPING BETWEEN STATA AND PYTHON
---------------------------------
  Stata command           | Python equivalent (linearmodels / statsmodels)
  ------------------------+------------------------------------------------
  xtreg y x, fe           | PanelOLS(y, x, entity_effects=True)
  xtreg y x, re           | RandomEffects(y, x)
  xtreg y x, be           | BetweenOLS(y, x)
  regress y x (pooled)    | PooledOLS(y, x) or sm.OLS(y, x)
  regress D.(y x)         | FirstDifferenceOLS(y, x)
  vce(cluster id)         | .fit(cov_type='clustered', cluster_entity=True)
  hausman fe re           | Manual computation (see Section 9 below)
  logit y x               | sm.Logit(y, x)
  poisson y x             | sm.GLM(y, x, family=Poisson())
  nbreg y x               | sm.GLM(y, x, family=NegativeBinomial())

=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
# We import the core scientific Python stack:
#   - pandas: data manipulation (DataFrames, similar to Stata datasets)
#   - numpy: numerical computing (arrays, linear algebra)
#   - matplotlib: plotting (similar to Stata's graph commands)
#   - scipy.stats: statistical distributions (chi-squared for Hausman test)
#   - statsmodels: econometric models (OLS, Logit, GLM for Poisson/NB)
#   - linearmodels.panel: dedicated panel data estimators that closely
#     mirror Stata's xtreg command (FE, RE, BE, FD, Pooled OLS)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
try:
    from linearmodels.panel import (
        PooledOLS, BetweenOLS, FirstDifferenceOLS,
        PanelOLS, RandomEffects
    )
except ImportError:
    raise ImportError(
        "This script requires the 'linearmodels' package.\n"
        "Install with: pip install linearmodels"
    )

# Set display formatting for cleaner output.
# np.set_printoptions controls how numpy arrays are printed (4 decimal places).
# pd.set_option controls how pandas DataFrames are displayed (up to 15 columns).
np.set_printoptions(precision=4)
pd.set_option('display.max_columns', 15)

print("=" * 60)
print("CHAPTER 6: PANEL DATA - LINEAR AND NONLINEAR MODELS")
print("=" * 60)

# ============================================================
# PART A: LINEAR PANEL DATA (PSID)
# ============================================================
# ---------------------------------------------------------------
# The Panel Study of Income Dynamics (PSID) is one of the most
# important panel datasets in labor economics. It has tracked the
# same families in the US since 1968, collecting annual data on
# wages, employment, education, and many other socioeconomic
# variables.
#
# This dataset contains 595 individuals observed over 7 years
# (1976-1982), giving us N=595 and T=7 for a total of N*T = 4,165
# observations (assuming a balanced panel).
#
# Key variables:
#   lwage  - log of hourly wage (the dependent variable)
#   exp    - years of work experience
#   exp2   - experience squared (captures diminishing returns)
#   wks    - weeks worked in the year
#   ed     - years of education (TIME-INVARIANT for most adults)
#   south  - dummy for living in the US South
#   id     - individual identifier (the "entity" dimension)
#   t      - time period (the "time" dimension)
#
# The standard panel wage equation is:
#   lwage_it = beta_0 + beta_1*exp_it + beta_2*exp2_it
#              + beta_3*wks_it + beta_4*ed_i + alpha_i + u_it
#
# where alpha_i is the unobserved individual fixed effect (e.g.,
# innate ability, motivation, family background) and u_it is the
# idiosyncratic error. The key econometric question is whether
# alpha_i is correlated with the regressors X -- this determines
# which estimator is appropriate.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("PART A: LINEAR PANEL DATA (PSID)")
print("=" * 60)

df = pd.read_excel("Chap6_panel.xlsx")
print(f"\nDataset: 595 individuals x 7 years (1976-82)")
print(f"Total observations: {len(df)}")
print(f"\n--- Summary Statistics ---")
print(df[['lwage', 'exp', 'exp2', 'wks', 'ed', 'south']].describe().round(3))

# ---------------------------------------------------------------
# SET PANEL INDEX
# ---------------------------------------------------------------
# Panel data in Python (using the linearmodels library) requires a
# MultiIndex on the DataFrame: the first level is the entity (id)
# and the second level is the time period (t).
#
# This is analogous to Stata's:
#   xtset id t
#
# Once set, linearmodels knows which dimension is the entity
# (individual) and which is time, enabling it to compute within
# transformations, between transformations, etc.
# ---------------------------------------------------------------
df = df.set_index(['id', 't'])

# ============================================================
# 1. PANEL SUMMARY (xtsum equivalent)
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Within vs. Between Variation
# ---------------------------------------------------------------
# One of the most important diagnostics in panel data is decomposing
# the total variation of each variable into two components:
#
#   BETWEEN variation: How much do individuals differ FROM EACH OTHER
#     on average? Computed as the standard deviation of group means
#     (one mean per individual). If every individual has a very
#     different average wage, the between SD is large.
#
#   WITHIN variation: How much does each individual vary OVER TIME
#     around their own mean? Computed by first demeaning each
#     observation (subtracting the individual's mean), then taking
#     the SD of these deviations. If individuals' wages fluctuate
#     a lot year-to-year, the within SD is large.
#
# This decomposition matters because:
#   - The Between estimator uses ONLY between variation.
#   - The Fixed Effects (Within) estimator uses ONLY within variation.
#   - Random Effects uses a weighted combination of both.
#   - Pooled OLS uses both but does not properly weight them.
#
# For example, education (ed) is time-invariant in this dataset,
# so it has ZERO within variation but positive between variation.
# This means FE CANNOT estimate the effect of education (it gets
# swept away in the demeaning), while BE and RE can.
#
# In Stata, this is the output from:
#   xtsum lwage exp wks ed
#
# The code below manually computes the three standard deviations:
#   - overall_std: SD of the raw variable across all N*T obs
#   - between_std: SD of the individual means (across N groups)
#   - within_std:  SD of (x_it - x_bar_i) across all N*T obs
# ---------------------------------------------------------------
print("\n--- Panel Summary: Within and Between Variation ---")
for var in ['lwage', 'exp', 'wks', 'ed']:
    overall_mean = df[var].mean()
    overall_std = df[var].std()
    between_std = df[var].groupby('id').mean().std()
    within_std = (df[var] - df[var].groupby('id').transform('mean')).std()
    print(f"  {var:8s}: overall SD={overall_std:.4f}, between SD={between_std:.4f}, within SD={within_std:.4f}")

# ---------------------------------------------------------------
# INTERPRETATION: What to look for in the xtsum output
# ---------------------------------------------------------------
# - 'lwage': Both between and within variation are substantial,
#   meaning wage differences exist across individuals AND within
#   individuals over time. Both FE and BE have something to work with.
#
# - 'ed' (education): Should have nonzero between SD but near-zero
#   within SD, because education is (essentially) fixed for working
#   adults in this sample. This confirms that FE cannot identify
#   the education coefficient.
#
# - 'exp' (experience): Within SD should be moderate (experience
#   grows by ~1 each year for everyone), while between SD captures
#   the fact that individuals started working at different ages.
#
# - 'wks' (weeks worked): If within SD is large, it means individuals'
#   labor supply varies substantially year-to-year, providing useful
#   within-person variation for FE estimation.
# ---------------------------------------------------------------

# ============================================================
# 2. TIME-SERIES PLOTS
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Visualizing Panel Data
# ---------------------------------------------------------------
# "Spaghetti plots" show one line per individual over time. They
# are valuable for understanding:
#   1. Level heterogeneity: Do lines start at very different levels?
#      (This is the alpha_i -- individual fixed effects.)
#   2. Slope heterogeneity: Do trends differ across individuals?
#   3. General patterns: Is there a common upward/downward trend?
#
# Here we plot the first 20 individuals for readability. Each line
# represents one person's lwage (left panel) or weeks worked (right
# panel) trajectory across the 7 time periods.
#
# If you see lines that are roughly parallel but at very different
# vertical positions, that is strong visual evidence of large
# individual fixed effects (alpha_i), motivating the use of FE or
# RE estimators rather than Pooled OLS.
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_plot = df.reset_index()
for i, uid in enumerate(df_plot['id'].unique()[:20]):
    sub = df_plot[df_plot['id'] == uid]
    axes[0].plot(sub['t'], sub['lwage'], alpha=0.6, linewidth=1)
    axes[1].plot(sub['t'], sub['wks'], alpha=0.6, linewidth=1)
axes[0].set_title('Log Wage by Individual', fontweight='bold')
axes[0].set_xlabel('Time'); axes[0].set_ylabel('lwage')
axes[1].set_title('Weeks Worked by Individual', fontweight='bold')
axes[1].set_xlabel('Time'); axes[1].set_ylabel('wks')
plt.tight_layout()
plt.savefig('fig_ch6_panel_ts.png', dpi=150)
plt.show()

# ============================================================
# 3. POOLED OLS
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Pooled OLS
# ---------------------------------------------------------------
# Pooled OLS simply stacks all N*T observations and runs a standard
# OLS regression, completely ignoring the panel structure. The model:
#
#   lwage_it = beta_0 + beta_1*exp_it + ... + beta_4*ed_i + epsilon_it
#
# where epsilon_it = alpha_i + u_it combines the individual effect
# with the idiosyncratic error.
#
# Problems with Pooled OLS:
#   (a) BIASED coefficients if alpha_i is correlated with any X.
#       For example, if more "able" people (high alpha_i) also get
#       more education, then the ed coefficient picks up both the
#       true return to education AND the ability effect, biasing it
#       upward (omitted variable bias).
#   (b) INEFFICIENT even if alpha_i is uncorrelated with X, because
#       observations from the same individual are correlated (the
#       alpha_i appears in every period's error). Standard OLS SEs
#       are wrong -- they understate true uncertainty.
#
# Solution for (b): Use CLUSTER-ROBUST standard errors, clustering
# on the individual (id). This allows arbitrary within-individual
# correlation of the errors across time periods.
#
# In Stata:  regress lwage exp exp2 wks ed, vce(cluster id)
# In Python: PooledOLS(...).fit(cov_type='clustered', cluster_entity=True)
#
# sm.add_constant() adds a column of 1s for the intercept (beta_0),
# analogous to Stata's automatic inclusion of _cons.
# ---------------------------------------------------------------
print("\n--- Pooled OLS (with cluster-robust SEs) ---")
# Stata: regress lwage exp exp2 wks ed, vce(cluster id)
exog_vars = ['exp', 'exp2', 'wks', 'ed']
y = df['lwage']
X = sm.add_constant(df[exog_vars])

model_pooled = PooledOLS(y, X).fit(cov_type='clustered', cluster_entity=True)
print(model_pooled.summary.tables[1])

# ---------------------------------------------------------------
# INTERPRETATION: Pooled OLS Results
# ---------------------------------------------------------------
# - The coefficient on 'ed' (education) represents the pooled
#   return to an additional year of schooling, combining both cross-
#   sectional and within-person variation. This is likely UPWARD
#   biased due to omitted ability (alpha_i correlated with ed).
#
# - exp should be positive (wages rise with experience) and exp2
#   negative (diminishing returns -- the wage-experience profile
#   is concave).
#
# - Cluster-robust SEs will generally be LARGER than default SEs
#   because they account for intra-individual error correlation.
#   Compare these SEs to what you would get without clustering to
#   see the degree of within-cluster correlation.
# ---------------------------------------------------------------

# ============================================================
# 4. BETWEEN ESTIMATOR
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Between Estimator
# ---------------------------------------------------------------
# The Between Estimator (BE) collapses each individual's data to
# a single observation by computing time-means:
#
#   y_bar_i = beta_0 + beta_1 * x_bar_i + ... + (alpha_i + u_bar_i)
#
# where y_bar_i = (1/T) * sum_t(y_it) is individual i's mean over
# time, and similarly for x_bar_i.
#
# This is equivalent to running OLS on a cross-section of N group
# means. It uses ONLY between-individual variation.
#
# Properties:
#   - CAN estimate effects of time-invariant regressors (like ed),
#     since these vary across individuals.
#   - BIASED if alpha_i is correlated with X (same omitted variable
#     bias as Pooled OLS, but only using cross-sectional variation).
#   - Particularly susceptible to bias from unobserved ability
#     because more able people tend to get more education.
#   - Useful for capturing long-run or permanent relationships.
#   - Less commonly used in practice; mainly a pedagogical tool.
#
# In Stata:  xtreg lwage exp exp2 wks ed, be
# In Python: BetweenOLS(y, X).fit()
# ---------------------------------------------------------------
print("\n--- Between Estimator ---")
# Stata: xtreg lwage exp exp2 wks ed, be
model_be = BetweenOLS(y, X).fit()
print(model_be.summary.tables[1])

# ---------------------------------------------------------------
# INTERPRETATION: Between Estimator Results
# ---------------------------------------------------------------
# - The BE coefficient on education is often the LARGEST among all
#   estimators, because cross-sectional variation in education is
#   heavily confounded with unobserved ability. People with more
#   education earn more partly because they are more able, and this
#   ability premium inflates the between-education coefficient.
#
# - Compare the BE education coefficient to the RE and Pooled OLS
#   coefficients. If BE >> FE (when FE can estimate it) or BE >> RE,
#   this is strong evidence of omitted variable bias in the cross-
#   sectional direction.
#
# - The BE uses only N=595 effective observations (one per person),
#   while other estimators use up to N*T=4165. So BE standard errors
#   will generally be larger.
# ---------------------------------------------------------------

# ============================================================
# 5. FIXED EFFECTS (WITHIN) ESTIMATOR
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Fixed Effects (Within) Estimator
# ---------------------------------------------------------------
# The Fixed Effects (FE) estimator is the workhorse of panel data
# econometrics. It eliminates unobserved individual heterogeneity
# by subtracting individual means from each observation:
#
#   (y_it - y_bar_i) = beta_1*(x_it - x_bar_i) + (u_it - u_bar_i)
#
# This "within transformation" sweeps out the individual fixed
# effect alpha_i (since alpha_i - alpha_i = 0), eliminating all
# time-invariant unobserved heterogeneity.
#
# Key properties and consequences:
#   (a) CONSISTENT even when alpha_i is correlated with X. This is
#       the main advantage of FE over Pooled OLS and RE.
#   (b) CANNOT estimate coefficients on time-invariant regressors.
#       Since ed_i - ed_bar_i = 0 for all i,t, the education
#       variable is perfectly collinear with the individual effects
#       and drops out. This is a real cost of FE estimation.
#   (c) Uses ONLY within-individual variation (changes over time).
#       If a variable has little within variation (e.g., education),
#       FE estimates will be imprecise or impossible.
#   (d) Equivalent to the Least Squares Dummy Variables (LSDV)
#       estimator, which includes N-1 individual dummy variables.
#   (e) For FE to identify causal effects, we need the "strict
#       exogeneity" assumption: E[u_it | X_i1,...,X_iT, alpha_i] = 0,
#       meaning that the idiosyncratic error is uncorrelated with
#       past, present, and future values of X.
#
# In Stata:  xtreg lwage exp exp2 wks ed, fe vce(cluster id)
# In Python: PanelOLS(y, X_time_varying, entity_effects=True)
#            .fit(cov_type='clustered', cluster_entity=True)
#
# IMPORTANT: We exclude 'ed' from the regressor list because it is
# time-invariant and would be automatically dropped (or cause an
# error). The entity_effects=True parameter tells PanelOLS to include
# individual fixed effects (alpha_i) in the model.
# ---------------------------------------------------------------
print("\n--- Fixed Effects (Within) Estimator ---")
# Stata: xtreg lwage exp exp2 wks ed, fe vce(cluster id)
# Note: FE drops time-invariant variables (ed)
model_fe = PanelOLS(y, df[['exp', 'exp2', 'wks']], entity_effects=True).fit(
    cov_type='clustered', cluster_entity=True)
print(model_fe.summary.tables[1])

# FE with ed included (will be absorbed by entity effects)
print("\n  Note: In FE, time-invariant variable 'ed' is absorbed by individual effects")
print(f"  R-squared (within): {model_fe.rsquared_within:.4f}")
print(f"  R-squared (between): {model_fe.rsquared_between:.4f}")
print(f"  R-squared (overall): {model_fe.rsquared_overall:.4f}")

# ---------------------------------------------------------------
# INTERPRETATION: Fixed Effects Results
# ---------------------------------------------------------------
# - The coefficient on 'ed' is MISSING because education does not
#   vary over time for individuals in this sample. This is the
#   fundamental trade-off of FE: you control for all time-invariant
#   confounders but lose the ability to estimate time-invariant
#   effects.
#
# - R-squared decomposition:
#     * R-sq (within):  How well the model explains within-individual
#       variation over time. This is the "natural" R-sq for FE.
#     * R-sq (between): How well predicted group means match actual
#       group means. FE typically has low between R-sq because it
#       does not directly model cross-sectional differences.
#     * R-sq (overall): Weighted combination; often between the
#       within and between R-sq values.
#
# - Compare FE coefficients on exp and wks to Pooled OLS. If they
#   differ substantially, it suggests that unobserved heterogeneity
#   (alpha_i) is correlated with these regressors, biasing the
#   Pooled OLS results.
#
# - The cluster-robust SEs account for any remaining serial
#   correlation in the idiosyncratic errors u_it.
# ---------------------------------------------------------------

# ============================================================
# 6. RANDOM EFFECTS ESTIMATOR
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Random Effects (GLS) Estimator
# ---------------------------------------------------------------
# The Random Effects (RE) estimator assumes that the individual
# effect alpha_i is a random variable UNCORRELATED with all
# regressors X_it. Under this assumption, the composite error
# (alpha_i + u_it) has a specific covariance structure that can be
# exploited for efficient GLS estimation.
#
# The RE transformation is a "quasi-demeaning":
#   (y_it - theta * y_bar_i) = (1-theta)*beta_0
#                              + beta_1*(x_it - theta * x_bar_i) + ...
#
# where theta = 1 - sqrt(sigma_u^2 / (T * sigma_alpha^2 + sigma_u^2))
# is between 0 and 1.
#
# When theta = 0: RE reduces to Pooled OLS (no individual effects).
# When theta = 1: RE reduces to FE (full demeaning).
# In practice, theta is between 0 and 1, so RE is a WEIGHTED AVERAGE
# of the within and between estimators.
#
# Advantages of RE over FE:
#   (a) More EFFICIENT (smaller SEs) when its assumptions hold.
#   (b) CAN estimate coefficients on time-invariant regressors (ed).
#   (c) Uses both within and between variation.
#
# Disadvantage:
#   (a) BIASED and INCONSISTENT if alpha_i is correlated with X.
#       The Hausman test (Section 9) is used to check this.
#
# In Stata:  xtreg lwage exp exp2 wks ed, re vce(cluster id) theta
# In Python: RandomEffects(y, X).fit(cov_type='clustered',
#            cluster_entity=True)
# ---------------------------------------------------------------
print("\n--- Random Effects Estimator ---")
# Stata: xtreg lwage exp exp2 wks ed, re vce(cluster id) theta
model_re = RandomEffects(y, X).fit(cov_type='clustered', cluster_entity=True)
print(model_re.summary.tables[1])

# ---------------------------------------------------------------
# INTERPRETATION: Random Effects Results
# ---------------------------------------------------------------
# - RE produces coefficients for ALL variables, including the
#   time-invariant 'ed'. The education coefficient from RE is a
#   weighted average of the within (FE) and between (BE) estimates.
#
# - If the RE assumptions hold (alpha_i uncorrelated with X), the
#   RE SEs should be smaller than FE SEs because RE is efficient.
#
# - Compare the RE coefficients on exp, exp2, wks to the FE
#   coefficients. If they are very different, the RE assumption
#   likely fails. The Hausman test formalizes this comparison.
#
# - Theta (the quasi-demeaning parameter) tells you how much RE
#   "looks like" FE. Theta near 1 means RE is close to FE (large
#   sigma_alpha relative to sigma_u). Theta near 0 means RE is
#   close to Pooled OLS (small individual effects).
# ---------------------------------------------------------------

# ============================================================
# 7. FIRST DIFFERENCES
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: First Differences (FD) Estimator
# ---------------------------------------------------------------
# First Differencing is an alternative to the within transformation
# for eliminating individual fixed effects. Instead of subtracting
# the individual mean, we subtract the previous period:
#
#   (y_it - y_{i,t-1}) = beta_1*(x_it - x_{i,t-1}) + (u_it - u_{i,t-1})
#   Delta(y_it) = beta_1 * Delta(x_it) + Delta(u_it)
#
# Properties:
#   (a) Like FE, eliminates alpha_i (since alpha_i - alpha_i = 0).
#   (b) Loses one time period per individual (the first period has
#       no "previous period" to difference against), so effective
#       sample size is N*(T-1) instead of N*T.
#   (c) When T = 2, FD and FE produce IDENTICAL estimates.
#   (d) When T > 2, FD and FE generally differ. FD is more efficient
#       when Delta(u_it) is serially uncorrelated (i.e., u_it follows
#       a random walk). FE is more efficient when u_it is serially
#       uncorrelated.
#   (e) Cannot estimate time-invariant regressors (same as FE).
#   (f) Practical advantage: FD requires a weaker exogeneity
#       assumption than FE in some dynamic settings.
#
# In Stata:  regress D.(lwage exp exp2 wks ed), vce(cluster id) noconstant
#            (Stata's D. operator takes first differences)
# In Python: FirstDifferenceOLS(y, X_time_varying).fit(
#            cov_type='clustered', cluster_entity=True)
# ---------------------------------------------------------------
print("\n--- First Differences Estimator ---")
# Stata: regress D.(lwage exp exp2 wks ed), vce(cluster id) noconstant
model_fd = FirstDifferenceOLS(y, df[['exp', 'exp2', 'wks']]).fit(
    cov_type='clustered', cluster_entity=True)
print(model_fd.summary.tables[1])

# ---------------------------------------------------------------
# INTERPRETATION: First Differences Results
# ---------------------------------------------------------------
# - Like FE, the education variable is absent (differenced out).
#
# - Compare FD coefficients to FE coefficients. With T=7 periods,
#   they will generally differ. If they are very different, it
#   suggests potential issues with serial correlation assumptions.
#
# - FD tends to amplify measurement error (differencing can reduce
#   signal-to-noise ratio), so attenuation bias may be worse.
#
# - In practice, researchers often prefer FE over FD for T > 2,
#   unless there are specific reasons to prefer differencing
#   (e.g., unit root concerns, or wanting to require only
#   sequential exogeneity).
# ---------------------------------------------------------------

# ============================================================
# 8. COMPARISON OF ALL ESTIMATORS
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Why Compare Estimators?
# ---------------------------------------------------------------
# Placing all estimator results side-by-side reveals important
# patterns about the data and the role of unobserved heterogeneity:
#
# - If Pooled OLS and FE give very different estimates for the same
#   variable, it means that ignoring individual effects leads to
#   biased inference. The direction of the bias tells you about the
#   correlation between alpha_i and X.
#
# - If BE gives a much larger education coefficient than RE, the
#   between-individual component is driving up the RE estimate,
#   suggesting ability bias.
#
# - If FE and FD are similar, this increases confidence in the
#   within-individual identification strategy.
#
# - RE coefficients should lie between FE and BE (since RE is a
#   weighted average). If they do not, something may be wrong with
#   the estimation or the model specification.
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARISON OF PANEL ESTIMATORS")
print("=" * 60)

comparison = pd.DataFrame({
    'Pooled_OLS': model_pooled.params.round(4),
    'Between': model_be.params.round(4),
})
# Add FE and FD (which don't include ed)
for var in ['exp', 'exp2', 'wks']:
    comparison.loc[var, 'FE'] = round(model_fe.params[var], 4)
    comparison.loc[var, 'FD'] = round(model_fd.params[var], 4)

for var in X.columns:
    comparison.loc[var, 'RE'] = round(model_re.params[var], 4)

print(comparison)

# ---------------------------------------------------------------
# INTERPRETATION: Comparison Table
# ---------------------------------------------------------------
# Look for these patterns:
#
# 1. Education (ed): Only Pooled OLS, BE, and RE can estimate this.
#    - BE typically gives the largest coefficient (most ability bias).
#    - RE will be between FE (not available) and BE.
#    - Pooled OLS is biased upward if alpha_i (ability) correlates
#      positively with education.
#
# 2. Experience (exp, exp2): All estimators can estimate these.
#    - FE and FD use only within variation (year-to-year changes).
#    - Pooled OLS and BE also use cross-sectional variation.
#    - Differences suggest omitted variable bias in Pooled OLS.
#
# 3. Weeks worked (wks): Check if FE vs Pooled differ.
#    - If Pooled OLS overestimates the effect, unobserved motivation
#      (alpha_i) may be driving both more weeks worked and higher
#      wages.
#
# 4. FE vs FD: NaN for 'ed' and 'const' in both columns confirms
#    that time-invariant variables cannot be estimated.
# ---------------------------------------------------------------

# ============================================================
# 9. HAUSMAN TEST (FE vs RE)
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: The Hausman Test
# ---------------------------------------------------------------
# The Hausman (1978) specification test is the standard tool for
# choosing between Fixed Effects and Random Effects. It tests:
#
#   H0: alpha_i is uncorrelated with X (RE is consistent and
#       efficient; both FE and RE are consistent)
#   H1: alpha_i IS correlated with X (only FE is consistent;
#       RE is inconsistent)
#
# Intuition: Under H0, FE and RE should give "similar" coefficient
# estimates (both are consistent). Under H1, they should differ
# systematically (only FE converges to the true values).
#
# The test statistic is:
#   H = (b_FE - b_RE)' * [V(b_FE) - V(b_RE)]^{-1} * (b_FE - b_RE)
#
# Under H0, H ~ chi-squared with K degrees of freedom, where K is
# the number of common coefficients being compared.
#
# IMPORTANT IMPLEMENTATION NOTES:
# - We compare only COMMON variables (those estimated by BOTH FE
#   and RE), namely exp, exp2, wks. We exclude 'ed' because FE
#   cannot estimate it.
# - We use NON-ROBUST variance matrices for the Hausman test.
#   The classic Hausman test requires homoskedastic errors for the
#   difference V(b_FE) - V(b_RE) to be positive semi-definite.
#   Using cluster-robust variances may produce a non-PSD difference,
#   leading to a negative test statistic (nonsensical).
# - If V_FE - V_RE is not positive definite, the test fails. In
#   that case, Wooldridge (2010) suggests a regression-based robust
#   Hausman test alternative.
#
# Decision rule:
#   - If p-value < 0.05: REJECT H0 -> use FE (alpha_i correlated
#     with X, so RE is inconsistent).
#   - If p-value >= 0.05: Fail to reject H0 -> RE is acceptable
#     and preferred for its efficiency.
#
# In Stata:
#   quietly xtreg lwage exp exp2 wks ed, fe
#   estimates store FE
#   quietly xtreg lwage exp exp2 wks ed, re
#   hausman FE
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("HAUSMAN TEST: FE vs RE")
print("=" * 60)

# Manual Hausman test
# Compare coefficients from FE and RE on common variables
common_vars = ['exp', 'exp2', 'wks']
b_fe = np.array([model_fe.params[v] for v in common_vars])
b_re = np.array([model_re.params[v] for v in common_vars])

# Get variance matrices
# We re-estimate FE and RE WITHOUT cluster-robust SEs for the Hausman test.
# This is critical: the classic Hausman test requires the standard
# (non-robust) variance-covariance matrices so that the difference
# V(b_FE) - V(b_RE) is guaranteed to be positive semi-definite under H0.
model_fe_h = PanelOLS(y, df[common_vars], entity_effects=True).fit()
model_re_h = RandomEffects(y, X).fit()

# Extract the variance-covariance submatrices for the common variables.
# .cov gives the full variance-covariance matrix of all estimated parameters.
# We slice it to keep only the rows and columns for common_vars.
V_fe = model_fe_h.cov.loc[common_vars, common_vars].values
V_re = model_re_h.cov.loc[common_vars, common_vars].values

# The Hausman statistic: H = (b_FE - b_RE)' * (V_FE - V_RE)^{-1} * (b_FE - b_RE)
# Under H0, V_FE - V_RE should be positive semi-definite because FE is less
# efficient than RE (FE has larger variance), so V_FE >= V_RE in matrix sense.
diff = b_fe - b_re
V_diff = V_fe - V_re

try:
    hausman_stat = diff @ np.linalg.inv(V_diff) @ diff
    hausman_pval = 1 - stats.chi2.cdf(hausman_stat, len(common_vars))
    print(f"  Hausman statistic: {hausman_stat:.4f}")
    print(f"  Degrees of freedom: {len(common_vars)}")
    print(f"  p-value: {hausman_pval:.4f}")
    if hausman_pval < 0.05:
        print("  -> REJECT H0: Use Fixed Effects (alpha_i correlated with X)")
    else:
        print("  -> Fail to reject H0: Random Effects is consistent and efficient")
except np.linalg.LinAlgError:
    print("  Hausman test: V_FE - V_RE is not positive definite.")
    print("  Use the robust Hausman test (Wooldridge method) instead.")

# ---------------------------------------------------------------
# INTERPRETATION: Hausman Test
# ---------------------------------------------------------------
# - A LARGE Hausman statistic (and small p-value) means FE and RE
#   coefficients differ significantly, implying that alpha_i IS
#   correlated with X. In this case, RE is inconsistent and FE
#   should be used.
#
# - With PSID wage data, the Hausman test typically REJECTS H0,
#   strongly suggesting that unobserved ability (alpha_i) is
#   correlated with experience and education. This makes FE the
#   appropriate estimator for the time-varying regressors.
#
# - If the test produces a negative statistic (possible with robust
#   SEs), the test is inconclusive. This is a well-known limitation.
#   Solutions include: (1) use non-robust variances (as done here),
#   (2) use Wooldridge's regression-based robust Hausman test, or
#   (3) use the Mundlak/Chamberlain approach (add group means of
#   time-varying regressors to the RE model and test their joint
#   significance).
#
# - Practical note: Many applied researchers default to FE regardless
#   of the Hausman test, because the assumption that individual
#   unobserved effects are uncorrelated with ALL regressors is very
#   strong and hard to justify in most economic settings.
# ---------------------------------------------------------------

# ============================================================
# PART B: NONLINEAR PANEL MODELS (RAND HIE)
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Nonlinear Panel Models
# ---------------------------------------------------------------
# In Part A, the dependent variable (lwage) was continuous, and we
# used linear models. In many applied settings, the outcome is:
#   - BINARY (0/1): e.g., did the person use medical services?
#   - COUNT (0, 1, 2, ...): e.g., how many doctor visits?
#
# These require nonlinear models: Logit/Probit for binary, Poisson
# or Negative Binomial for counts.
#
# Panel versions of these models exist (conditional FE logit,
# FE Poisson, RE Poisson, etc.), but they involve complications:
#   - FE logit requires variation in the outcome within each
#     individual (individuals always 0 or always 1 are dropped).
#   - FE Poisson is possible (Hausman, Hall, Griliches 1984) but
#     less commonly implemented.
#   - RE versions require distributional assumptions on alpha_i.
#
# For simplicity and as a practical starting point, this section
# estimates POOLED models (ignoring individual effects) with
# CLUSTER-ROBUST standard errors, which at least corrects the SEs
# for within-individual correlation even if it does not eliminate
# omitted variable bias from alpha_i.
#
# Dataset: The RAND Health Insurance Experiment (HIE), one of the
# most famous randomized experiments in health economics. Families
# were randomly assigned to insurance plans with different cost-
# sharing rates. Key variables:
#   dmdu    - binary: 1 if individual had any medical utilization
#   mdu     - count: number of medical utilization events
#   lcoins  - log of coinsurance rate (the "price" of medical care)
#   ndisease- number of chronic diseases (health status measure)
#   female  - gender dummy (1 = female)
#   age     - age in years
#   lfam    - log of family size
#   child   - dummy: 1 if individual is a child
#   id      - individual identifier for clustering
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("PART B: NONLINEAR PANEL MODELS (RAND HIE)")
print("=" * 60)

df_np = pd.read_excel("Chap6_npanel.xlsx")
print(f"\nDataset: RAND Health Insurance Experiment")
print(f"Total observations: {len(df_np)}")

np_vars = ['dmdu', 'med', 'mdu', 'lcoins', 'ndisease', 'female', 'age', 'lfam', 'child']
available = [v for v in np_vars if v in df_np.columns]
print(f"\n--- Summary Statistics ---")
print(df_np[available].describe().round(3))

# ============================================================
# 10. BINARY OUTCOME: LOGIT MODELS
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Pooled Logit for Binary Outcomes
# ---------------------------------------------------------------
# When the outcome variable is binary (dmdu = 0 or 1), we use a
# logistic regression model:
#
#   Pr(dmdu_it = 1 | X_it) = Lambda(X_it * beta)
#                           = exp(X_it * beta) / (1 + exp(X_it * beta))
#
# where Lambda() is the logistic CDF.
#
# The reported coefficients are in LOG-ODDS scale, which are not
# directly interpretable as marginal effects on the probability.
# For example, a coefficient of 0.5 on 'female' means that being
# female increases the log-odds of medical utilization by 0.5,
# which is NOT a 0.5 increase in probability.
#
# To get interpretable effects, we compute AVERAGE MARGINAL EFFECTS
# (AME): for each observation, compute the marginal effect of each
# variable on Pr(y=1), then average across all observations.
# AME = (1/N) * sum_i [ Lambda'(X_i * beta) * beta_k ]
#
# This gives us: "On average, a one-unit increase in X_k changes
# the probability of the outcome by AME_k percentage points."
#
# We use cluster-robust SEs to account for the fact that individuals
# appear in multiple time periods. Without clustering, SEs would be
# too small (overstating statistical significance).
#
# In Stata:  xtlogit dmdu lcoins ndisease female age lfam child, pa vce(robust)
#   or:      logit dmdu ..., vce(cluster id)
#            margins, dydx(*)
# In Python: sm.Logit(y, X).fit(cov_type='cluster', cov_kwds={'groups': id})
#            model.get_margeff()  -> computes AME
# ---------------------------------------------------------------
if 'dmdu' in df_np.columns:
    print("\n--- Binary Outcome: Pooled Logit ---")
    xvars = [v for v in ['lcoins', 'ndisease', 'female', 'age', 'lfam', 'child']
             if v in df_np.columns]
    df_logit = df_np[['dmdu'] + xvars + ['id']].dropna()

    X_logit = sm.add_constant(df_logit[xvars])
    y_logit = df_logit['dmdu']

    # Pooled logit with cluster-robust SEs
    model_logit = sm.Logit(y_logit, X_logit).fit(
        cov_type='cluster', cov_kwds={'groups': df_logit['id']}, disp=0)
    print(model_logit.summary2().tables[1].round(4))

    # ---------------------------------------------------------------
    # INTERPRETATION: Logit Coefficients
    # ---------------------------------------------------------------
    # - Coefficients are in LOG-ODDS units. A positive coefficient means
    #   the variable INCREASES the probability of medical utilization.
    # - lcoins (log coinsurance): Expected NEGATIVE sign. Higher cost-
    #   sharing (larger coinsurance rate) should reduce the probability
    #   of seeking medical care (moral hazard / price effect).
    # - ndisease: Expected POSITIVE sign. More chronic diseases should
    #   increase the probability of any medical use.
    # - female: Often positive -- women tend to use more medical services.
    # - age: Expected positive (older people use more care).
    # - child: Direction depends on the context and age distribution.
    # ---------------------------------------------------------------

    # Marginal effects
    print("\n--- Average Marginal Effects (Logit) ---")
    mfx = model_logit.get_margeff()
    print(mfx.summary())

    # ---------------------------------------------------------------
    # INTERPRETATION: Average Marginal Effects
    # ---------------------------------------------------------------
    # - AMEs are now in PROBABILITY units, which are directly
    #   interpretable. For example, if AME of 'female' is 0.05, it
    #   means that being female increases the probability of any
    #   medical utilization by 5 percentage points, on average.
    #
    # - lcoins AME: A negative value tells you the PERCENTAGE POINT
    #   decrease in probability for a one-unit increase in log
    #   coinsurance. This is the key policy variable in the RAND HIE.
    #
    # - AMEs sum up the full nonlinear relationship at every observed
    #   data point and average them, so they account for the fact that
    #   marginal effects differ depending on where you are on the
    #   logistic curve (effects are largest when Pr is near 0.5 and
    #   smallest near 0 or 1).
    #
    # - Compare AMEs to OLS (linear probability model) coefficients.
    #   They should be similar in magnitude, confirming that the logit
    #   specification is reasonable.
    # ---------------------------------------------------------------

# ============================================================
# 11. COUNT OUTCOME: POISSON AND NEGATIVE BINOMIAL
# ============================================================
# ---------------------------------------------------------------
# CONCEPTUAL EXPLANATION: Count Data Models
# ---------------------------------------------------------------
# When the dependent variable is a non-negative integer (count),
# OLS is inappropriate because:
#   (a) OLS can predict negative values (counts must be >= 0).
#   (b) Count data is often highly right-skewed with excess zeros.
#   (c) The variance typically increases with the mean (heteroskedasticity).
#
# POISSON REGRESSION:
#   Assumes y_it | X_it ~ Poisson(lambda_it)
#   where log(lambda_it) = X_it * beta  (log link function)
#   and E[y|X] = Var[y|X] = lambda (equidispersion assumption).
#
#   Coefficients are semi-elasticities: a one-unit increase in X_k
#   changes E[y|X] by approximately (beta_k * 100)%.
#   Exactly: E[y|X] is multiplied by exp(beta_k).
#
# NEGATIVE BINOMIAL (NB) REGRESSION:
#   Relaxes the equidispersion assumption by introducing an extra
#   parameter alpha (overdispersion parameter):
#     Var[y|X] = E[y|X] + alpha * E[y|X]^2  (NB2 parameterization)
#   or Var[y|X] = (1 + alpha) * E[y|X]       (NB1 parameterization)
#
#   When alpha = 0, NB reduces to Poisson. When alpha > 0, there is
#   overdispersion (variance exceeds the mean), which is extremely
#   common in real data (especially health care utilization data).
#
# Overdispersion diagnostic: Compare Mean(y) and Var(y). If Var >> Mean,
# the Poisson equidispersion assumption fails, and NB is preferred.
# The ratio Var/Mean is a quick check (should be ~1 under Poisson).
#
# In Stata:
#   xtpoisson mdu ..., pa vce(robust)
#   xtnbreg mdu ..., pa
# In Python:
#   sm.GLM(y, X, family=Poisson()).fit(cov_type='cluster', ...)
#   sm.GLM(y, X, family=NegativeBinomial(alpha=1.0)).fit(...)
#
# NOTE: statsmodels' NegativeBinomial family requires specifying the
# alpha parameter. In Stata, alpha is estimated from the data. Here
# alpha=1.0 is a starting value; for more precise estimation, use
# sm.NegativeBinomial (discrete model class) which estimates alpha.
# ---------------------------------------------------------------
if 'mdu' in df_np.columns:
    print("\n--- Count Outcome: Pooled Poisson ---")
    xvars_count = [v for v in ['lcoins', 'ndisease', 'female', 'age', 'lfam', 'child']
                   if v in df_np.columns]
    df_count = df_np[['mdu'] + xvars_count + ['id']].dropna()

    # ---------------------------------------------------------------
    # OVERDISPERSION CHECK
    # ---------------------------------------------------------------
    # Before estimating, compare Mean and Variance of the count variable.
    # Under Poisson: Var[y] should equal Mean[y] (ratio = 1).
    # If the ratio is much larger than 1, overdispersion is present
    # and Negative Binomial is more appropriate.
    #
    # Health care utilization data is almost always overdispersed
    # because of unobserved heterogeneity in health status and
    # preferences -- some people rarely visit doctors, others visit
    # very frequently, creating excess variance.
    # ---------------------------------------------------------------
    mean_mdu = df_count['mdu'].mean()
    var_mdu = df_count['mdu'].var()
    print(f"  Mean(mdu) = {mean_mdu:.3f}, Var(mdu) = {var_mdu:.3f}, Ratio = {var_mdu/mean_mdu:.3f}")

    # ---------------------------------------------------------------
    # INTERPRETATION: Overdispersion Check
    # ---------------------------------------------------------------
    # - If Ratio >> 1 (e.g., 5, 10, or more), the data is severely
    #   overdispersed. Poisson standard errors will be too small
    #   (overconfident), though the coefficient estimates themselves
    #   remain consistent if the conditional mean is correctly specified
    #   (this is the "Poisson pseudo-MLE" property of Gourieroux,
    #   Monfort, and Trognon 1984). Cluster-robust SEs help fix the
    #   SE problem even under overdispersion.
    # - If Ratio is near 1, Poisson is appropriate and more efficient.
    # ---------------------------------------------------------------

    X_count = sm.add_constant(df_count[xvars_count])
    y_count = df_count['mdu']

    # Poisson with cluster-robust SEs
    model_pois = sm.GLM(y_count, X_count, family=sm.families.Poisson()).fit(
        cov_type='cluster', cov_kwds={'groups': df_count['id']})
    print(model_pois.summary2().tables[1].round(4))

    # ---------------------------------------------------------------
    # INTERPRETATION: Poisson Coefficients
    # ---------------------------------------------------------------
    # - Coefficients are LOG-LINEAR (semi-elasticities). For a
    #   continuous regressor, beta_k means a one-unit increase in X_k
    #   changes E[y|X] by approximately 100*beta_k percent.
    #   Exactly: E[y|X] is multiplied by exp(beta_k).
    #
    # - lcoins: Expected negative. Higher coinsurance rate (price)
    #   reduces the expected number of medical utilization events.
    #   If beta = -0.15, a one-unit increase in log coinsurance
    #   reduces expected utilization by about 15%.
    #
    # - ndisease: Expected positive. Each additional chronic disease
    #   increases expected medical utilization.
    #
    # - Cluster-robust SEs ensure valid inference even if the Poisson
    #   variance assumption fails (which it almost certainly does).
    # ---------------------------------------------------------------

    # Negative Binomial
    print("\n--- Count Outcome: Negative Binomial ---")
    model_nb = sm.GLM(y_count, X_count, family=sm.families.NegativeBinomial(alpha=1.0)).fit(
        cov_type='cluster', cov_kwds={'groups': df_count['id']})
    print(model_nb.summary2().tables[1].round(4))

    # ---------------------------------------------------------------
    # INTERPRETATION: Negative Binomial Coefficients
    # ---------------------------------------------------------------
    # - NB coefficients have the same log-linear interpretation as
    #   Poisson (semi-elasticities), since both use a log link.
    #
    # - The key difference is that NB explicitly models overdispersion.
    #   The alpha parameter captures the degree of extra-Poisson
    #   variation. When alpha > 0, the NB model has "fatter tails"
    #   than Poisson, better fitting the observed distribution.
    #
    # - NB standard errors are typically LARGER than Poisson SEs
    #   (when using default SEs), reflecting the additional uncertainty
    #   from overdispersion. However, with cluster-robust SEs, the
    #   difference may be smaller because robust SEs already account
    #   for some misspecification.
    #
    # - If Poisson and NB give very different coefficient estimates,
    #   it may indicate that the conditional mean specification
    #   (the log-linear model for lambda) is misspecified, not just
    #   the variance. When only the variance is wrong but the mean
    #   is correct, Poisson coefficients are still consistent.
    # ---------------------------------------------------------------

    # Comparison
    print("\n--- Poisson vs Negative Binomial ---")
    comp_count = pd.DataFrame({
        'Poisson': model_pois.params.round(4),
        'NegBin': model_nb.params.round(4),
    })
    print(comp_count)

    # ---------------------------------------------------------------
    # INTERPRETATION: Poisson vs. Negative Binomial Comparison
    # ---------------------------------------------------------------
    # - If the coefficients are very similar between Poisson and NB,
    #   this is reassuring: it suggests the conditional mean function
    #   E[y|X] = exp(X*beta) is correctly specified, and the choice
    #   between Poisson and NB primarily affects the standard errors
    #   and the modeling of dispersion.
    #
    # - If the coefficients differ substantially, investigate whether
    #   the functional form for the conditional mean is correct.
    #   Consider adding interaction terms, nonlinear transformations,
    #   or checking for influential observations (count data models
    #   can be sensitive to large counts).
    #
    # - In practice, with cluster-robust SEs, Poisson regression is
    #   often preferred as a "quasi-MLE" approach because it only
    #   requires correct specification of the conditional mean (not
    #   the full distribution). This is the recommendation of
    #   Wooldridge (2010) and Cameron & Trivedi (2005).
    # ---------------------------------------------------------------

print("\n" + "=" * 60)
print("END OF CHAPTER 6")
print("=" * 60)
