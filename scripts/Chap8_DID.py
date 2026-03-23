# MIT License
# Copyright (c) 2026 Tiraphap Fakthong
# See LICENSE file for full license text.
"""
=============================================================================
EC627 Microeconometrics - Chapter 8
Difference-in-Differences
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

Dataset: Chap8_autor.xlsx (Autor, 2003, JOLE)
This script replicates "Chap 8 DID_2024.do" in Python.
Designed to run on Google Colab.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib linearmodels openpyxl

=============================================================================
OVERVIEW OF KEY TOPICS COVERED IN THIS SCRIPT
=============================================================================

This script implements a Difference-in-Differences (DID) analysis replicating
the seminal paper by David Autor (2003), "Outsourcing at Will: The Contribution
of Unjust Dismissal Doctrine to the Growth of Employment Outsourcing," published
in the Journal of Labor Economics (JOLE).

Topics and methods covered:

1. DIFFERENCE-IN-DIFFERENCES (DID) AS A CAUSAL INFERENCE METHOD
   - DID compares the change in outcomes over time between a treated group and
     a control group. The key insight is that by differencing twice (across
     groups AND across time), we remove both time-invariant group differences
     and common time shocks, isolating the causal effect of the treatment.
   - The fundamental equation is:
         DID = (Y_treated,post - Y_treated,pre) - (Y_control,post - Y_control,pre)
   - In a regression framework with panel data, this becomes a two-way fixed
     effects model: Y_it = alpha_i + gamma_t + delta * D_it + epsilon_it
     where alpha_i are unit (state) fixed effects, gamma_t are time (year)
     fixed effects, and D_it is the treatment indicator.

2. THE PARALLEL TRENDS ASSUMPTION
   - DID is only valid if, in the absence of treatment, the treated and control
     groups would have followed parallel trends over time. This is fundamentally
     untestable (we never observe the counterfactual), but we can check whether
     pre-treatment trends were parallel as suggestive evidence.
   - If the assumption fails, the DID estimate captures both the treatment effect
     AND differential trends, leading to biased estimates.

3. THE AUTOR (2003) APPLICATION
   - Context: U.S. states adopted "employment-at-will" exceptions at different
     times between the 1970s and 1990s. These exceptions made it harder for
     employers to fire workers at will, increasing the cost of direct employment.
   - Hypothesis: If firing costs increase, firms substitute toward temporary
     help services (THS) to maintain labor flexibility.
   - Outcome variable: Log of state-level THS employment (lstateths).
   - Treatment: Staggered adoption of exceptions across states over time.
   - Three types of exceptions (treatment variables):
       (a) Implied Contract (mico): Courts rule that employer handbooks or verbal
           promises create an implied contract, limiting the right to fire.
           This is the MOST COMMON and STRONGEST exception.
       (b) Public Policy (mppa): Courts rule that employees cannot be fired for
           reasons that violate public policy (e.g., refusing to commit a crime,
           filing a workers' comp claim, whistleblowing).
       (c) Good Faith (mgfa): Courts rule that employers must act in good faith
           and cannot fire to avoid paying earned benefits. This is the RAREST
           and most controversial exception.

4. TWO-WAY FIXED EFFECTS (TWFE) REGRESSION
   - State fixed effects (alpha_i) absorb all time-invariant state
     characteristics: geography, culture, industrial composition, etc.
   - Year fixed effects (gamma_t) absorb all nationwide shocks common to all
     states: recessions, federal policy changes, national THS trends, etc.
   - The treatment coefficient (delta) is identified from WITHIN-state variation
     in the timing of exception adoption, compared to other states.

5. STATE-SPECIFIC TIME TRENDS
   - Adding state-specific linear time trends (state_i * t) allows each state
     to have its own trajectory over time. This tests whether results are robust
     to differential pre-existing trends across states.
   - If the DID coefficient is similar with and without state trends, we gain
     confidence that the parallel trends assumption holds.
   - If the coefficient changes dramatically, it suggests confounding trends
     were biasing the baseline estimate.

6. CLUSTERED STANDARD ERRORS
   - Following Bertrand, Duflo, and Mullainathan (2004, QJE), "How Much Should
     We Trust Differences-in-Differences Estimates?", standard errors are
     clustered at the state level.
   - Clustering accounts for: (a) serial correlation within states over time,
     and (b) heteroskedasticity across states.
   - Without clustering, standard errors are typically too small, leading to
     over-rejection of the null hypothesis (false positives).

7. INTERPRETING LOG-LINEAR DID COEFFICIENTS
   - Since the dependent variable is log(THS employment), the DID coefficient
     approximates a percentage change: a coefficient of 0.10 means the exception
     increased THS employment by roughly 10%.
   - More precisely, the percentage change is (exp(beta) - 1) * 100, but for
     small beta, beta * 100 is a good approximation.

8. DESCRIPTIVE ANALYSIS
   - Partial regression is used to remove the effect of total employment on THS
     employment, isolating relative THS growth net of state economic conditions.
   - This produces a "residualized" measure of THS growth that is plotted
     alongside the number of states adopting exceptions over time.
=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
# --------------------------------------------------------------------------
# We import the standard scientific Python stack:
#   - pandas: for data manipulation (DataFrames, similar to Stata's dataset)
#   - numpy: for numerical operations (arrays, math functions)
#   - matplotlib: for plotting (similar to Stata's -graph- commands)
#   - scipy.stats: for statistical distributions and tests
#   - statsmodels: for econometric models (OLS, clustered SEs, etc.)
#     * statsmodels.api (sm): provides OLS, add_constant, etc.
#     * statsmodels.formula.api (smf): provides R-style formula interface
#
# np.set_printoptions(precision=4) limits decimal places in numpy output.
# pd.set_option('display.max_columns', 15) ensures pandas displays enough
# columns without truncation when printing DataFrames.
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

np.set_printoptions(precision=4)
pd.set_option('display.max_columns', 15)

print("=" * 60)
print("CHAPTER 8: DIFFERENCE-IN-DIFFERENCES")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
# --------------------------------------------------------------------------
# DATA LOADING AND PREPARATION
# --------------------------------------------------------------------------
# We load the Autor (2003) dataset from an Excel file. This dataset is a
# state-year panel covering all 50 U.S. states plus D.C. over multiple years.
#
# Key variables in the dataset:
#   - state:     Numeric state identifier (FIPS code or similar)
#   - year:      Year (coded as two digits, e.g., 79 = 1979)
#   - lstateths: Log of state-level temporary help services (THS) employment.
#                This is the DEPENDENT VARIABLE in all regressions.
#   - lnemp:     Log of total state employment. Used as a control to partial
#                out overall state economic conditions in the descriptive section.
#   - mico:      Indicator (0/1) = 1 if the state has adopted an implied
#                contract exception by that year. This is a "treatment" dummy
#                that turns on in the year of adoption and stays on.
#   - mppa:      Indicator (0/1) = 1 if the state has adopted a public policy
#                exception by that year.
#   - mgfa:      Indicator (0/1) = 1 if the state has adopted a good faith
#                exception by that year.
#
# The staggered adoption design means different states adopt these exceptions
# at different times, providing variation for identification. This is sometimes
# called a "generalized DID" or "staggered DID" design.
#
# We sort by state and year to ensure proper panel ordering, then restrict the
# sample to 1979-1995, matching Autor's analysis period.
# --------------------------------------------------------------------------
df = pd.read_excel("Chap8_autor.xlsx")
print(f"\nDataset: Autor (2003, JOLE) - Employment-at-Will Exceptions")
print(f"Total observations: {len(df)}")

# Check available variables
needed_vars = ['state', 'year', 'lstateths', 'lnemp', 'mico', 'mppa', 'mgfa']
avail = [v for v in needed_vars if v in df.columns]
print(f"Available variables: {avail}")

# Sort and restrict sample
df = df.sort_values(['state', 'year'])
df = df[(df['year'] >= 79) & (df['year'] <= 95)].copy()
print(f"Sample after restricting to 1979-1995: {len(df)}")

print(f"\n--- Summary Statistics ---")
print(df[['lstateths', 'lnemp', 'mico', 'mppa', 'mgfa']].describe().round(3))

# --------------------------------------------------------------------------
# INTERPRETATION OF SUMMARY STATISTICS:
# --------------------------------------------------------------------------
# Look at the mean of the treatment indicators (mico, mppa, mgfa). Since these
# are 0/1 dummies, the mean tells you the fraction of state-year observations
# that are "treated" (i.e., the exception has been adopted). For example, if
# mean(mico) = 0.40, it means 40% of state-year observations are in states
# that have adopted the implied contract exception by that year.
#
# lstateths: The mean and standard deviation of log THS employment tell you
# about the distribution of THS across states and years. The variation is what
# allows us to estimate the DID regression.
#
# lnemp: Log total employment varies across states and over time; it will be
# used to partial out overall economic conditions in the descriptive analysis.
# --------------------------------------------------------------------------

# ============================================================
# PART 1: DESCRIPTIVE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("PART 1: DESCRIPTIVE ANALYSIS")
print("=" * 60)

# --------------------------------------------------------------------------
# DESCRIPTIVE ANALYSIS: PARTIAL REGRESSION AND RELATIVE GROWTH
# --------------------------------------------------------------------------
# The goal of this section is to create Figure 1 from Autor (2003): a plot
# showing two series over time:
#   (1) The relative growth of THS employment (net of overall employment)
#   (2) The number of states that have adopted any employment-at-will exception
#
# Why partial regression?
# -----------------------
# Raw THS employment is affected by overall state employment levels. A state
# with a booming economy will have high THS employment simply because ALL
# employment is high. We want to isolate the THS-SPECIFIC component of growth.
#
# Partial regression (also known as the Frisch-Waugh-Lovell theorem) works as
# follows: to see the relationship between Y and X1 after controlling for X2,
# you can regress Y on X2, take the residuals, and those residuals represent
# the variation in Y that is NOT explained by X2.
#
# Here, we regress lstateths (log THS employment) on lnemp (log total
# employment). The residuals (ehat) represent THS employment AFTER removing
# the component explained by overall state employment -- i.e., THS employment
# relative to what we would expect given the state's total employment level.
#
# Step 1.3: Run the partial regression and save residuals.
# The sm.add_constant() function adds a column of ones (the intercept term).
# model_partial.resid gives us the OLS residuals (ehat_i = Y_i - Yhat_i).
# --------------------------------------------------------------------------

# Step 1.3: Regress lstateths on lnemp, keep residuals
model_partial = sm.OLS(
    df['lstateths'],
    sm.add_constant(df['lnemp'])
).fit()
df['ehat'] = model_partial.resid

# --------------------------------------------------------------------------
# Step 1.4: Create relative growth (relative to 1979)
# --------------------------------------------------------------------------
# We want each state's THS residual expressed relative to its OWN 1979 value.
# This creates a "relative growth" measure: how much has this state's THS
# employment (net of total employment) grown since 1979?
#
# Procedure:
#   1. For each state, extract the residual from the FIRST year (1979).
#      df.groupby('state')['ehat'].first() gets the first observation per state
#      after sorting by year.
#   2. Merge this base-year residual back onto the full dataset as variable 'a'.
#   3. Subtract the base-year residual: lrelativeg = ehat - a.
#      This makes lrelativeg = 0 for all states in 1979, and measures cumulative
#      growth since 1979.
#   4. Drop the temporary variables 'a' and 'ehat' (no longer needed).
#
# IMPORTANT: This is in log units, so a value of 0.50 means approximately 50%
# more THS employment (relative to total employment) than in 1979.
# --------------------------------------------------------------------------
df = df.sort_values(['state', 'year'])
first_year_resid = df.groupby('state')['ehat'].first().reset_index()
first_year_resid.columns = ['state', 'a']
df = df.merge(first_year_resid, on='state')
df['lrelativeg'] = df['ehat'] - df['a']
df = df.drop(columns=['a', 'ehat'])

# --------------------------------------------------------------------------
# Step 1.5: Create "any exception" indicator
# --------------------------------------------------------------------------
# This creates a binary variable 'any' that equals 1 if the state has adopted
# AT LEAST ONE of the three exceptions (implied contract, public policy, or
# good faith) by that year. This is used for the descriptive figure only.
#
# The condition (sum > 0) & (sum < inf) ensures we flag states with at least
# one exception while guarding against any missing/infinite values.
# The .astype(int) converts the boolean True/False to 1/0.
# --------------------------------------------------------------------------
df['any'] = ((df['mico'] + df['mppa'] + df['mgfa'] > 0) &
             (df['mico'] + df['mppa'] + df['mgfa'] < np.inf)).astype(int)

# --------------------------------------------------------------------------
# Step 1.6: Collapse to year-level averages
# --------------------------------------------------------------------------
# We aggregate the state-level data to year-level means and counts:
#   - lrelativeg: mean across states of the relative THS growth measure.
#     This gives the average relative growth of THS employment across all states.
#   - any_count: sum of 'any' across states = number of states with at least
#     one exception recognized in that year.
#
# year_full converts the two-digit year to a four-digit year for plotting.
#
# In pandas, .agg() applies different aggregation functions to different columns.
# This is analogous to Stata's -collapse- command.
# --------------------------------------------------------------------------
year_data = df.groupby('year').agg(
    lrelativeg=('lrelativeg', 'mean'),
    any_count=('any', 'sum')
).reset_index()
year_data['year_full'] = year_data['year'] + 1900

# --------------------------------------------------------------------------
# Step 1.7: Two-axis plot (Figure 1 from Autor, 2003)
# --------------------------------------------------------------------------
# This figure is crucial for motivating the DID analysis. It shows:
#   LEFT AXIS (blue diamonds): Average relative growth of THS employment across
#       states over time. A rising line means THS is growing faster than overall
#       employment, consistent with the outsourcing hypothesis.
#   RIGHT AXIS (orange triangles): Number of states that have adopted at least
#       one employment-at-will exception. A rising step-like pattern reflects
#       the staggered adoption of exceptions across states.
#
# What to look for:
#   - Do THS employment and exception adoption move together over time?
#   - The figure should show that as more states adopt exceptions (right axis
#     rises), THS employment grows faster (left axis rises).
#   - This is SUGGESTIVE but NOT CAUSAL evidence. The regression analysis in
#     Part 2 provides the formal causal test.
#
# Technical notes on matplotlib:
#   - fig, ax1 = plt.subplots() creates a figure and primary axis.
#   - ax2 = ax1.twinx() creates a second y-axis sharing the same x-axis.
#   - 'D-' means diamond markers connected by lines; '^-' means triangle markers.
# --------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.plot(year_data['year_full'], year_data['lrelativeg'], 'D-',
         color='#4472C4', linewidth=2, markersize=6, label='Log relative growth of THS employment')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Log THS Industry Size Relative to 1979', fontsize=11, color='#4472C4')
ax1.axhline(y=0, color='black', linewidth=0.5)

ax2.plot(year_data['year_full'], year_data['any_count'], '^-',
         color='#ED7D31', linewidth=2, markersize=6, label='States recognizing an exception')
ax2.set_ylabel('Number of States Recognizing An Exception', fontsize=11, color='#ED7D31')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.title('Relative THS Growth vs Employment-at-Will Exceptions', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('fig_ch8_did_descriptive.png', dpi=150)
plt.show()
print("Figure saved: fig_ch8_did_descriptive.png")

# --------------------------------------------------------------------------
# INTERPRETATION OF THE DESCRIPTIVE FIGURE:
# --------------------------------------------------------------------------
# If you see both series rising together, this is consistent with the hypothesis
# that employment-at-will exceptions drove THS growth. However, correlation does
# not imply causation -- many other factors were also changing over this period
# (e.g., globalization, deregulation, IT revolution). The formal DID regression
# in Part 2 is needed to control for confounders via fixed effects.
# --------------------------------------------------------------------------

# ============================================================
# PART 2: DID REGRESSION (Table 3 Replication)
# ============================================================
print("\n" + "=" * 60)
print("PART 2: DID REGRESSION (Table 3 Replication)")
print("=" * 60)

# --------------------------------------------------------------------------
# DID REGRESSION SETUP
# --------------------------------------------------------------------------
# We now estimate the formal DID regression model. The general specification is:
#
#   lstateths_it = alpha_i + gamma_t + delta * Exception_it + epsilon_it
#
# where:
#   lstateths_it = log THS employment in state i, year t (dependent variable)
#   alpha_i      = state fixed effects (absorbed by state dummies)
#   gamma_t      = year fixed effects (absorbed by year dummies)
#   Exception_it = treatment dummy (mico, mppa, or mgfa), equals 1 if state i
#                  has adopted the exception by year t
#   delta        = the DID coefficient of interest: the causal effect of adopting
#                  the exception on log THS employment
#   epsilon_it   = error term, clustered at the state level
#
# The augmented specification adds state-specific linear time trends:
#
#   lstateths_it = alpha_i + gamma_t + (alpha_i * t) + delta * Exception_it + epsilon_it
#
# where (alpha_i * t) is a set of interactions between state dummies and a
# linear time trend. This allows each state to follow its own linear trajectory,
# absorbing differential pre-existing trends.
#
# IMPORTANT IDENTIFICATION POINT:
# The coefficient delta is identified from DEVIATIONS of the treatment indicator
# from the state's own time path and from common year effects. In the staggered
# design, states that adopt exceptions earlier serve as "treated" while states
# that adopt later (or never) serve as "controls."
#
# DATA RELOAD:
# We reload the original data to start fresh, avoiding any transformations from
# the descriptive analysis (like residualizing or sample restrictions).
# --------------------------------------------------------------------------

# Reload original data for regression
df2 = pd.read_excel("Chap8_autor.xlsx")
df2 = df2.sort_values(['state', 'year'])

# --------------------------------------------------------------------------
# Create time trend variable 't':
# t = year - 78, so t=1 in 1979, t=2 in 1980, ..., t=17 in 1995.
# This is used to construct state-specific linear time trends (state_i * t).
# Subtracting 78 (rather than using the raw year) normalizes the trend to
# start near zero, which improves numerical stability but does not affect
# the regression results.
# --------------------------------------------------------------------------
df2['t'] = df2['year'] - 78

# --------------------------------------------------------------------------
# Create dummy variable matrices for fixed effects:
#
# STATE DUMMIES (state_dummies):
#   pd.get_dummies(df2['state'], prefix='state', drop_first=True) creates a
#   binary indicator for each state EXCEPT the first (which is the reference
#   category). If there are 51 state codes, this produces 50 dummies.
#   These dummies absorb all time-invariant state characteristics.
#   In Stata, this is equivalent to: i.state
#
# YEAR DUMMIES (year_dummies):
#   Similarly, year dummies absorb all state-invariant time effects.
#   In Stata: i.year
#
# drop_first=True: One category is dropped to avoid perfect multicollinearity
#   with the intercept (the "dummy variable trap"). The dropped category
#   becomes the reference group.
#
# dtype=float: Ensures the dummies are floating-point numbers (required by
#   statsmodels for matrix operations).
# --------------------------------------------------------------------------
state_dummies = pd.get_dummies(df2['state'], prefix='state', drop_first=True, dtype=float)
year_dummies = pd.get_dummies(df2['year'], prefix='year', drop_first=True, dtype=float)

# --------------------------------------------------------------------------
# Create state-specific time trends:
#
# These are interactions between each state dummy and the linear time trend t.
# For state j: state_j_trend = (state == j) * t
# This variable equals 0 for all states except j, and equals t (1, 2, 3, ...)
# for state j.
#
# Including these in the regression allows each state to have its own linear
# trajectory in log THS employment. The DID coefficient is then identified from
# deviations AROUND each state's own trend, which is a much stronger test.
#
# We drop the first state's trend (state_trends.iloc[:, 1:]) for identification,
# paralleling the drop_first=True for state dummies.
#
# In Stata, this would be: i.state#c.t
#
# WHY THIS MATTERS FOR IDENTIFICATION:
# Without state trends, a state that was already experiencing THS growth before
# adopting an exception would falsely attribute that pre-existing growth to the
# exception. State-specific trends absorb this confound.
# --------------------------------------------------------------------------
state_dummies_trend = pd.get_dummies(df2['state'], prefix='st', dtype=float)
state_trends = state_dummies_trend.multiply(df2['t'].values, axis=0)
state_trends.columns = [col + '_t' for col in state_trends.columns]
state_trends = state_trends.iloc[:, 1:]  # drop first for identification

# --------------------------------------------------------------------------
# Construct the two base design matrices:
#
# base_X:       State FE + Year FE (no state trends)
# base_X_trend: State FE + Year FE + State-specific linear time trends
#
# The treatment variables (mico, mppa, mgfa) will be added inside the
# run_did() function, so these matrices contain only the fixed effects and
# trends.
# --------------------------------------------------------------------------
y = df2['lstateths']
base_X = pd.concat([state_dummies, year_dummies], axis=1)
base_X_trend = pd.concat([state_dummies, year_dummies, state_trends], axis=1)

results = {}

# --------------------------------------------------------------------------
# FUNCTION: run_did()
# --------------------------------------------------------------------------
# This function estimates a single DID specification with:
#   - Two-way fixed effects (state + year dummies in X_base)
#   - Optionally, state-specific time trends (if X_base = base_X_trend)
#   - Cluster-robust standard errors at the state level
#
# Parameters:
#   y:              Dependent variable (Series): log THS employment
#   treatment_vars: List of treatment variable names (e.g., ['mico'] or
#                   ['mico', 'mppa', 'mgfa'])
#   X_base:         Design matrix with fixed effects (and optionally trends)
#   cluster_var:    Variable to cluster on (Series): state identifier
#   label:          String label for the specification (for printing)
#
# How cluster-robust standard errors work:
#   The cov_type='cluster' option in statsmodels computes standard errors
#   that are robust to arbitrary within-cluster correlation and heteroskedasticity.
#   The 'groups' key specifies the cluster variable (state).
#
#   In Stata, this is equivalent to: reg y x, vce(cluster state)
#
#   Bertrand, Duflo, and Mullainathan (2004) showed that ignoring serial
#   correlation in DID settings leads to massive over-rejection. For example,
#   a 5% test may reject 45% of the time! Clustering at the state level is
#   the standard correction.
#
# Returns: A dictionary with coefficients, standard errors, p-values, R-squared,
#          and number of observations for the treatment variables.
# --------------------------------------------------------------------------
def run_did(y, treatment_vars, X_base, cluster_var, label):
    """Run DID regression with state and year FE, clustered SEs"""
    X = X_base.copy()
    for tv in treatment_vars:
        X[tv] = df2[tv].values
    X = sm.add_constant(X)

    # Drop rows with NaN
    mask = y.notna() & X.notna().all(axis=1)
    y_clean = y[mask]
    X_clean = X[mask]
    cluster_clean = cluster_var[mask]

    model = sm.OLS(y_clean, X_clean).fit(
        cov_type='cluster', cov_kwds={'groups': cluster_clean})

    result = {}
    for tv in treatment_vars:
        result[tv] = {
            'coef': model.params[tv],
            'se': model.bse[tv],
            'pval': model.pvalues[tv]
        }
    result['r2'] = model.rsquared
    result['n'] = model.nobs
    return result

cluster = df2['state']

# --------------------------------------------------------------------------
# THE 8 SPECIFICATIONS (Table 3 from Autor, 2003)
# --------------------------------------------------------------------------
# Table 3 in Autor (2003) presents 8 columns, systematically varying:
#   (A) Which exception is included (mico, mppa, mgfa, or all three together)
#   (B) Whether state-specific linear time trends are included
#
# The 8 specifications are:
#
#   Col 1: mico only,               State+Year FE, NO trends
#   Col 2: mico only,               State+Year FE, WITH state trends
#   Col 3: mppa only,               State+Year FE, NO trends
#   Col 4: mppa only,               State+Year FE, WITH state trends
#   Col 5: mgfa only,               State+Year FE, NO trends
#   Col 6: mgfa only,               State+Year FE, WITH state trends
#   Col 7: all three together,      State+Year FE, NO trends
#   Col 8: all three together,      State+Year FE, WITH state trends
#
# The even-numbered columns (2, 4, 6, 8) include state-specific time trends.
# Comparing odd vs. even columns tells us whether the result is ROBUST to
# controlling for differential state trajectories.
#
# HOW TO READ THE TABLE:
#   - Each cell reports a coefficient and its clustered standard error.
#   - Stars (*, **, ***) denote statistical significance at 10%, 5%, 1%.
#   - A positive coefficient means the exception INCREASED THS employment.
#   - A negative coefficient means the exception DECREASED THS employment.
#   - The magnitude is approximately a percentage change (since the dependent
#     variable is in logs). E.g., coef = 0.10 means ~10% increase.
#
# KEY COMPARISON:
#   - If a coefficient is significant in the odd column but not in the even
#     column (or changes sign), the result is NOT ROBUST to state trends.
#     This suggests the baseline estimate was contaminated by pre-existing
#     differential trends, casting doubt on the causal interpretation.
#   - If a coefficient is stable across both columns, we gain confidence
#     in the causal interpretation, because the estimate survives a test
#     that absorbs differential state-level trends.
# --------------------------------------------------------------------------

# Column 1: mico only, no state trends
print("\n--- Running 8 specifications (Table 3) ---")
specs = [
    ('Col 1', ['mico'], base_X, 'State+Year FE'),
    ('Col 2', ['mico'], base_X_trend, 'State+Year FE + State trends'),
    ('Col 3', ['mppa'], base_X, 'State+Year FE'),
    ('Col 4', ['mppa'], base_X_trend, 'State+Year FE + State trends'),
    ('Col 5', ['mgfa'], base_X, 'State+Year FE'),
    ('Col 6', ['mgfa'], base_X_trend, 'State+Year FE + State trends'),
    ('Col 7', ['mico', 'mppa', 'mgfa'], base_X, 'State+Year FE'),
    ('Col 8', ['mico', 'mppa', 'mgfa'], base_X_trend, 'State+Year FE + State trends'),
]

all_results = {}
for label, tvars, xbase, desc in specs:
    try:
        res = run_did(y, tvars, xbase, cluster, label)
        all_results[label] = res
        print(f"  {label} ({desc}):")
        for tv in tvars:
            star = '***' if res[tv]['pval'] < 0.01 else '**' if res[tv]['pval'] < 0.05 else '*' if res[tv]['pval'] < 0.10 else ''
            print(f"    {tv}: {res[tv]['coef']:.4f} ({res[tv]['se']:.4f}){star}")
        print(f"    R2={res['r2']:.4f}, N={res['n']:.0f}")
    except Exception as e:
        print(f"  {label}: Error - {e}")

# --------------------------------------------------------------------------
# INTERPRETATION OF INDIVIDUAL SPECIFICATION RESULTS:
# --------------------------------------------------------------------------
# As each specification runs, examine:
#
# 1. SIGN of the coefficient: Positive means the exception increased THS
#    employment; negative means it decreased it.
#
# 2. MAGNITUDE: Since lstateths is in logs, the coefficient approximates a
#    percentage change. A coefficient of 0.15 means roughly 15% higher THS
#    employment after adoption.
#
# 3. STATISTICAL SIGNIFICANCE: The p-value tests H0: delta = 0 (no effect).
#    Stars indicate: * p<0.10, ** p<0.05, *** p<0.01.
#    Remember these are CLUSTERED standard errors, so they are conservative
#    compared to non-clustered SEs.
#
# 4. ROBUSTNESS (odd vs. even columns): The critical comparison. If adding
#    state trends dramatically changes the coefficient, the parallel trends
#    assumption is suspect for that exception.
#
# 5. JOINT SPECIFICATION (Cols 7-8): When all three exceptions are included
#    simultaneously, each coefficient is estimated conditional on the others.
#    This tests whether each exception has an INDEPENDENT effect.
# --------------------------------------------------------------------------

# ============================================================
# 3. RESULTS TABLE
# ============================================================
# --------------------------------------------------------------------------
# FORMATTED SUMMARY TABLE
# --------------------------------------------------------------------------
# This section prints a formatted table combining all 8 specifications,
# making it easy to compare coefficients across columns (as in the published
# paper's Table 3).
#
# Each row corresponds to one treatment variable (mico, mppa, mgfa).
# Each column corresponds to one specification (Col 1 through Col 8).
# Cells show the coefficient with significance stars.
# "--" indicates that the variable was not included in that specification.
#
# The bottom row indicates whether state-specific time trends were included.
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY TABLE: DID ESTIMATES")
print("=" * 60)

print(f"\n  {'':8s} {'Col1':>8s} {'Col2':>8s} {'Col3':>8s} {'Col4':>8s} "
      f"{'Col5':>8s} {'Col6':>8s} {'Col7':>8s} {'Col8':>8s}")
print(f"  {'-'*72}")

for var in ['mico', 'mppa', 'mgfa']:
    row = f"  {var:8s}"
    for col_label in ['Col 1', 'Col 2', 'Col 3', 'Col 4', 'Col 5', 'Col 6', 'Col 7', 'Col 8']:
        if col_label in all_results and var in all_results[col_label]:
            coef = all_results[col_label][var]['coef']
            pval = all_results[col_label][var]['pval']
            star = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
            row += f" {coef:>7.3f}{star}"
        else:
            row += f" {'--':>8s}"
    print(row)

print(f"\n  {'Trends':8s}", end="")
for i in range(8):
    print(f" {'Yes' if i % 2 == 1 else 'No':>8s}", end="")
print()

# --------------------------------------------------------------------------
# HOW TO READ THE SUMMARY TABLE:
# --------------------------------------------------------------------------
# 1. Focus first on the mico row (implied contract exception):
#    - Compare Col 1 (no trends) vs Col 2 (with trends).
#    - If both are positive, significant, and similar in magnitude, the implied
#      contract exception has a ROBUST positive effect on THS employment.
#    - This is Autor's main finding.
#
# 2. Then examine mppa (public policy) and mgfa (good faith):
#    - Compare Col 3 vs Col 4 for mppa, Col 5 vs Col 6 for mgfa.
#    - If the coefficient changes sign, loses significance, or changes magnitude
#      substantially, the result is NOT ROBUST.
#    - This means pre-existing state trends were confounding the baseline estimate.
#
# 3. Finally, look at the joint specification (Cols 7-8):
#    - When all three are included simultaneously, does mico remain significant?
#    - Do mppa and mgfa change?
#    - This tests for omitted variable bias from excluding related exceptions.
#
# 4. The "Trends" row at the bottom reminds you which columns include
#    state-specific trends (even columns = Yes, odd columns = No).
# --------------------------------------------------------------------------

# ============================================================
# 4. INTERPRETATION
# ============================================================
# --------------------------------------------------------------------------
# FINAL INTERPRETATION AND ECONOMIC SIGNIFICANCE
# --------------------------------------------------------------------------
# This section summarizes the key substantive findings from the replication
# of Autor (2003), tying the statistical results back to economic theory
# and policy implications.
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. INTERPRETATION OF DID RESULTS")
print("=" * 60)

print("""
Key findings from Autor (2003):

1. IMPLIED CONTRACT (mico):
   - Positive and statistically significant in BOTH specifications
   - Robust to inclusion of state-specific time trends
   - Interpretation: Adoption of implied contract exceptions INCREASES
     temporary help services (THS) employment
   - Mechanism: Workers seek temp agency employment for protection
     against wrongful discharge

2. PUBLIC POLICY (mppa) and GOOD FAITH (mgfa):
   - Results change substantially when state trends are included
   - NOT robust -- suggests confounding state-level trends
   - The parallel trends assumption may not hold for these exceptions

3. WHY ROBUSTNESS TO STATE TRENDS MATTERS:
   - DID relies on the parallel trends assumption
   - If states that adopt exceptions were already trending differently,
     the DID estimate is biased
   - Including state x time trends allows for differential trends
   - If the estimate survives, the causal interpretation is stronger
""")

# --------------------------------------------------------------------------
# ADDITIONAL INTERPRETATION NOTES FOR STUDENTS:
# --------------------------------------------------------------------------
# WHY DOES MICO (IMPLIED CONTRACT) HAVE A ROBUST EFFECT?
#   The implied contract exception is the broadest and most consequential for
#   employers. When courts recognize implied contracts (e.g., from employee
#   handbooks promising job security), firing becomes costly because employers
#   must honor those implicit promises. This directly increases the relative
#   cost of permanent workers vs. temporary workers, making temp agencies an
#   attractive alternative. The economic mechanism is clear, and the DID
#   estimate is robust because states that adopted mico were not on differential
#   THS growth paths prior to adoption.
#
# WHY ARE MPPA AND MGFA NOT ROBUST?
#   Public policy exceptions (mppa) protect workers from being fired for
#   specific reasons (e.g., refusing to commit a crime). This is narrower than
#   implied contract and may not significantly increase overall firing costs.
#   Good faith exceptions (mgfa) are rare and weakly enforced. For both, the
#   lack of robustness to state trends suggests that states adopting these
#   exceptions were systematically different in ways that also affected THS
#   growth, violating the parallel trends assumption.
#
# WHAT DOES IT MEAN WHEN A COEFFICIENT CHANGES SIGN OR SIGNIFICANCE WITH TRENDS?
#   If a coefficient is positive without trends but becomes negative or
#   insignificant with trends, it means the initial positive finding was driven
#   by pre-existing differential trends -- states that adopted the exception
#   were ALREADY experiencing THS growth before adoption. The "effect" was
#   actually a continuation of a pre-existing trend, not a causal consequence
#   of the policy change. This is a textbook example of why testing robustness
#   to trends is essential in any DID analysis.
#
# THE BROADER LESSON FOR DID PRACTITIONERS:
#   Always test the parallel trends assumption. Methods include:
#     (a) Including group-specific time trends (as done here)
#     (b) Event study / leads-and-lags analysis (checking for pre-trends)
#     (c) Placebo tests (using fake treatment dates)
#     (d) Visual inspection of pre-treatment trends
#   If results are not robust, the causal interpretation is weakened, and you
#   should be transparent about this limitation in your research.
# --------------------------------------------------------------------------

print("\n" + "=" * 60)
print("END OF CHAPTER 8")
print("=" * 60)
