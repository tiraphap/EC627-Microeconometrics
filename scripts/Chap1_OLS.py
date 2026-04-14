# MIT License
# Copyright (c) 2026 Tiraphap Fakthong
# See LICENSE file for full license text.
"""
=============================================================================
EC627 Microeconometrics - Chapter 1
OLS Regression, Specification Analysis, and Prediction
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

Dataset: MEPS (Medical Expenditure Panel Survey)
File: Chap1_data.xlsx

This script replicates the Stata do-file "Chap 1.do" in Python.
Designed to run on Google Colab.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib seaborn openpyxl

OVERVIEW:
---------
This chapter covers the foundations of Ordinary Least Squares (OLS) regression
in the context of health economics. We model medical expenditure as a function
of insurance status, health conditions, and demographic characteristics.

Key topics:
  1. Descriptive statistics and data exploration
  2. OLS estimation with heteroskedasticity-robust standard errors
  3. Hypothesis testing (Wald tests)
  4. Specification testing (RESET, Breusch-Pagan, White)
  5. Prediction from log-linear models (naive, normal retransformation, Duan smearing)
  6. Average Treatment Effect (ATE) estimation via counterfactual prediction
  7. Sampling weights (pweight) and Weighted Least Squares
=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
# We import the main libraries needed throughout the analysis.
# - pandas: data manipulation (DataFrames, similar to Stata's dataset)
# - numpy: numerical operations (arrays, math functions)
# - matplotlib/seaborn: plotting (similar to Stata's graph commands)
# - scipy.stats: statistical distributions and tests
# - statsmodels: regression and econometric models (the Python equivalent of Stata)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import omni_normtest

# For Google Colab: upload your Excel file or mount Google Drive
# from google.colab import files
# uploaded = files.upload()  # Upload Chap1_data.xlsx

# Display settings so that output tables are readable
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
np.set_printoptions(precision=4)

# ============================================================
# 1. LOAD DATA
# ============================================================
# The dataset comes from the Medical Expenditure Panel Survey (MEPS),
# a nationally representative survey of the U.S. civilian non-institutionalized
# population. It contains individual-level data on medical expenditure, insurance
# status, health conditions, and demographics.
#
# Stata equivalent: use "Chap 1 data.dta", clear
# In Python, we read from the Excel file that was converted from Stata's .dta format.
# Try multiple paths (Colab, local scripts/, repo root)
for _path in ['Chap1_data.xlsx', '../data/Chap1_data.xlsx', 'data/Chap1_data.xlsx']:
    if os.path.exists(_path):
        df = pd.read_excel(_path)
        break
else:
    raise FileNotFoundError('Cannot find Chap1_data.xlsx. Upload it or check data/ folder.')


print("=" * 60)
print("CHAPTER 1: OLS REGRESSION & SPECIFICATION ANALYSIS")
print("=" * 60)

# ============================================================
# 2. DATA SUMMARY STATISTICS
# ============================================================
# Before running any regression, it is essential to understand the data.
# We examine the key variables:
#   - totexp:   Total medical expenditure (in dollars) -- the dependent variable in levels
#   - ltotexp:  Log of total medical expenditure -- the dependent variable in logs
#   - posexp:   Indicator = 1 if totexp > 0 (some people have zero expenditure)
#   - suppins:  Supplementary private insurance (1 = has insurance) -- key treatment variable
#   - phylim:   Physical limitation (1 = yes)
#   - actlim:   Activity limitation (1 = yes)
#   - totchr:   Total number of chronic conditions (0, 1, 2, ...)
#   - age:      Age in years
#   - female:   Gender (1 = female)
#   - income:   Household income

# Stata equivalent: describe totexp ltotexp posexp suppins phylim actlim totchr age female income
print("\n--- Variable Descriptions ---")
key_vars = ['totexp', 'ltotexp', 'posexp', 'suppins', 'phylim',
            'actlim', 'totchr', 'age', 'female', 'income']
print(df[key_vars].dtypes)

# Stata equivalent: summarize totexp ltotexp posexp suppins phylim actlim totchr age female income
# The .describe() method produces count, mean, std, min, 25%, 50%, 75%, max
# -- similar to Stata's "summarize" command.
print("\n--- Summary Statistics ---")
print(df[key_vars].describe().T.round(3))

# INTERPRETATION:
# Look at the mean and median of totexp. Medical expenditure is typically
# right-skewed: most people spend little, but a few spend a great deal.
# If mean >> median, this confirms right skewness.
# The log transformation (ltotexp) helps normalize the distribution.

# Stata equivalent: summarize totexp, detail
# "detail" gives additional statistics: skewness, kurtosis, percentiles.
print("\n--- Detailed Summary of totexp ---")
totexp = df['totexp']
print(f"  Mean:     {totexp.mean():.2f}")
print(f"  Median:   {totexp.median():.2f}")
print(f"  Std Dev:  {totexp.std():.2f}")
print(f"  Skewness: {totexp.skew():.3f}")
print(f"  Kurtosis: {totexp.kurtosis():.3f}")
print(f"  Min:      {totexp.min():.2f}")
print(f"  Max:      {totexp.max():.2f}")

# INTERPRETATION:
# Skewness > 0 means the distribution has a long right tail (a few very high spenders).
# High kurtosis (excess > 0) means heavier tails than a normal distribution.
# This motivates using log(totexp) as the dependent variable instead of totexp in levels.

# Stata equivalent: tabstat totexp ltotexp, stat(count mean p50 sd skew kurt) col(stat)
# Compare the distribution of totexp vs ltotexp.
print("\n--- tabstat equivalent ---")
for var in ['totexp', 'ltotexp']:
    s = df[var].dropna()
    print(f"\n  {var}:")
    print(f"    N={len(s)}, Mean={s.mean():.2f}, Median={s.median():.2f}, "
          f"SD={s.std():.2f}, Skew={s.skew():.3f}, Kurt={s.kurtosis():.3f}")

# INTERPRETATION:
# After log transformation, skewness should be much closer to 0 and kurtosis
# closer to 3 (or excess kurtosis close to 0), indicating a distribution
# that is much closer to normal. This justifies using ltotexp in OLS.

# ============================================================
# 3. KERNEL DENSITY PLOTS
# ============================================================
# Kernel density estimation (KDE) is a non-parametric way to visualize the
# shape of a distribution. It's a smoothed version of a histogram.
#
# We plot the density of totexp (levels) and ltotexp (log) side by side.
# This visually confirms that the log transformation produces a more
# symmetric, approximately normal distribution.
#
# Stata equivalent: kdensity totexp / kdensity ltotexp
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# We restrict to positive expenditures (posexp == 1) because log(0) is undefined.
pos = df[df['posexp'] == 1]

# Left panel: density of totexp (in levels) -- expected to be heavily right-skewed
pos['totexp'][pos['totexp'] < 40000].plot.kde(ax=axes[0], linewidth=2)
axes[0].set_title('Kernel Density: Total Expenditure (Levels)')
axes[0].set_xlabel('totexp')

# Right panel: density of ltotexp (in logs) -- expected to be approximately bell-shaped
pos['ltotexp'][pos['ltotexp'] < np.log(40000)].plot.kde(ax=axes[1], linewidth=2, color='orange')
axes[1].set_title('Kernel Density: Log Total Expenditure')
axes[1].set_xlabel('ltotexp')

plt.tight_layout()
plt.savefig('fig1_density.png', dpi=150)
plt.show()
print("Figure saved: fig1_density.png")

# INTERPRETATION:
# Left plot: The density of totexp is heavily right-skewed with a long tail,
#   confirming that OLS on levels would violate the normality assumption.
# Right plot: The density of ltotexp is approximately symmetric and bell-shaped,
#   supporting the use of a log-linear model.

# ============================================================
# 4. CORRELATION MATRIX
# ============================================================
# Pairwise correlations help us understand the linear relationships between variables.
# This is useful for:
#   1. Identifying which regressors are most strongly associated with the outcome
#   2. Detecting potential multicollinearity (very high correlation between regressors)
#
# Stata equivalent: correlate ltotexp suppins phylim actlim totchr age female income
corr_vars = ['ltotexp', 'suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'income']
print("\n--- Pairwise Correlations ---")
print(df[corr_vars].corr().round(3))

# INTERPRETATION:
# - Look at the first column/row: correlations with ltotexp show which variables
#   are most associated with log medical expenditure.
# - totchr (chronic conditions) should have the highest positive correlation with ltotexp,
#   since sicker people spend more on healthcare.
# - Check for multicollinearity: if any two regressors have |r| > 0.8, consider
#   whether both should be in the model. phylim and actlim are related (both measure
#   functional limitation) but capture different aspects.

# ============================================================
# 5. OLS REGRESSION WITH ROBUST STANDARD ERRORS
# ============================================================
# This is the main regression: we estimate the conditional mean of ltotexp
# as a linear function of insurance status, health conditions, and demographics.
#
# Model: ltotexp = beta0 + beta1*suppins + beta2*phylim + beta3*actlim
#                + beta4*totchr + beta5*age + beta6*female + beta7*income + u
#
# We use HC1 (heteroskedasticity-consistent) standard errors, which is the
# default "robust" option in Stata. This ensures valid inference even if the
# error variance is not constant across observations.
#
# Stata equivalent: regress ltotexp suppins phylim actlim totchr age female income, vce(robust)
print("\n" + "=" * 60)
print("OLS REGRESSION: ltotexp on covariates (HC1 robust SEs)")
print("=" * 60)

# Drop any rows with missing values in the regression variables.
# OLS requires complete cases (no NaN values).
reg_vars = ['ltotexp', 'suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'income']
df_reg = df[reg_vars].dropna()

# sm.add_constant() adds a column of ones for the intercept term.
# This is required because statsmodels does not automatically include a constant.
X = sm.add_constant(df_reg[['suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'income']])
y = df_reg['ltotexp']

# OLS with HC1 (Stata's "robust") standard errors
# HC1 applies a small-sample correction: multiply the White variance estimator by n/(n-k).
# This matches Stata's vce(robust) exactly.
model_ols = sm.OLS(y, X).fit(cov_type='HC1')
print(model_ols.summary())

# INTERPRETATION OF OLS RESULTS:
# Each coefficient represents the expected change in log(medical expenditure)
# for a one-unit change in the regressor, holding other variables constant.
#
# Since the dependent variable is in logs, the coefficients can be interpreted
# as approximate percentage changes:
#   - beta*100 ~ percentage change in expenditure for a 1-unit change in X
#   - For dummy variables (suppins, phylim, actlim, female):
#     exp(beta) - 1 ~ exact percentage change
#
# Expected signs:
#   suppins (+): Insurance coverage increases expenditure (moral hazard)
#   phylim  (+): Physical limitations increase expenditure
#   actlim  (+): Activity limitations increase expenditure
#   totchr  (+): More chronic conditions -> more spending
#   age     (+): Older people spend more on healthcare
#   female  (+): Women typically have higher expenditures (pregnancy, preventive care)
#   income  (+): Higher income -> more healthcare consumption
#
# R-squared: Proportion of variance in ltotexp explained by the model.
# Adjusted R-squared: Penalizes for the number of regressors.
# F-statistic: Tests H0: all slope coefficients = 0 simultaneously.

# ============================================================
# 6. WALD TESTS
# ============================================================
# Wald tests allow us to test linear restrictions on the coefficients.
# These are important for testing economic hypotheses.

# TEST 1: Are the effects of physical limitation and activity limitation equal?
# H0: beta_phylim = beta_actlim (they have the same effect on spending)
# H1: beta_phylim != beta_actlim (their effects differ)
#
# Stata equivalent: test phylim = actlim
print("\n--- Wald Test: phylim = actlim ---")
print("H0: beta_phylim = beta_actlim")
wald_test = model_ols.wald_test('phylim = actlim')
wval = wald_test.statistic
print(f"  F-statistic: {(wval[0][0] if hasattr(wval, '__getitem__') else wval):.4f}")
print(f"  p-value:     {wald_test.pvalue:.4f}")

# INTERPRETATION:
# If p-value < 0.05: Reject H0 -> physical and activity limitations have statistically
#   different effects on medical spending.
# If p-value > 0.05: Fail to reject H0 -> no evidence that the two effects differ.

# TEST 2: Are the health variables jointly significant?
# H0: beta_phylim = beta_actlim = beta_totchr = 0 (health doesn't matter)
# H1: At least one health coefficient != 0
#
# This is an exclusion restriction test -- asking whether a group of variables
# can be dropped from the model without loss.
#
# Stata equivalent: test phylim actlim totchr
print("\n--- Joint F-test: phylim = actlim = totchr = 0 ---")
print("H0: beta_phylim = beta_actlim = beta_totchr = 0")
f_test = model_ols.f_test('phylim = 0, actlim = 0, totchr = 0')
fval = f_test.statistic
print(f"  F-statistic: {(fval[0][0] if hasattr(fval, '__getitem__') else fval):.4f}")
print(f"  p-value:     {f_test.pvalue:.4f}")

# INTERPRETATION:
# If p-value < 0.05: Reject H0 -> health variables are jointly significant.
#   This almost certainly will be rejected, confirming that health status
#   is an important determinant of medical spending.

# ============================================================
# 7. MODEL COMPARISON
# ============================================================
# We compare two specifications to see whether income or education better
# explains medical expenditure. This illustrates the concept of model selection.
#
# REG1: includes income as a regressor (our main model above)
# REG2: replaces income with years of education (educyr)
#
# We compare using AIC, BIC, and R-squared. Lower AIC/BIC = better fit (penalized).
#
# Stata equivalent: estimates store REG1 / estimates table REG1 REG2
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

if 'educyr' in df.columns:
    df_reg2 = df[['ltotexp', 'suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'educyr']].dropna()
    X2 = sm.add_constant(df_reg2[['suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'educyr']])
    y2 = df_reg2['ltotexp']
    model_ols2 = sm.OLS(y2, X2).fit(cov_type='HC1')
    print("\nREG2 (with educyr instead of income):")
    print(model_ols2.summary2().tables[1])

    # INTERPRETATION:
    # Compare the R-squared values of REG1 and REG2.
    # The model with a higher R-squared explains more variation in ltotexp.
    # Also compare the AIC/BIC values: the model with a lower value is preferred.
    # Note: These models are not nested, so we cannot use a simple F-test to compare them.
else:
    print("Variable 'educyr' not found in dataset; skipping REG2.")

# ============================================================
# 8. SPECIFICATION TESTS
# ============================================================
# Specification tests check whether the OLS assumptions are satisfied.
# If they are violated, OLS may produce biased or inefficient estimates.
print("\n" + "=" * 60)
print("SPECIFICATION TESTS")
print("=" * 60)

# NOTE: Some specification tests require OLS with default (non-robust) standard errors
# because they use the OLS residuals under the assumption of homoskedasticity.
model_default = sm.OLS(y, X).fit()

# --- Ramsey RESET test ---
# PURPOSE: Tests for functional form misspecification (omitted nonlinear terms).
# The test adds powers of the fitted values (y-hat^2, y-hat^3, y-hat^4) to the
# regression and tests whether they are jointly significant.
#
# H0: The model is correctly specified (no omitted nonlinear terms)
# H1: The model has omitted nonlinear terms
#
# If we reject H0, this suggests we should add squared or interaction terms,
# or use a different functional form (e.g., Box-Cox transformation).
#
# Stata equivalent: estat ovtest
print("\n--- Ramsey RESET Test (Functional Form) ---")
print("H0: No omitted nonlinear terms")
from statsmodels.stats.diagnostic import linear_reset
reset_result = linear_reset(model_default, power=4, use_f=True)
print(f"  F-statistic: {reset_result.fvalue:.4f}")
print(f"  p-value:     {reset_result.pvalue:.4f}")
if reset_result.pvalue < 0.05:
    print("  -> REJECT H0: Evidence of misspecification.")
    print("     The linear functional form may not be adequate.")
    print("     Consider adding polynomial terms, interactions, or using a different model.")
else:
    print("  -> FAIL TO REJECT H0: No evidence of omitted nonlinear terms.")
    print("     The linear specification appears adequate.")

# --- Breusch-Pagan test ---
# PURPOSE: Tests for heteroskedasticity (non-constant error variance).
# The test regresses the squared OLS residuals on the original regressors
# and checks if the regressors explain the variance of the errors.
#
# H0: Var(u|X) = sigma^2 (homoskedasticity -- constant variance)
# H1: Var(u|X) = sigma^2 * h(X) (heteroskedasticity -- variance depends on X)
#
# If we reject H0, we should use robust standard errors (which we already do
# with HC1) or use WLS/FGLS for efficiency.
#
# Stata equivalent: estat hettest, iid
print("\n--- Breusch-Pagan Test (Heteroskedasticity) ---")
print("H0: Homoskedasticity (constant variance)")
bp_stat, bp_pval, _, _ = het_breuschpagan(model_default.resid, X)
print(f"  LM statistic: {bp_stat:.4f}")
print(f"  p-value:      {bp_pval:.4f}")
if bp_pval < 0.05:
    print("  -> REJECT H0: Evidence of heteroskedasticity.")
    print("     OLS is still consistent but inefficient. Use robust SEs for valid inference.")
    print("     Consider WLS/FGLS for efficiency improvement (see Chapter 3).")
else:
    print("  -> FAIL TO REJECT H0: No evidence of heteroskedasticity.")

# --- White test ---
# PURPOSE: A more general test for heteroskedasticity.
# Unlike Breusch-Pagan which assumes a specific functional form for the variance,
# the White test includes all regressors, their squares, and cross-products.
# It also serves as a general specification test.
#
# H0: Homoskedasticity (and correct specification)
# H1: Heteroskedasticity or misspecification
#
# Stata equivalent: estat imtest
print("\n--- White Test (General Specification) ---")
print("H0: Homoskedasticity")
white_stat, white_pval, _, _ = het_white(model_default.resid, X)
print(f"  Test statistic: {white_stat:.4f}")
print(f"  p-value:        {white_pval:.4f}")

# INTERPRETATION:
# If both BP and White tests reject, there is strong evidence of heteroskedasticity.
# This is common in cross-sectional micro data -- individual-level data almost always
# exhibits heteroskedasticity. This is why we use vce(robust) by default.

# ============================================================
# 9. RESIDUAL PLOT
# ============================================================
# A visual check for heteroskedasticity and functional form.
# We plot the OLS residuals against the fitted values.
#
# What to look for:
#   - Random scatter around zero: model is well-specified
#   - Fan shape (wider spread as y-hat increases): heteroskedasticity
#   - Curved pattern: functional form misspecification
#
# Stata equivalent: rvfplot (Residual vs. Fitted Values plot)
fig, ax = plt.subplots(figsize=(8, 5))
yhat = model_default.fittedvalues
resid = model_default.resid
ax.scatter(yhat, resid, alpha=0.3, s=10)
ax.axhline(y=0, color='red', linestyle='--')
ax.set_xlabel('Fitted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residuals vs Fitted Values')
plt.tight_layout()
plt.savefig('fig2_rvfplot.png', dpi=150)
plt.show()
print("Figure saved: fig2_rvfplot.png")

# INTERPRETATION:
# If the plot shows a "fan shape" (residuals spread out as fitted values increase),
# this is visual evidence of heteroskedasticity.
# If the plot shows a curve, the linear model may be misspecified.
# A good model should show residuals randomly scattered around zero with constant spread.

# ============================================================
# 10. PREDICTION
# ============================================================
# When the dependent variable is log(Y), predicting E[Y] is not straightforward.
# Simply taking exp(predicted log Y) gives the WRONG answer because of Jensen's inequality:
#   E[exp(X)] != exp(E[X])   (since exp() is convex)
#
# We compare four prediction methods:
#   1. Levels OLS:       Directly predict Y from a levels regression
#   2. Naive (WRONG):    exp(x'beta) -- ignores the variance correction
#   3. Normal retrans:   exp(x'beta) * exp(sigma^2/2) -- assumes log-normal errors
#   4. Duan smearing:    exp(x'beta) * E[exp(u_hat)] -- non-parametric correction
#
# The Duan smearing estimator is preferred because it does not assume normality of errors.
print("\n" + "=" * 60)
print("PREDICTION FROM LOG-LINEAR MODEL")
print("=" * 60)

# Filter to positive expenditures (log is only defined for positive values)
df_pos = df[df['totexp'] > 0].copy()
reg_vars_pos = ['totexp', 'ltotexp', 'suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'income']
df_pos = df_pos[reg_vars_pos].dropna()

# --- Levels regression (for comparison) ---
# This directly regresses totexp on X. It's simple but may perform poorly
# because totexp is heavily skewed.
# Stata equivalent: regress totexp suppins phylim actlim totchr age female income, vce(robust)
X_pos = sm.add_constant(df_pos[['suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'income']])
model_levels = sm.OLS(df_pos['totexp'], X_pos).fit(cov_type='HC1')
df_pos['yhat_levels'] = model_levels.fittedvalues

# --- Log model ---
# Regress ltotexp = log(totexp) on X. We need the residuals for the smearing estimator.
model_log = sm.OLS(df_pos['ltotexp'], X_pos).fit()
df_pos['lyhat'] = model_log.fittedvalues
df_pos['uhat'] = model_log.resid
sigma2 = model_log.mse_resid  # Mean squared residual = Var(u_hat)

# Method 1: Naive (WRONG -- systematically underestimates E[Y])
# yhat_naive = exp(x'beta)
# This underestimates because E[exp(log Y)] = E[Y], but exp(E[log Y]) < E[Y]
# by Jensen's inequality (exp is convex).
df_pos['yhat_naive'] = np.exp(df_pos['lyhat'])

# Method 2: Normal retransformation (correct IF errors are normal)
# yhat = exp(x'beta) * exp(sigma^2 / 2)
# The correction factor exp(sigma^2/2) is the moment generating function of the
# normal distribution evaluated at t=1. It adjusts for the systematic bias.
df_pos['yhat_normal'] = np.exp(df_pos['lyhat']) * np.exp(0.5 * sigma2)

# Method 3: Duan smearing estimator (correct regardless of error distribution)
# yhat = exp(x'beta) * [1/n * SUM exp(u_hat_i)]
# The smearing factor is the sample average of exp(residuals).
# This is a non-parametric correction that works even when errors are not normal.
smearing_factor = np.exp(df_pos['uhat']).mean()
df_pos['yhat_duan'] = np.exp(df_pos['lyhat']) * smearing_factor

print(f"\n  Smearing factor (Duan): {smearing_factor:.4f}")
print(f"  Normal correction factor: {np.exp(0.5 * sigma2):.4f}")
print(f"  (If errors are truly normal, these two factors should be similar)")

print("\n--- Comparison of Prediction Methods ---")
pred_cols = ['totexp', 'yhat_levels', 'yhat_naive', 'yhat_normal', 'yhat_duan']
print(df_pos[pred_cols].describe().loc[['mean', '50%', 'std']].round(2))

# INTERPRETATION:
# - yhat_naive will underestimate the mean of totexp (always biased downward)
# - yhat_normal and yhat_duan should produce mean predictions closer to the
#   actual mean of totexp
# - The Duan estimator is preferred because it does not assume log-normal errors
# - Compare the predicted means: a good predictor should have a mean close to
#   the actual mean of totexp (but individual predictions may still differ)

# ============================================================
# 11. TREATMENT EFFECT OF SUPPLEMENTARY INSURANCE
# ============================================================
# We estimate the Average Treatment Effect (ATE) of having supplementary insurance
# on medical expenditure using counterfactual prediction.
#
# Method:
#   1. Predict Y for everyone IF they had insurance (suppins=1)
#   2. Predict Y for everyone IF they had no insurance (suppins=0)
#   3. ATE = E[Y(1)] - E[Y(0)] = average difference
#
# This gives us the causal effect of insurance IF we assume the OLS model is
# correctly specified and suppins is exogenous (uncorrelated with the error term).
# The exogeneity assumption may not hold (selection into insurance), so the ATE
# should be interpreted with caution. For a causal approach, see Chapter 4 (IV).
print("\n" + "=" * 60)
print("AVERAGE TREATMENT EFFECT (ATE) OF SUPPLEMENTARY INSURANCE")
print("=" * 60)

# Counterfactual 1: Set suppins = 1 for everyone and predict
X_all1 = X_pos.copy()
X_all1['suppins'] = 1
lyhat1 = model_log.predict(X_all1)
yhat1 = np.exp(lyhat1) * np.exp(0.5 * sigma2)  # Normal retransformation

# Counterfactual 0: Set suppins = 0 for everyone and predict
X_all0 = X_pos.copy()
X_all0['suppins'] = 0
lyhat0 = model_log.predict(X_all0)
yhat0 = np.exp(lyhat0) * np.exp(0.5 * sigma2)

# ATE = average difference in predicted expenditure
ate = (yhat1 - yhat0).mean()
print(f"  E[Y | suppins=1] = {yhat1.mean():.2f}")
print(f"  E[Y | suppins=0] = {yhat0.mean():.2f}")
print(f"  ATE = {ate:.2f}")
print(f"\n  Interpretation: Supplementary insurance is associated with")
print(f"  approximately ${ate:.0f} higher medical expenditure on average.")
print(f"  This could reflect moral hazard (insured people use more care)")
print(f"  or selection (sicker people are more likely to buy insurance).")
print(f"  Causal interpretation requires the exogeneity assumption.")

# ============================================================
# 12. SAMPLING WEIGHTS
# ============================================================
# Survey data often uses complex sampling designs (stratification, clustering,
# oversampling of certain groups). Sampling weights adjust for unequal
# probability of selection so that estimates are representative of the population.
#
# In Stata, [pweight=w] uses probability weights. The weight for each observation
# is proportional to the inverse of its probability of being sampled.
#
# In Python, WLS (Weighted Least Squares) with weights = 1/w replicates Stata's
# pweight behavior. The combination of WLS + HC1 robust SEs gives the same
# results as Stata's regression with pweights.
#
# Here we use an artificial weight (totchr^2 + 0.5) for demonstration purposes.
# In practice, you would use the survey-provided sampling weights.
print("\n" + "=" * 60)
print("SAMPLING WEIGHTS")
print("=" * 60)

# Stata equivalent: generate swght = totchr^2 + 0.5
df_pos['swght'] = df_pos['totchr'] ** 2 + 0.5

# Weighted regression
# Stata equivalent: regress totexp suppins ... [pweight=swght]
# In WLS, weights = 1/variance. For pweights, we set weights = 1/swght.
model_wt = sm.WLS(
    df_pos['totexp'],
    sm.add_constant(df_pos[['suppins', 'phylim', 'actlim', 'totchr', 'age', 'female', 'income']]),
    weights=1.0 / df_pos['swght']
).fit(cov_type='HC1')

print("\nWeighted OLS (pweight=swght):")
print(model_wt.summary2().tables[1].round(4))

print("\nUnweighted OLS:")
print(model_levels.summary2().tables[1].round(4))

# INTERPRETATION:
# Compare the weighted and unweighted coefficient estimates:
# - If they are similar, the unweighted model is adequate.
# - If they differ substantially, the sampling design affects the estimates,
#   and the weighted estimator is needed for population-representative inference.
# - Standard errors also change: weighted regression adjusts for heteroskedasticity
#   induced by the sampling design.
# - In practice, always use the survey-provided weights when they are available.

print("\n" + "=" * 60)
print("END OF CHAPTER 1")
print("=" * 60)
