"""
=============================================================================
EE627 Microeconometrics - Chapter 4
Instrumental Variables Estimation
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

Dataset: Chap4_data.xlsx
This script replicates "Chap 4.do" in Python.
Designed to run on Google Colab.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib linearmodels openpyxl

=============================================================================
OVERVIEW OF KEY TOPICS
=============================================================================

This script provides a comprehensive walkthrough of Instrumental Variables
(IV) estimation, one of the most important techniques in applied
microeconometrics for addressing endogeneity. The script covers:

1.  DATA LOADING AND PREPARATION
    - Medical Expenditure Panel Survey (MEPS) data on drug expenditures
    - Variable definitions: dependent, endogenous, exogenous, and instruments

2.  OLS ESTIMATION (Baseline for Comparison)
    - Standard OLS regression of log drug expenditure on health insurance
      and controls
    - Why OLS is INCONSISTENT when a regressor is endogenous

3.  2SLS ESTIMATION -- JUST-IDENTIFIED MODEL
    - IV estimation with exactly one instrument for one endogenous regressor
    - The model is "just-identified" (number of instruments = number of
      endogenous regressors), so we have exactly enough information to
      identify the parameters but cannot test instrument validity

4.  FIRST-STAGE REGRESSION
    - Regressing the endogenous variable on the instrument(s) and exogenous
      controls
    - First-stage F-statistic as a diagnostic for instrument strength
    - Stock-Yogo (2005) rule of thumb: F > 10

5.  2SLS ESTIMATION -- OVER-IDENTIFIED MODEL
    - IV estimation with MORE instruments than endogenous regressors
    - The "extra" instruments provide testable overidentifying restrictions

6.  GMM ESTIMATION
    - Generalized Method of Moments: efficient under heteroskedasticity
    - Why GMM is preferred over 2SLS when errors are heteroskedastic

7.  LIML ESTIMATION
    - Limited Information Maximum Likelihood: less biased with weak
      instruments (fewer observations, many instruments, or low F-stat)

8.  COMPARISON OF ESTIMATORS
    - Side-by-side coefficient comparison across OLS, 2SLS, GMM, and LIML

9.  DURBIN-WU-HAUSMAN (DWH) ENDOGENEITY TEST
    - Tests H0: "the regressor is exogenous" (OLS is consistent)
    - Manual implementation via the "control function" / augmented
      regression approach

10. HANSEN J OVERIDENTIFICATION TEST
    - Tests H0: "all instruments are valid" (satisfy the exclusion
      restriction)
    - Only feasible when the model is over-identified

11. WEAK INSTRUMENT DIAGNOSTICS
    - Correlations between instruments and the endogenous regressor
    - Individual first-stage partial F-statistics
    - Sensitivity of IV estimates to instrument choice

=============================================================================
ECONOMETRIC BACKGROUND
=============================================================================

THE ENDOGENEITY PROBLEM
-----------------------
Consider the structural equation:

    y = X1*beta1 + X2*beta2 + u

where X1 is endogenous [Cov(X1, u) != 0] and X2 is exogenous
[Cov(X2, u) = 0]. Endogeneity arises from:

  (a) Omitted variable bias: a variable correlated with both X1 and y is
      left out of the regression, and its effect is absorbed into u.
  (b) Simultaneity: X1 and y are jointly determined (e.g., supply and
      demand), so y causes X1 at the same time X1 causes y.
  (c) Measurement error: X1 is measured with error, creating correlation
      between the observed X1 and u.

In our application, the endogenous regressor is `hi_empunion` (whether the
individual has employer- or union-provided health insurance). This is
endogenous because unobserved factors (e.g., health consciousness, risk
aversion, job characteristics) simultaneously affect insurance status AND
drug expenditure. OLS will produce BIASED and INCONSISTENT estimates.

THE IV SOLUTION
---------------
Instrumental Variables estimation uses "instruments" Z that satisfy:

  1. RELEVANCE:           Cov(Z, X1) != 0
     The instruments must be correlated with the endogenous regressor.
     (Testable via the first-stage F-statistic.)

  2. EXCLUSION RESTRICTION (Exogeneity): Cov(Z, u) = 0
     The instruments must affect y ONLY through X1, not directly.
     (NOT directly testable in a just-identified model; partially testable
     via the Hansen J test in an over-identified model.)

Our instruments are:
  - ssiratio:   Ratio of SSI (Supplemental Security Income) to income
  - lowincome:  Indicator for low income
  - multlc:     Multiple locations (firm characteristic)
  - firmsz:     Firm size

These are characteristics of the individual's economic/employment situation
that plausibly affect whether they have employer-provided insurance (relevance)
but do not directly affect drug expenditure (exclusion restriction).

MAPPING TO STATA
----------------
This script replicates the Stata do-file "Chap 4.do". Key correspondences:

  Stata:   ivregress 2sls y (X1 = Z) X2, vce(robust)
  Python:  IV2SLS(y, exog=X2, endog=X1, instruments=Z).fit(cov_type='robust')

  Stata:   ivregress gmm y (X1 = Z) X2, wmatrix(robust)
  Python:  IVGMM(y, exog=X2, endog=X1, instruments=Z).fit(cov_type='robust')

  Stata:   ivregress liml y (X1 = Z) X2, vce(robust)
  Python:  IVLIML(y, exog=X2, endog=X1, instruments=Z).fit(cov_type='robust')

Note on `linearmodels` terminology:
  - `exog`: exogenous regressors INCLUDING the constant (these appear in both
    the structural equation and the first stage)
  - `endog`: endogenous regressors (instrumented variables)
  - `instruments`: EXCLUDED instruments (variables that appear ONLY in the
    first stage, not in the structural equation)

=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
# We import the core scientific Python stack:
#   - pandas: data manipulation (DataFrames, like Stata's datasets)
#   - numpy: numerical arrays and linear algebra
#   - matplotlib: plotting (not heavily used here, but available)
#   - scipy.stats: statistical distributions for p-value computation
#   - statsmodels: OLS and other standard econometric models
#   - linearmodels: IV estimators (2SLS, GMM, LIML) -- this package
#     was specifically designed to replicate Stata's ivregress functionality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
try:
    from linearmodels.iv import IV2SLS, IVGMM, IVLIML
except ImportError:
    raise ImportError(
        "This script requires the 'linearmodels' package.\n"
        "Install with: pip install linearmodels"
    )

np.set_printoptions(precision=4)

print("=" * 60)
print("CHAPTER 4: INSTRUMENTAL VARIABLES ESTIMATION")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Loads the MEPS (Medical Expenditure Panel Survey) dataset and defines the
# key variable groups used throughout the analysis. This mirrors the Stata
# do-file's `global x2list totchr age female blhisp linc` command.
#
# VARIABLE DEFINITIONS:
# ---------------------
#   Dependent variable:
#     ldrugexp    = log of drug expenditure (continuous, in dollars)
#
#   Endogenous regressor (the "suspect" variable):
#     hi_empunion = 1 if individual has employer- or union-provided health
#                   insurance, 0 otherwise.
#                   This is ENDOGENOUS because unobserved factors (health
#                   consciousness, risk preferences, job type) affect both
#                   insurance status and drug spending.
#
#   Exogenous controls (X2 -- included instruments):
#     totchr      = total number of chronic conditions (health status)
#     age         = age of the individual
#     female      = 1 if female, 0 if male
#     blhisp      = 1 if Black or Hispanic, 0 otherwise
#     linc        = log of income
#
#   Excluded instruments (Z -- used ONLY in the first stage):
#     ssiratio    = ratio of SSI (Supplemental Security Income) payments
#                   to income. Reflects the generosity of the safety net
#                   in the individual's area, which affects insurance
#                   take-up but should not directly affect drug spending.
#     lowincome   = indicator for low income bracket
#     multlc      = whether the employer firm has multiple locations
#                   (firms with multiple locations are more likely to
#                   offer insurance, but this firm characteristic should
#                   not directly cause higher drug expenditure)
#     firmsz      = employer firm size (larger firms more likely to offer
#                   insurance)
#
# WHY COMPLETE CASES:
# -------------------
# We use `dropna()` to ensure all variables are observed for every
# individual in the sample. This is critical because IV estimation
# requires the same sample across first-stage and second-stage regressions.
# Mismatched samples would invalidate the procedure.
#
df = pd.read_excel("Chap4_data.xlsx")

x2list = ['totchr', 'age', 'female', 'blhisp', 'linc']
dep_var = 'ldrugexp'
endog_var = 'hi_empunion'
instruments = ['ssiratio', 'lowincome', 'multlc', 'firmsz']

# Keep complete cases
all_vars = [dep_var, endog_var] + x2list + instruments
df_iv = df[all_vars].dropna()

print(f"\nSample size: {len(df_iv)}")
print(f"\nSummary of key variables:")
print(df_iv[[dep_var, endog_var] + instruments].describe().round(3))

# INTERPRETATION:
# ---------------
# Check the summary statistics carefully:
#   - ldrugexp: The log of drug expenditure. The mean and spread tell us
#     about the typical level and variation in drug spending.
#   - hi_empunion: Since this is binary (0/1), the mean gives us the
#     proportion of individuals with employer/union insurance. If it is,
#     say, 0.60, then 60% of the sample has such insurance.
#   - ssiratio: Look at variation -- an instrument with very little
#     variation across individuals will be a weak instrument.
#   - multlc, firmsz: These are firm-level characteristics. Check their
#     distributions to understand instrument variation.


# ============================================================
# 2. OLS ESTIMATION (for comparison)
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Estimates the structural equation by Ordinary Least Squares (OLS):
#
#     ldrugexp = beta0 + beta1*hi_empunion + beta2*totchr + beta3*age
#              + beta4*female + beta5*blhisp + beta6*linc + u
#
# This provides a BASELINE estimate of beta1 (the effect of insurance on
# drug expenditure). However, if hi_empunion is endogenous (correlated
# with u), then OLS produces BIASED and INCONSISTENT estimates of beta1.
#
# WHY OLS FAILS WITH ENDOGENEITY:
# --------------------------------
# OLS minimizes sum of squared residuals, which requires E(X'u) = 0 for
# consistency. When hi_empunion is correlated with u (endogenous), this
# condition fails, and:
#
#   plim(beta1_OLS) = beta1_true + Cov(hi_empunion, u) / Var(hi_empunion)
#
# The bias term [Cov(hi_empunion, u) / Var(hi_empunion)] does NOT vanish
# as sample size grows -- OLS is INCONSISTENT, not just biased.
#
# Direction of bias: If healthier/more health-conscious people both
# (a) seek insurance and (b) spend less on drugs (or vice versa), the
# bias could go in either direction. We need IV to determine the true
# causal effect.
#
# CODE DETAILS:
# -------------
# - sm.add_constant(): adds a column of 1's for the intercept. In Stata,
#   the constant is added automatically; in statsmodels, you must add it.
# - .fit(cov_type='HC1'): uses heteroskedasticity-robust standard errors,
#   equivalent to Stata's `vce(robust)`. HC1 applies a degrees-of-freedom
#   correction: n/(n-k) * the HC0 sandwich estimator.
# - .summary2().tables[1]: extracts just the coefficient table (cleaner
#   output than the full summary).
#
print("\n" + "=" * 60)
print("2. OLS ESTIMATION (Baseline)")
print("=" * 60)

X_ols = sm.add_constant(df_iv[[endog_var] + x2list])
y = df_iv[dep_var]
model_ols = sm.OLS(y, X_ols).fit(cov_type='HC1')
print(model_ols.summary2().tables[1].round(4))

# INTERPRETATION:
# ---------------
# Key things to look for in the OLS output:
#   1. The coefficient on hi_empunion: This is the OLS estimate of the
#      "effect" of insurance on log drug expenditure. HOWEVER, this likely
#      reflects BOTH the true causal effect AND the endogeneity bias.
#      We will compare this to IV estimates below.
#   2. The R-squared: OLS typically has a higher R-squared than IV because
#      OLS exploits ALL variation in X1 (including the endogenous part),
#      while IV uses only the exogenous variation induced by instruments.
#   3. Other coefficients (totchr, age, female, etc.): These exogenous
#      controls should be reasonably stable across OLS and IV if the
#      instruments are valid.
#   4. Robust standard errors: Look for the HC1 designation. Without
#      robust SEs, inference could be misleading if errors are
#      heteroskedastic (very common with medical expenditure data).


# ============================================================
# 3. IV ESTIMATION: 2SLS (Just-Identified)
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Estimates the structural equation using Two-Stage Least Squares (2SLS)
# with ONE excluded instrument (ssiratio) for ONE endogenous regressor
# (hi_empunion). This is a JUST-IDENTIFIED model because:
#
#     Number of excluded instruments (1) = Number of endogenous regressors (1)
#
# HOW 2SLS WORKS:
# ---------------
# Stage 1: Regress the endogenous variable on ALL instruments (both
#           excluded instruments Z and included exogenous regressors X2):
#
#           hi_empunion = pi0 + pi1*ssiratio + pi2*totchr + ... + v
#
#           Save the fitted values: hi_empunion_hat
#
# Stage 2: Replace hi_empunion with hi_empunion_hat in the structural
#           equation and estimate by OLS:
#
#           ldrugexp = beta0 + beta1*hi_empunion_hat + beta2*totchr + ... + u
#
# WHY THIS WORKS:
# ---------------
# hi_empunion_hat is a linear combination of the instruments (Z, X2), which
# are all exogenous. Therefore hi_empunion_hat is exogenous too, and
# Cov(hi_empunion_hat, u) = 0. The 2SLS estimator is CONSISTENT even when
# the original hi_empunion is endogenous.
#
# JUST-IDENTIFIED vs OVER-IDENTIFIED:
# ------------------------------------
# - Just-identified (# instruments = # endogenous):
#   * 2SLS, GMM, and LIML all produce IDENTICAL estimates
#   * Cannot test instrument validity (no overidentifying restrictions)
#   * The IV estimand is exactly determined
#
# - Over-identified (# instruments > # endogenous):
#   * 2SLS, GMM, and LIML generally produce DIFFERENT estimates
#   * CAN test instrument validity via the Hansen J test
#   * GMM is more efficient than 2SLS under heteroskedasticity
#
# LINEARMODELS SYNTAX:
# --------------------
# IV2SLS(dependent, exog, endog, instruments)
#   - dependent: the y variable (ldrugexp)
#   - exog: exogenous regressors INCLUDING constant (X2 + intercept)
#   - endog: endogenous regressors (hi_empunion)
#   - instruments: EXCLUDED instruments only (ssiratio)
#
# This maps to Stata's:
#   ivregress 2sls ldrugexp (hi_empunion = ssiratio) $x2list, vce(robust)
#
# Note: In linearmodels, `exog` must include the constant. The constant
# is NOT automatically added (unlike Stata).
#
print("\n" + "=" * 60)
print("3. 2SLS - JUST-IDENTIFIED (1 instrument: ssiratio)")
print("=" * 60)

# Stata: ivregress 2sls ldrugexp (hi_empunion = ssiratio) $x2list, vce(robust) first

# Using linearmodels
y_iv = df_iv[dep_var]
endog = df_iv[[endog_var]]
exog = sm.add_constant(df_iv[x2list])
instruments_just = df_iv[['ssiratio']]

model_2sls_just = IV2SLS(y_iv, exog, endog, instruments_just).fit(cov_type='robust')
print(model_2sls_just.summary)

# INTERPRETATION:
# ---------------
# Compare the 2SLS coefficient on hi_empunion with the OLS estimate:
#   - If the 2SLS estimate is LARGER than OLS, it suggests OLS was biased
#     downward (perhaps healthier people select into insurance AND spend
#     less on drugs, creating negative omitted variable bias).
#   - If the 2SLS estimate is SMALLER, OLS was biased upward.
#   - The 2SLS standard errors will almost always be LARGER than OLS,
#     because IV only uses the exogenous variation in X1 (induced by Z),
#     which is a subset of total variation. This is the "price" of
#     consistency -- you trade efficiency for unbiasedness.
#   - The R-squared may be negative or very low in IV regressions. This
#     is normal and NOT a problem -- R-squared has no meaningful
#     interpretation in IV models.


# ============================================================
# 4. FIRST STAGE
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Explicitly estimates the first-stage regression, which is the regression
# of the endogenous variable on all instruments and exogenous controls:
#
#     hi_empunion = pi0 + pi1*ssiratio + pi2*totchr + pi3*age
#                 + pi4*female + pi5*blhisp + pi6*linc + v
#
# The key question is: Is ssiratio a STRONG instrument?
#
# WHY THE FIRST STAGE MATTERS:
# ----------------------------
# The relevance condition requires Cov(Z, X1) != 0, i.e., pi1 != 0.
# If pi1 is close to zero (a "weak instrument"), then:
#
#   1. 2SLS estimates are biased toward OLS (the bias is proportional to
#      1/F, where F is the first-stage F-statistic)
#   2. Standard errors are unreliable (too small), leading to over-
#      rejection of null hypotheses
#   3. Confidence intervals have incorrect coverage
#
# STOCK-YOGO CRITICAL VALUES:
# ----------------------------
# Staiger & Stock (1997) and Stock & Yogo (2005) established that:
#
#   - F > 10 is the commonly cited "rule of thumb" for a strong instrument
#     in the case of one endogenous regressor.
#   - More precisely, Stock-Yogo provides critical values based on:
#     * The number of endogenous regressors
#     * The number of excluded instruments
#     * The desired maximal bias (relative to OLS) or size distortion
#   - For ONE endogenous regressor and ONE instrument, the 10% maximal
#     bias critical value is F = 16.38 (stricter than the rule of thumb).
#
# CODE DETAILS:
# -------------
# - We use statsmodels OLS here (not linearmodels) to get a familiar
#   regression output for the first stage.
# - model_first.fvalue: this is the OVERALL F-statistic for the regression,
#   not the partial F for just the instrument. For a just-identified model
#   with one excluded instrument, the relevant statistic is the t-squared
#   (or equivalently the F) on ssiratio alone. The overall F is shown for
#   convenience; ideally, we would compute the partial F (done in Section 11).
#
print("\n" + "=" * 60)
print("4. FIRST STAGE REGRESSION")
print("=" * 60)

X_first = sm.add_constant(df_iv[['ssiratio'] + x2list])
model_first = sm.OLS(df_iv[endog_var], X_first).fit(cov_type='HC1')
print(model_first.summary2().tables[1].round(4))
print(f"\n  First-stage F-statistic for ssiratio: {model_first.fvalue:.2f}")
print(f"  Rule of thumb: F > 10 means instruments are strong")
print(f"  Result: {'STRONG' if model_first.fvalue > 10 else 'WEAK'} instrument")

# INTERPRETATION:
# ---------------
# In the first-stage output, focus on:
#   1. The coefficient on ssiratio: Is it statistically significant?
#      A large, significant coefficient means ssiratio has strong
#      predictive power for hi_empunion (the relevance condition holds).
#   2. The t-statistic on ssiratio: The squared t-statistic is the partial
#      F-statistic (for one excluded instrument). Compare to 10 (or the
#      Stock-Yogo critical value of 16.38 for 10% maximal bias).
#   3. The overall F-statistic: This tests whether ALL regressors jointly
#      predict hi_empunion. It is not the same as the partial F for the
#      excluded instrument, but for a single excluded instrument, the
#      partial F can be obtained as t^2(ssiratio).
#   4. Sign of the coefficient: Does ssiratio increase or decrease the
#      probability of having employer insurance? This should make economic
#      sense given the instrument's definition.


# ============================================================
# 5. IV ESTIMATION: 2SLS (Over-Identified)
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Estimates 2SLS with TWO excluded instruments (ssiratio and multlc) for
# ONE endogenous regressor (hi_empunion). Since 2 > 1, the model is
# OVER-IDENTIFIED by one degree of overidentification.
#
# WHY OVER-IDENTIFICATION MATTERS:
# ---------------------------------
# With more instruments than endogenous regressors, we have "extra"
# moment conditions (more equations than unknowns in the GMM framework).
# This provides:
#
#   1. Potentially more efficient estimates (more information used)
#   2. The ability to TEST instrument validity via the Hansen J test
#      (Section 10). The overidentifying restrictions ask: "Do ALL
#      instruments produce the same estimate?" If they disagree, at
#      least one may be invalid.
#
# In the 2SLS framework with overidentification, the first stage projects
# hi_empunion onto the space spanned by [ssiratio, multlc, X2, constant],
# and the fitted values capture a richer set of exogenous variation than
# with a single instrument.
#
# TRADE-OFF:
# ----------
# More instruments CAN increase efficiency, but too many instruments
# (especially weak ones) can:
#   - Increase finite-sample bias (2SLS bias grows with the number of
#     instruments)
#   - Make the 2SLS estimator behave more like OLS (which is biased)
# This is why LIML (Section 7) is often preferred with many instruments.
#
# Stata equivalent:
#   ivregress 2sls ldrugexp (hi_empunion = ssiratio multlc) $x2list, vce(robust)
#
print("\n" + "=" * 60)
print("5. 2SLS - OVER-IDENTIFIED (2 instruments: ssiratio, multlc)")
print("=" * 60)

instruments_over = df_iv[['ssiratio', 'multlc']]
model_2sls_over = IV2SLS(y_iv, exog, endog, instruments_over).fit(cov_type='robust')
print(model_2sls_over.summary)

# INTERPRETATION:
# ---------------
# Compare the over-identified 2SLS with the just-identified 2SLS:
#   1. If the estimates are similar, this is reassuring -- both instruments
#      point to the same causal effect.
#   2. If estimates differ substantially, one or both instruments may be
#      weak or invalid. Investigate further with the Hansen J test
#      (Section 10) and weak instrument diagnostics (Section 11).
#   3. Standard errors may be smaller (more efficient) due to using more
#      instrument variation, but this is not guaranteed.
#   4. The overidentified model allows us to perform the Hansen J test
#      for instrument validity, which is done in Section 10.


# ============================================================
# 6. GMM ESTIMATION
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Estimates the model using the Generalized Method of Moments (GMM) with
# the same two excluded instruments (ssiratio, multlc). GMM is a more
# general estimation framework than 2SLS.
#
# 2SLS vs GMM -- WHAT IS THE DIFFERENCE?
# ----------------------------------------
# Both 2SLS and GMM use the same moment conditions: E[Z'u] = 0 (instruments
# are uncorrelated with the structural error). The difference is in HOW
# they weight these moment conditions when the model is over-identified:
#
#   - 2SLS uses a weighting matrix based on (Z'Z), which is EFFICIENT only
#     under homoskedasticity (constant variance of u).
#
#   - GMM uses an OPTIMAL weighting matrix W = [E(Z'uu'Z)]^{-1}, which
#     accounts for heteroskedasticity. This makes GMM EFFICIENT in the
#     class of IV estimators under heteroskedasticity.
#
# KEY POINTS:
# -----------
#   1. Under HOMOSKEDASTICITY: 2SLS and GMM are both efficient; 2SLS is
#      preferred because it is simpler and has better finite-sample properties.
#   2. Under HETEROSKEDASTICITY: GMM is asymptotically more efficient than
#      2SLS because it optimally weights the moment conditions.
#   3. In a JUST-IDENTIFIED model: 2SLS = GMM = LIML (all three are
#      algebraically identical because there is only one way to combine
#      the moments).
#   4. GMM requires estimating the weighting matrix, which introduces
#      additional finite-sample variability. With small samples, 2SLS
#      may actually perform better despite being theoretically less efficient.
#
# The `cov_type='robust'` option ensures heteroskedasticity-robust standard
# errors and uses the Eicker-Huber-White sandwich estimator for inference.
#
# Stata equivalent:
#   ivregress gmm ldrugexp (hi_empunion = ssiratio multlc) $x2list, wmatrix(robust)
#
print("\n" + "=" * 60)
print("6. GMM ESTIMATION (Efficient under heteroskedasticity)")
print("=" * 60)

model_gmm = IVGMM(y_iv, exog, endog, instruments_over).fit(cov_type='robust')
print(model_gmm.summary)

# INTERPRETATION:
# ---------------
# Compare GMM estimates with 2SLS estimates:
#   1. If coefficients are very similar, heteroskedasticity is not causing
#      large efficiency differences, and 2SLS is fine.
#   2. If they differ noticeably, GMM is exploiting heteroskedasticity
#      information to produce better-weighted estimates.
#   3. GMM standard errors may be smaller than 2SLS (reflecting efficiency
#      gains), but check that the point estimates are sensible.
#   4. The GMM model also reports the Hansen J-statistic (used in Section
#      10 for the overidentification test). This is a key advantage of
#      GMM estimation -- the J-stat falls out naturally from the GMM
#      objective function.


# ============================================================
# 7. LIML ESTIMATION
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Estimates the model using Limited Information Maximum Likelihood (LIML)
# with ALL FOUR excluded instruments (ssiratio, lowincome, multlc, firmsz).
#
# WHAT IS LIML?
# -------------
# LIML is a maximum likelihood estimator that uses only the "limited
# information" from the structural equation and the first-stage equation
# (as opposed to "full information" methods like 3SLS/FIML that would
# use all equations in a simultaneous system).
#
# LIML can also be expressed as a k-class estimator with k = LIML
# eigenvalue, which is always >= 1. For 2SLS, k = 1 exactly.
#
# WHY USE LIML?
# -------------
#   1. LIML is LESS BIASED than 2SLS when instruments are weak. The
#      finite-sample bias of 2SLS is approximately (K/n) * sigma^2,
#      where K is the number of instruments. LIML's bias does not grow
#      with K in the same way.
#   2. LIML has better coverage properties (confidence intervals are
#      closer to nominal levels) with weak instruments.
#   3. When instruments are strong, LIML and 2SLS are very similar.
#   4. LIML is median-unbiased (the median of its sampling distribution
#      equals the true parameter) even with weak instruments, while
#      2SLS is not.
#
# TRADE-OFFS:
# -----------
#   - LIML can have HEAVIER TAILS than 2SLS (larger variance, possibly
#     no finite moments), so extreme estimates are more likely.
#   - In practice, if the first-stage F is well above 10, 2SLS and LIML
#     give very similar results and either is fine.
#   - If F is borderline (5-15), LIML is preferred for reduced bias.
#
# NOTE: Here we use all 4 instruments, making the model over-identified
# by 3 degrees. With many instruments, LIML's advantage over 2SLS in
# terms of bias reduction is more pronounced.
#
# Stata equivalent:
#   ivregress liml ldrugexp (hi_empunion = ssiratio lowincome multlc firmsz) ///
#       $x2list, vce(robust)
#
print("\n" + "=" * 60)
print("7. LIML ESTIMATION (Less biased with weak instruments)")
print("=" * 60)

instruments_all = df_iv[['ssiratio', 'lowincome', 'multlc', 'firmsz']]
model_liml = IVLIML(y_iv, exog, endog, instruments_all).fit(cov_type='robust')
print(model_liml.summary)

# INTERPRETATION:
# ---------------
# Compare LIML to 2SLS and GMM:
#   1. If LIML and 2SLS give SIMILAR coefficients: Instruments are likely
#      strong, and finite-sample bias is not a concern. The LIML k-value
#      should be close to 1.
#   2. If LIML and 2SLS DIVERGE substantially: This is a RED FLAG for weak
#      instruments. The 2SLS estimate may be biased toward OLS, while LIML
#      is less affected. Trust LIML more in this case.
#   3. LIML standard errors may be slightly larger than 2SLS due to the
#      heavier tails of its sampling distribution.
#   4. With 4 instruments (3 overidentifying restrictions), the model is
#      heavily over-identified. The Hansen J test (Section 10) can test
#      whether all instruments are jointly valid.


# ============================================================
# 8. COMPARISON OF ESTIMATORS
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Creates a side-by-side table comparing the coefficient estimates from
# all five models: OLS, 2SLS (just-identified), 2SLS (over-identified),
# GMM, and LIML. This is the Python equivalent of Stata's
# `estimates table` command.
#
# WHY COMPARE?
# ------------
# Comparing estimators serves several diagnostic purposes:
#
#   1. OLS vs IV: The difference reveals the magnitude and direction of
#      endogeneity bias. If OLS and IV differ substantially, endogeneity
#      is a real concern (formally tested in Section 9).
#
#   2. 2SLS_just vs 2SLS_over: If these differ substantially, the extra
#      instrument may be providing different information, suggesting
#      possible instrument invalidity.
#
#   3. 2SLS vs GMM: Large differences suggest heteroskedasticity is
#      important for efficiency. If similar, the simpler 2SLS is adequate.
#
#   4. 2SLS vs LIML: Large differences suggest weak instruments are
#      biasing 2SLS toward OLS. LIML is more robust to this bias.
#
#   5. Exogenous controls: Coefficients on totchr, age, female, blhisp,
#      and linc should be relatively STABLE across estimators. Large
#      changes could indicate misspecification.
#
print("\n" + "=" * 60)
print("8. COMPARISON OF ESTIMATORS")
print("=" * 60)

comparison = pd.DataFrame({
    'OLS': model_ols.params[[endog_var] + x2list].round(4),
    '2SLS_just': [model_2sls_just.params[endog_var]] + [model_2sls_just.params[v] for v in x2list],
    '2SLS_over': [model_2sls_over.params[endog_var]] + [model_2sls_over.params[v] for v in x2list],
    'GMM': [model_gmm.params[endog_var]] + [model_gmm.params[v] for v in x2list],
    'LIML': [model_liml.params[endog_var]] + [model_liml.params[v] for v in x2list],
}, index=[endog_var] + x2list)

print(comparison.round(4))

# INTERPRETATION:
# ---------------
# This comparison table is one of the most informative outputs in the script.
# Key patterns to look for:
#
#   1. The hi_empunion row: Compare across columns. The OLS estimate may
#      differ markedly from the IV estimates. If ALL IV estimates are
#      similar to each other but different from OLS, this strongly suggests
#      endogeneity was present and the IV correction is working.
#
#   2. Stability of controls: The coefficients on totchr, age, female,
#      blhisp, and linc should be broadly similar across columns. If they
#      are, the IV specification is well-behaved.
#
#   3. 2SLS_just vs 2SLS_over: Ideally close. Large differences warrant
#      investigating the validity of the added instrument (multlc).
#
#   4. The magnitude of the IV effect: If the IV estimate of beta1 is
#      much larger than OLS, this is common in health economics -- OLS
#      typically underestimates the true effect of insurance because
#      healthier people select into insurance.


# ============================================================
# 9. ENDOGENEITY TEST (Durbin-Wu-Hausman)
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Performs the Durbin-Wu-Hausman (DWH) test for endogeneity of
# hi_empunion. This is one of the most important specification tests
# in IV analysis.
#
# THE FUNDAMENTAL QUESTION:
# -------------------------
# Is hi_empunion actually endogenous? If NOT, we should use OLS (which
# is more efficient). If YES, we MUST use IV (which is consistent).
#
# HYPOTHESES:
#   H0: hi_empunion is exogenous (OLS is consistent AND efficient)
#   H1: hi_empunion is endogenous (OLS is inconsistent; IV is needed)
#
# THE CONTROL FUNCTION APPROACH (Manual DWH Test):
# ------------------------------------------------
# This implements the DWH test using the "augmented regression" or
# "control function" method, which is algebraically equivalent to the
# standard Hausman test but is robust to heteroskedasticity:
#
# Step 1: Estimate the first-stage regression:
#         hi_empunion = pi0 + pi1*ssiratio + pi2*totchr + ... + v
#         Save the residuals: v1hat = hi_empunion - hi_empunion_hat
#
# Step 2: Include v1hat as an additional regressor in the structural
#         equation:
#         ldrugexp = beta0 + beta1*hi_empunion + beta2*totchr + ...
#                  + delta*v1hat + u*
#
# Step 3: Test H0: delta = 0 using a t-test (with robust SEs).
#
# INTUITION:
# ----------
# The residual v1hat captures the "endogenous part" of hi_empunion
# (the part correlated with the structural error u). If delta = 0,
# then this endogenous part does not affect y, meaning hi_empunion
# is effectively exogenous. If delta != 0, the endogenous component
# matters, confirming endogeneity.
#
# WHY THIS WORKS:
# ---------------
# We can decompose hi_empunion = hi_empunion_hat + v1hat, where
# hi_empunion_hat is the predicted value from the first stage (exogenous)
# and v1hat is the residual (potentially correlated with u). Testing
# whether v1hat matters in the structural equation is equivalent to
# testing whether hi_empunion's endogenous variation affects the outcome.
#
# STATA EQUIVALENT:
# -----------------
# This replicates `estat endogenous` after `ivregress 2sls`, and also
# the manual approach in the Stata do-file:
#   quietly regress hi_empunion ssiratio $x2list
#   quietly predict v1hat, resid
#   quietly regress ldrugexp hi_empunion v1hat $x2list, vce(robust)
#   test v1hat
#
# NOTE ON ROBUSTNESS:
# -------------------
# Using vce(robust) / cov_type='HC1' in Step 2 makes the test robust
# to heteroskedasticity. The standard Hausman test (without robust SEs)
# can give incorrect results under heteroskedasticity. This robust
# version is sometimes called the "regression-based Hausman test" or
# the "Durbin-Wu-Hausman-Regression" test.
#
print("\n" + "=" * 60)
print("9. ENDOGENEITY TEST (Durbin-Wu-Hausman)")
print("=" * 60)

# Manual DWH test:
# Step 1: Regress endogenous variable on instruments + exogenous
X_first_stage = sm.add_constant(df_iv[['ssiratio'] + x2list])
first_stage = sm.OLS(df_iv[endog_var], X_first_stage).fit()
v1hat = first_stage.resid

# Step 2: Include residual in structural equation
X_aug = sm.add_constant(df_iv[[endog_var] + x2list])
X_aug = X_aug.copy()
X_aug['v1hat'] = v1hat.values
model_aug = sm.OLS(y, X_aug).fit(cov_type='HC1')

t_v1hat = model_aug.tvalues['v1hat']
p_v1hat = model_aug.pvalues['v1hat']

print(f"  t-statistic on v1hat: {t_v1hat:.4f}")
print(f"  p-value:              {p_v1hat:.4f}")
if p_v1hat < 0.05:
    print("  -> REJECT H0: hi_empunion IS endogenous -> Use IV")
else:
    print("  -> FAIL TO REJECT H0: No evidence of endogeneity -> OLS may be OK")

# INTERPRETATION:
# ---------------
# Decision rule at the 5% significance level:
#   - If p-value < 0.05: REJECT H0. There IS evidence of endogeneity.
#     OLS is inconsistent, and we should use IV estimation (2SLS, GMM,
#     or LIML). The cost is larger standard errors, but the estimates
#     are consistent.
#   - If p-value >= 0.05: FAIL TO REJECT H0. There is no strong
#     evidence of endogeneity. OLS may be adequate, and since it is more
#     efficient, we might prefer OLS. BUT -- failing to reject does NOT
#     prove exogeneity. The test may lack power if the instruments are
#     weak.
#
# CAUTION: The DWH test is only as good as the instruments. If
# instruments are weak, the test has low power (unlikely to detect
# endogeneity even when it exists). Always check instrument strength
# (Section 4 and 11) alongside the DWH test.


# ============================================================
# 10. OVERIDENTIFICATION TEST (Hansen J)
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Performs the Hansen J test (also called the Sargan-Hansen test) for the
# validity of overidentifying restrictions. This test is ONLY possible
# when the model is OVER-IDENTIFIED (more instruments than endogenous
# regressors).
#
# THE FUNDAMENTAL QUESTION:
# -------------------------
# Are ALL instruments valid (i.e., do they all satisfy the exclusion
# restriction Cov(Z, u) = 0)?
#
# HYPOTHESES:
#   H0: All instruments are valid (all overidentifying restrictions hold)
#   H1: At least one instrument is invalid (at least one instrument is
#       correlated with the structural error u)
#
# HOW THE TEST WORKS:
# -------------------
# After GMM estimation, compute:
#
#   J = n * u_hat' * Z * W * Z' * u_hat
#
# where u_hat are the GMM residuals, Z is the matrix of instruments,
# and W is the GMM optimal weighting matrix. Under H0:
#
#   J ~ chi-squared(q)
#
# where q = (number of instruments) - (number of endogenous regressors)
#         = degree of overidentification.
#
# For our over-identified model with 2 instruments and 1 endogenous
# regressor, q = 2 - 1 = 1. So J ~ chi-squared(1) under H0.
#
# LIMITATIONS:
# ------------
#   1. The J test can only detect that SOME instruments are invalid,
#      not WHICH ones. If the test rejects, you need economic reasoning
#      to decide which instrument(s) to drop.
#   2. If ALL instruments are invalid in the same direction, the J test
#      may NOT reject (it tests whether instruments agree with each other,
#      not whether they are all exogenous).
#   3. The test requires at least one overidentifying restriction. In a
#      just-identified model, the J-statistic is identically zero.
#   4. The test is a JOINT test of instrument validity AND correct model
#      specification. Rejection could mean the instruments are bad OR the
#      structural model is misspecified.
#
# STATA EQUIVALENT:
#   ivregress gmm ldrugexp (hi_empunion = ssiratio multlc) $x2list, wmatrix(robust)
#   estat overid
#
print("\n" + "=" * 60)
print("10. OVERIDENTIFICATION TEST")
print("=" * 60)

# The GMM model from linearmodels provides the J-test
print(f"  Hansen J-statistic: {model_gmm.j_stat.stat:.4f}")
print(f"  p-value:            {model_gmm.j_stat.pval:.4f}")
if model_gmm.j_stat.pval > 0.05:
    print("  -> FAIL TO REJECT H0: Instruments are valid (overidentifying restrictions hold)")
else:
    print("  -> REJECT H0: At least one instrument may be invalid")

# INTERPRETATION:
# ---------------
# Decision rule at the 5% significance level:
#   - If p-value > 0.05: FAIL TO REJECT H0. The instruments pass the
#     overidentification test. This is NECESSARY (but NOT sufficient)
#     evidence for instrument validity. The instruments are at least
#     consistent with each other.
#   - If p-value <= 0.05: REJECT H0. At least one instrument may be
#     invalid. Consider:
#     (a) Dropping suspect instruments one at a time to see which one
#         causes the rejection.
#     (b) Re-examining the economic argument for each instrument's
#         exclusion restriction.
#     (c) Whether model misspecification (e.g., omitted variables,
#         wrong functional form) could explain the rejection.
#
# IMPORTANT CAVEAT: Passing the J test does NOT guarantee instruments
# are valid. If all instruments are invalid in the same way (e.g., all
# correlated with the same omitted variable), the J test will not detect
# the problem. The exclusion restriction is fundamentally an UNTESTABLE
# assumption that relies on economic reasoning, not statistical tests.


# ============================================================
# 11. WEAK INSTRUMENTS
# ============================================================
#
# WHAT THIS SECTION DOES:
# -----------------------
# Provides comprehensive diagnostics for instrument strength, including:
#   (a) Correlations between the endogenous variable and each instrument
#   (b) Partial F-statistics for each instrument individually
#   (c) Sensitivity analysis: IV estimates using each instrument separately
#
# WHY WEAK INSTRUMENTS ARE A SERIOUS PROBLEM:
# --------------------------------------------
# If instruments are only weakly correlated with the endogenous regressor,
# IV estimation suffers from:
#
#   1. FINITE-SAMPLE BIAS: 2SLS estimates are biased TOWARD OLS. The
#      bias is approximately (number of instruments / first-stage F) times
#      the OLS bias. With F = 10 and 1 instrument, the 2SLS bias is
#      about 10% of the OLS bias.
#
#   2. SIZE DISTORTION: Hypothesis tests reject the null too often. A
#      nominal 5% test might actually reject 15-25% of the time with
#      weak instruments.
#
#   3. INCONSISTENT CONFIDENCE INTERVALS: Standard Wald-based CIs have
#      incorrect coverage, sometimes dramatically so. Anderson-Rubin
#      confidence sets are robust to weak instruments but are often
#      wider.
#
# PARTIAL F-STATISTIC:
# --------------------
# The partial F-statistic isolates the contribution of the EXCLUDED
# instrument(s) to predicting the endogenous variable, after partialing
# out the effect of the included exogenous regressors (X2).
#
# It is computed as:
#   F_partial = [(RSS_restricted - RSS_unrestricted) / q] /
#               [RSS_unrestricted / (n - k)]
#
# where:
#   RSS_restricted = RSS from regressing X1 on X2 only (no instruments)
#   RSS_unrestricted = RSS from regressing X1 on X2 AND instruments
#   q = number of excluded instruments being tested
#   n = sample size, k = number of regressors in unrestricted model
#
# This is equivalent to testing H0: pi_1 = pi_2 = ... = pi_q = 0 in
# the first-stage regression (i.e., all excluded instruments have zero
# coefficients).
#
# SENSITIVITY ANALYSIS:
# ---------------------
# Estimating IV models with each instrument separately (just-identified
# each time) reveals whether instruments produce similar estimates. If
# they diverge widely:
#   - Weak instruments: An instrument with a low F produces an imprecise
#     estimate that could be anywhere.
#   - Invalid instruments: An instrument that violates the exclusion
#     restriction will produce a biased estimate that differs from the
#     estimate produced by a valid instrument.
#
# STOCK-YOGO CRITICAL VALUES (for reference):
# -------------------------------------------
# For 1 endogenous regressor, maximal 2SLS relative bias = 10%:
#   1 instrument:  F > 16.38
#   2 instruments: F > 19.93
#   3 instruments: F > 22.30
#   4 instruments: F > 24.58
#
# For 1 endogenous regressor, maximal 2SLS size distortion = 10%:
#   1 instrument:  F > 16.38
#   2 instruments: F > 8.68
#   3 instruments: F > 6.46
#   4 instruments: F > 5.53
#
print("\n" + "=" * 60)
print("11. WEAK INSTRUMENT DIAGNOSTICS")
print("=" * 60)

print("\n--- Correlations between endogenous variable and instruments ---")
corr_iv = df_iv[[endog_var] + instruments].corr()[endog_var].drop(endog_var)
print(corr_iv.round(4))

# First-stage F for each instrument separately
print("\n--- First-Stage F-statistics (individual instruments) ---")
for inst in instruments:
    X_fs = sm.add_constant(df_iv[[inst] + x2list])
    fs_model = sm.OLS(df_iv[endog_var], X_fs).fit()
    # Partial F-test for the excluded instrument
    r_model = sm.OLS(df_iv[endog_var], sm.add_constant(df_iv[x2list])).fit()
    f_stat = ((r_model.ssr - fs_model.ssr) / 1) / (fs_model.ssr / fs_model.df_resid)
    print(f"  {inst:12s}: F = {f_stat:.2f}  {'STRONG' if f_stat > 10 else 'WEAK'}")

# Compare IV estimates with different instruments
print("\n--- IV Estimates with Different Single Instruments ---")
for i, inst in enumerate(instruments):
    try:
        model_i = IV2SLS(y_iv, exog, endog, df_iv[[inst]]).fit(cov_type='robust')
        print(f"  {inst:12s}: beta(hi_empunion) = {model_i.params[endog_var]:.4f}, "
              f"SE = {model_i.std_errors[endog_var]:.4f}")
    except Exception as e:
        print(f"  {inst:12s}: Error - {e}")

print("\n  Note: If estimates differ widely across instruments, this suggests")
print("  some instruments may be weak or invalid.")

# INTERPRETATION:
# ---------------
# Putting together all the weak instrument diagnostics:
#
# 1. CORRELATIONS: These give a raw sense of the strength of the
#    relationship between each instrument and hi_empunion. However,
#    correlations can be misleading because they do not control for
#    the exogenous regressors (X2). The partial F is more informative.
#
# 2. PARTIAL F-STATISTICS: For each instrument individually:
#    - F > 10 (rule of thumb) or F > 16.38 (Stock-Yogo 10% bias):
#      The instrument is "strong" for this endogenous regressor.
#    - F between 5 and 10: "Borderline weak" -- consider LIML or
#      Anderson-Rubin confidence sets.
#    - F < 5: "Weak" -- standard IV inference is unreliable. Do NOT
#      trust 2SLS point estimates or standard errors.
#
# 3. SENSITIVITY OF ESTIMATES: If all four instruments produce similar
#    beta(hi_empunion) values, this is strong evidence that:
#    (a) The instruments are all identifying the same causal parameter
#    (b) They are likely all valid (assuming at least one is valid)
#    If estimates diverge:
#    - Check which instruments have low F (weak instruments produce
#      erratic estimates)
#    - Consider that instruments with very different estimates may have
#      different local average treatment effects (LATE) if hi_empunion
#      is binary and the effect is heterogeneous
#    - Consider that one or more instruments may be invalid
#
# 4. OVERALL ASSESSMENT: Use multiple diagnostics together. No single
#    test is definitive. A strong instrument (high F) that produces an
#    estimate similar to other instruments and passes the J-test is the
#    most trustworthy.

print("\n" + "=" * 60)
print("END OF CHAPTER 4")
print("=" * 60)
