# MIT License
# Copyright (c) 2026 Tiraphap Fakthong
# See LICENSE file for full license text.
"""
=============================================================================
EC627 Microeconometrics - Chapter 5
Quantile Regression and Count Data Models
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

Datasets: Chap5_1_data.xlsx (medical expenditures), Chap5_2_data.xlsx (doctor visits)
This script replicates "Chap 5.do" in Python.
Designed to run on Google Colab.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib openpyxl

=============================================================================
OVERVIEW OF KEY TOPICS COVERED IN THIS SCRIPT
=============================================================================

This script covers two major families of microeconometric methods that extend
beyond the standard OLS framework:

PART 1 -- QUANTILE REGRESSION (QR) FOR CONTINUOUS OUTCOMES
    1.  OLS as a baseline: Estimating the conditional MEAN of log total
        medical expenditures as a function of insurance, chronic conditions,
        age, gender, and race.
    2.  Quantile regression at the 25th, 50th, and 75th percentiles:
        Estimating how covariates shift different points of the CONDITIONAL
        DISTRIBUTION of expenditures -- not just the center.
    3.  Side-by-side comparison of OLS vs. QR coefficients to see where
        covariate effects are uniform and where they differ across the
        distribution.
    4.  QR coefficient process plot: Estimating quantile regression at every
        5th percentile from 0.05 to 0.95, then plotting how each coefficient
        changes across quantiles.  The OLS coefficient is overlaid as a flat
        reference line.  Divergence from the OLS line reveals heterogeneous
        effects -- the hallmark insight of quantile regression.
    5.  Formal Wald-type test for equality of a coefficient at two different
        quantiles (q=0.25 vs q=0.75).  This tests the null hypothesis that
        the effect of a covariate is the same at both ends of the conditional
        distribution.
    6.  Converting log-scale QR results to dollar-level average marginal
        effects (AME) using the retransformation multiplier E[exp(x'beta)].

PART 2 -- QUANTILE REGRESSION WITH SIMULATED HETEROSKEDASTIC DATA
    7.  Monte Carlo simulation that generates data with known DGP whose error
        variance depends on x (multiplicative heteroskedasticity).  This
        demonstrates *why* QR coefficients vary across quantiles: when the
        error term interacts with a regressor, the slope on that regressor
        is literally different at every quantile of the conditional
        distribution.
    8.  Comparison of estimated QR coefficients against the theoretically
        predicted values from the DGP, confirming the estimator recovers
        the correct quantile-specific parameters.

PART 3 -- COUNT DATA MODELS
    9.  Poisson regression for doctor visits: The workhorse count model with
        the exponential conditional mean E[Y|X] = exp(X*beta).  Estimated
        with robust (sandwich / HC1) standard errors to guard against
        misspecification of the variance.
   10.  Negative Binomial regression: Allows the variance to exceed the mean
        (overdispersion), relaxing the Poisson equidispersion restriction
        Var[Y|X] = E[Y|X].
   11.  Overdispersion diagnostic: Comparing the unconditional variance to
        the unconditional mean of the count variable.  A ratio much larger
        than 1 is a red flag for Poisson.
   12.  Marginal effects at the mean (MEM): Computing dE[Y|X]/dX_j evaluated
        at the sample means of all covariates, which yields the effect of a
        one-unit change in X_j on the expected count for a "representative"
        individual.
   13.  Jittered quantile count regression: Adding Uniform(0,1) noise to
        integer counts to make the outcome quasi-continuous, then applying
        standard QR.  This allows the analyst to examine heterogeneous
        covariate effects across the distribution of counts, analogous to
        the continuous-outcome QR above.

KEY ECONOMETRIC CONCEPTS
    -   Conditional quantile vs conditional mean
    -   Heterogeneous treatment effects across the outcome distribution
    -   Retransformation / Duan smearing for log-dependent-variable models
    -   Poisson pseudo-MLE and the quasi-Poisson interpretation
    -   Overdispersion and the Negative Binomial extension
    -   Marginal effects in nonlinear (exponential-mean) models
    -   Machado-Santos Silva jittering approach for count QR
=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
# -----------------------------------------------------------------------
# IMPORTS AND CONFIGURATION
# -----------------------------------------------------------------------
# pandas  : Data manipulation (DataFrames, reading Excel files).
# numpy   : Numerical computing (arrays, random number generation, math).
# matplotlib : Plotting library for the QR coefficient process plots.
# scipy.stats : Statistical distributions, used here for the standard
#               normal CDF when computing p-values for the Wald test.
# statsmodels : The main econometrics library in Python.  We use:
#   - sm.OLS for ordinary least squares
#   - sm.GLM for generalized linear models (Poisson, Negative Binomial)
#   - QuantReg for linear quantile regression (Koenker-Bassett estimator)
#   - sm.add_constant to prepend a column of 1s (the intercept)
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

np.set_printoptions(precision=4)

print("=" * 60)
print("CHAPTER 5: QUANTILE REGRESSION & COUNT DATA MODELS")
print("=" * 60)

# ============================================================
# PART 1: QUANTILE REGRESSION FOR MEDICAL EXPENDITURES
# ============================================================
# -----------------------------------------------------------------------
# WHY QUANTILE REGRESSION?
# -----------------------------------------------------------------------
# OLS estimates E[Y | X] -- the conditional MEAN of Y given X.  It tells
# us how the *average* outcome shifts when a covariate changes by one unit.
# But the average can hide a lot:
#   - A drug might help sick patients enormously (top of the distribution)
#     while barely affecting healthy patients (bottom).
#   - Insurance might raise spending a lot for high spenders but modestly
#     for low spenders.
#
# Quantile regression (QR), introduced by Koenker and Bassett (1978),
# estimates CONDITIONAL QUANTILES:
#
#     Q_tau(Y | X) = X * beta(tau)       for tau in (0, 1)
#
# where Q_tau is the tau-th quantile of Y conditional on X.
#
# Each quantile has its OWN coefficient vector beta(tau).  If all those
# vectors are the same, the covariate affects every part of the conditional
# distribution equally (a pure location shift) and OLS suffices.  If they
# differ, the covariate has HETEROGENEOUS effects across the distribution.
#
# In this Part we model log total medical expenditure (ltotexp) as a
# function of:
#   - suppins  : supplementary private health insurance (0/1)
#   - totchr   : total number of chronic conditions
#   - age      : age in years
#   - female   : gender indicator (1 = female)
#   - white    : race indicator (1 = white)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 1: QUANTILE REGRESSION")
print("=" * 60)

# Load data
# -----------------------------------------------------------------------
# We read the medical expenditure dataset and drop any observation with a
# missing value in the dependent variable (ltotexp = log of total medical
# expenditure).  Using the log transformation is standard for expenditure
# data because the raw level distribution is heavily right-skewed, and the
# log helps compress the right tail, making a linear conditional quantile
# model more plausible.
# -----------------------------------------------------------------------
df = pd.read_excel("Chap5_1_data.xlsx")
df = df.dropna(subset=['ltotexp'])
print(f"\nSample size: {len(df)}")

key_vars = ['ltotexp', 'suppins', 'totchr', 'age', 'female', 'white']
print("\n--- Summary Statistics ---")
print(df[key_vars].describe().round(3))

# -----------------------------------------------------------------------
# INTERPRETATION -- Summary Statistics:
# Look at the mean and standard deviation of ltotexp to gauge the center
# and spread of (log) spending.  Also check the mean of dummy variables:
#   - suppins mean ~ 0.4 means ~40% have supplementary insurance
#   - female mean  ~ 0.6 means ~60% are female in this sample
#   - white mean   ~ 0.8 means ~80% are white
# The min/max of totchr tells you the range of chronic conditions.
# These moments help you sanity-check whether your regression coefficients
# are of plausible magnitude.
# -----------------------------------------------------------------------

# ============================================================
# 1. OLS (for comparison)
# ============================================================
# -----------------------------------------------------------------------
# OLS REGRESSION -- THE BASELINE
# -----------------------------------------------------------------------
# We first estimate the conditional mean model:
#
#     E[ltotexp | X] = beta_0 + beta_1*suppins + beta_2*totchr
#                      + beta_3*age + beta_4*female + beta_5*white
#
# Key code details:
#   - sm.add_constant(df[...]) prepends a column of 1s (the intercept).
#   - sm.OLS(y, X).fit(cov_type='HC1') uses White/Huber heteroskedasticity-
#     robust standard errors (HC1 = "Stata-default" small-sample adjustment
#     dividing by n-k instead of n).  This is the Python equivalent of
#     Stata's ", robust" option.
#   - .summary2().tables[1] prints only the coefficient table (skipping
#     the model-level statistics header that .summary() includes).
#
# These OLS coefficients estimate the effect of each covariate on the
# conditional MEAN of log expenditure.  Because the dependent variable is
# in logs, each coefficient is approximately a semi-elasticity: a one-unit
# increase in X_j changes log expenditure by beta_j, or equivalently
# changes expenditure by approximately 100*beta_j percent (exact for
# small effects; for large effects use exp(beta_j)-1).
# -----------------------------------------------------------------------
print("\n--- OLS Regression (Conditional Mean) ---")
X = sm.add_constant(df[['suppins', 'totchr', 'age', 'female', 'white']])
y = df['ltotexp']
model_ols = sm.OLS(y, X).fit(cov_type='HC1')
print(model_ols.summary2().tables[1].round(4))

# -----------------------------------------------------------------------
# INTERPRETATION -- OLS Results:
# Each coefficient tells you how the MEAN of log expenditure changes when
# the covariate increases by one unit, holding all else constant.
#   - suppins > 0: individuals with supplementary insurance have higher
#     average log spending.
#   - totchr > 0: each additional chronic condition raises average log
#     spending (this is typically the strongest predictor).
#   - female, white: demographic shifters of the conditional mean.
#
# Note: OLS gives ONE coefficient per variable -- a single summary of how
# the entire conditional distribution shifts.  It cannot tell us whether
# insurance raises spending more at the top or bottom of the distribution.
# That is exactly what quantile regression can reveal.
# -----------------------------------------------------------------------

# ============================================================
# 2. QUANTILE REGRESSION AT DIFFERENT QUANTILES
# ============================================================
# -----------------------------------------------------------------------
# QUANTILE REGRESSION AT tau = 0.25, 0.50, 0.75
# -----------------------------------------------------------------------
# For each quantile tau, we solve:
#
#     min_{beta} SUM_i rho_tau( y_i - x_i' * beta )
#
# where rho_tau(u) = u*(tau - I(u < 0)) is the asymmetric "check"
# (or "tick") loss function.
#   - At tau = 0.50 this reduces to minimizing the sum of absolute
#     deviations (LAD / median regression).
#   - At tau = 0.25, the loss penalizes UNDER-predictions more lightly
#     (weight 0.25) and OVER-predictions more heavily (weight 0.75),
#     so the fitted line tracks the lower part of the conditional
#     distribution.
#   - At tau = 0.75, the asymmetry reverses: over-predictions get the
#     lighter penalty, and the fitted line tracks the upper conditional
#     distribution.
#
# Code details:
#   - QuantReg(y, X) sets up the quantile regression model object.
#   - .fit(q=tau) estimates the model at quantile tau.
#   - The default standard errors in statsmodels QuantReg use the
#     Koenker (2005) kernel-based ("Powell sandwich") estimator.
#     These are asymptotically valid under heteroskedasticity.
# -----------------------------------------------------------------------
quantiles = [0.25, 0.50, 0.75]
qr_models = {}

for q in quantiles:
    print(f"\n--- Quantile Regression: q = {q} ---")
    model_qr = QuantReg(y, X).fit(q=q)
    qr_models[q] = model_qr
    print(model_qr.summary().tables[1])

# -----------------------------------------------------------------------
# INTERPRETATION -- QR at 0.25, 0.50, 0.75:
# Compare the three sets of coefficients.  For each covariate, ask:
#   "Does the coefficient CHANGE appreciably across quantiles?"
#
# If beta_j(0.25) < beta_j(0.50) < beta_j(0.75), the covariate has a
# LARGER effect for higher spenders.  This is evidence of HETEROGENEOUS
# EFFECTS across the conditional distribution of spending.
#
# Example interpretations:
#   - If suppins coefficient grows from q=0.25 to q=0.75, supplementary
#     insurance raises spending more for already-high spenders than for
#     low spenders (possibly moral hazard that intensifies at higher
#     utilization levels).
#   - If totchr coefficients are roughly equal across quantiles, chronic
#     conditions shift the entire distribution uniformly (a location shift).
#
# The median regression (q=0.50) is a natural "robust" alternative to
# OLS because the median is less sensitive to outliers than the mean.
# If OLS and median regression give similar coefficients, the conditional
# distribution is roughly symmetric and well-behaved.
# -----------------------------------------------------------------------

# ============================================================
# 3. COMPARISON TABLE: OLS vs QR
# ============================================================
# -----------------------------------------------------------------------
# SIDE-BY-SIDE COEFFICIENT TABLE
# -----------------------------------------------------------------------
# Collecting all four sets of estimates (OLS + three QR quantiles) into a
# single DataFrame makes comparison easy.  Each column is a model; each
# row is a covariate.
#
# What to look for:
#   - If the QR columns are all roughly equal and close to OLS, the
#     conditional distribution shifts uniformly with X (pure location
#     shift model) and OLS is adequate.
#   - If the QR columns differ systematically, the covariates have
#     heterogeneous effects and QR adds valuable information.
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARISON: OLS vs Quantile Regression")
print("=" * 60)

coef_names = ['const', 'suppins', 'totchr', 'age', 'female', 'white']
comparison = pd.DataFrame({'OLS': model_ols.params.round(4)})
for q in quantiles:
    comparison[f'QR({q})'] = qr_models[q].params.round(4)
comparison.index = coef_names
print(comparison)

# -----------------------------------------------------------------------
# INTERPRETATION -- Comparison Table:
# Read across each row.  For instance, if the "suppins" row reads:
#   OLS=0.28, QR(0.25)=0.35, QR(0.50)=0.25, QR(0.75)=0.20
# then insurance has a larger effect on the 25th percentile of log
# spending than on the 75th, and OLS averages over these heterogeneous
# effects.
#
# The intercept row shows how the "baseline" level of the conditional
# quantile varies.  Typically the intercept rises from q=0.25 to q=0.75
# simply because higher quantiles correspond to higher values of Y.
# -----------------------------------------------------------------------

# ============================================================
# 4. PLOT: HOW COEFFICIENTS CHANGE ACROSS QUANTILES
# ============================================================
# -----------------------------------------------------------------------
# QR COEFFICIENT PROCESS PLOT (Koenker-style plot)
# -----------------------------------------------------------------------
# This is the signature visualization of quantile regression analysis.
# We estimate the model at every 5th percentile from tau = 0.05 to
# tau = 0.95 (19 models in total), then plot beta_j(tau) vs tau for each
# covariate j.
#
# On each subplot:
#   - The BLUE CURVE is beta_j(tau), the QR coefficient as a function
#     of the quantile index tau.
#   - The BLUE SHADED BAND is the pointwise 95% confidence interval
#     around the QR coefficient.
#   - The RED DASHED LINE is the OLS coefficient (constant across tau
#     because OLS does not depend on the quantile).
#
# How to read the plot:
#   - If the blue curve is roughly flat and lies inside the red dashed
#     line, the covariate's effect is approximately constant across the
#     distribution -- OLS is sufficient.
#   - If the blue curve has a clear slope (upward or downward), the
#     covariate's effect varies across quantiles.  An upward slope means
#     the effect is LARGER for higher quantiles (high spenders).
#   - If the red dashed line sits OUTSIDE the blue confidence band at
#     some quantiles, the OLS estimate is a poor summary of the effect
#     at those quantiles.
#
# Code details:
#   - quantile_range = np.arange(0.05, 0.96, 0.05) creates the grid
#     [0.05, 0.10, ..., 0.95].
#   - We loop over this grid, estimate QuantReg at each tau, and store
#     the point estimate and confidence interval bounds for every
#     covariate.
#   - matplotlib's fill_between draws the confidence band; axhline
#     draws the constant OLS reference.
# -----------------------------------------------------------------------
print("\n--- Plotting QR coefficients across quantiles ---")

quantile_range = np.arange(0.05, 0.96, 0.05)
coef_results = {var: [] for var in coef_names}
ci_lower = {var: [] for var in coef_names}
ci_upper = {var: [] for var in coef_names}

for q in quantile_range:
    model_q = QuantReg(y, X).fit(q=q)
    for i, var in enumerate(coef_names):
        coef_results[var].append(model_q.params.iloc[i])
        ci_lower[var].append(model_q.conf_int().iloc[i, 0])
        ci_upper[var].append(model_q.conf_int().iloc[i, 1])

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, var in enumerate(coef_names):
    ax = axes[idx]
    ax.plot(quantile_range, coef_results[var], 'b-', linewidth=2, label='QR')
    ax.fill_between(quantile_range, ci_lower[var], ci_upper[var], alpha=0.2, color='blue')
    ax.axhline(y=model_ols.params.iloc[idx], color='red', linestyle='--', linewidth=1.5, label='OLS')
    ax.set_title(var, fontweight='bold')
    ax.set_xlabel('Quantile')
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle('Quantile Regression Coefficients Across Quantiles\n(with 95% CI, OLS in red)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('fig_ch5_qrcoef.png', dpi=150)
plt.show()
print("Figure saved: fig_ch5_qrcoef.png")

# -----------------------------------------------------------------------
# INTERPRETATION -- QR Coefficient Plot:
# This plot is the single most informative output of a quantile regression
# analysis.  For each covariate, visually assess:
#
# 1) Is the blue curve flat?
#    YES --> The effect is homogeneous across the distribution.
#            OLS captures it well.  The blue curve and red line will
#            approximately coincide.
#    NO  --> The effect is heterogeneous.  QR reveals information that
#            OLS obscures.
#
# 2) Does the blue curve trend upward?
#    YES --> The covariate has a LARGER positive effect on higher
#            conditional quantiles.  Example: if totchr slopes upward,
#            chronic conditions raise expenditures more for people who
#            are already high spenders.
#
# 3) Does the 95% CI band include the OLS line everywhere?
#    YES --> The QR coefficient at each quantile is not statistically
#            distinguishable from the OLS estimate.  But this does not
#            prove they are equal -- it could be a power issue.
#    NO  --> At quantiles where the band excludes the OLS line, the
#            conditional-quantile effect differs significantly from the
#            conditional-mean effect.
# -----------------------------------------------------------------------

# ============================================================
# 5. TEST EQUALITY ACROSS QUANTILES
# ============================================================
# -----------------------------------------------------------------------
# FORMAL TEST: EQUALITY OF COEFFICIENTS AT q=0.25 AND q=0.75
# -----------------------------------------------------------------------
# The plots give visual evidence, but we also want a formal test.
# We test the null hypothesis:
#
#     H0: beta_suppins(0.25) = beta_suppins(0.75)
#
# Under the null, the effect of supplementary insurance is the same at
# the 25th and 75th conditional percentiles.
#
# We construct a Wald-type z-statistic:
#
#     z = [ beta(0.25) - beta(0.75) ] / sqrt( se(0.25)^2 + se(0.75)^2 )
#
# This assumes the estimators at two distinct quantiles are asymptotically
# independent, which is a simplification.  (In finite samples, they are
# correlated because they use the same data.)  For a fully rigorous joint
# test, one would need the joint covariance matrix of the two QR
# estimators -- see Koenker (2005, Ch. 3).  The simpler version here is
# still informative and widely used.
#
# Under H0, z ~ N(0,1).  The two-sided p-value is
#   p = 2 * [ 1 - Phi(|z|) ]
# where Phi is the standard normal CDF (stats.norm.cdf).
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("TEST: Are coefficients equal across quantiles?")
print("=" * 60)

# Manual Wald-type test comparing q=0.25 and q=0.75
# For suppins:
b25 = qr_models[0.25].params['suppins']
b75 = qr_models[0.75].params['suppins']
se25 = qr_models[0.25].bse['suppins']
se75 = qr_models[0.75].bse['suppins']
z_stat = (b25 - b75) / np.sqrt(se25**2 + se75**2)
p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\nTest: beta_suppins(q=0.25) = beta_suppins(q=0.75)")
print(f"  beta(0.25) = {b25:.4f}, beta(0.75) = {b75:.4f}")
print(f"  z-statistic = {z_stat:.4f}")
print(f"  p-value = {p_val:.4f}")
if p_val < 0.05:
    print("  -> REJECT: Coefficients differ across quantiles (heterogeneous effects!)")
else:
    print("  -> Fail to reject: No evidence of different effects across quantiles")

# -----------------------------------------------------------------------
# INTERPRETATION -- Equality Test:
# If p < 0.05 we reject the null that the insurance effect is the same
# at q=0.25 and q=0.75.  This provides formal evidence of HETEROGENEOUS
# EFFECTS: supplementary insurance shifts the lower and upper parts of
# the expenditure distribution by different amounts.
#
# If p >= 0.05, we do NOT reject.  This does not prove equality -- it
# might just be that our sample is too small (low power).  Always
# combine the formal test with the visual evidence from the QR
# coefficient process plot.
#
# In applied work you would repeat this test for every covariate and
# possibly use an omnibus F-type (or Khmaladze-type) test that jointly
# tests whether ALL coefficients are equal across quantiles.  The
# pairwise z-test is a simple first pass.
# -----------------------------------------------------------------------

# ============================================================
# 6. MULTIPLIER FOR CONVERTING LOG QR TO LEVEL AME
# ============================================================
# -----------------------------------------------------------------------
# RETRANSFORMATION FROM LOG TO LEVEL: THE AME MULTIPLIER
# -----------------------------------------------------------------------
# Since our dependent variable is ltotexp = log(totexp), the QR
# coefficient beta_j(tau) tells us the effect on the LOG of expenditure.
# Often we want the effect in DOLLAR LEVELS.
#
# For the conditional quantile in levels (assuming Q_tau is estimated
# in logs):
#
#     Q_tau(totexp | X) = exp( X' * beta(tau) )
#
# The marginal effect of a continuous X_j in levels is:
#
#     dQ_tau / dX_j = beta_j(tau) * exp( X' * beta(tau) )
#
# Since exp(X' beta) varies across observations, we summarize by
# averaging over the sample:
#
#     Average Marginal Effect (AME) = beta_j(tau) * (1/n) * SUM_i exp(X_i' beta(tau))
#
# The quantity  (1/n) * SUM_i exp(X_i' beta(tau))  is the "multiplier."
#
# Code details:
#   - xb = qr_models[0.50].fittedvalues  --> the fitted log expenditure
#     for each observation at the median quantile.
#   - expxb = np.exp(xb)  --> the fitted LEVEL expenditure for each
#     observation.
#   - multiplier = expxb.mean()  --> the average fitted level across the
#     sample.
#   - AME = beta_suppins(0.50) * multiplier gives the dollar-level
#     effect of having supplementary insurance at the conditional median.
#
# Note: This is analogous to the "Duan smearing" approach used in OLS
# retransformation.  The exact retransformation depends on assumptions
# about the error distribution; this approach is a simple and commonly
# used approximation.
# -----------------------------------------------------------------------
print("\n--- Converting QR in Logs to AME in Levels ---")
xb = qr_models[0.50].fittedvalues
expxb = np.exp(xb)
multiplier = expxb.mean()
print(f"  Multiplier = E[exp(x'beta)] = {multiplier:.2f}")
print(f"  AME of suppins in levels at q=0.50: {qr_models[0.50].params['suppins'] * multiplier:.2f}")

# -----------------------------------------------------------------------
# INTERPRETATION -- AME in Levels:
# If the multiplier is, say, 3000, and beta_suppins(0.50) = 0.25, then:
#     AME = 0.25 * 3000 = $750
# This means supplementary insurance is associated with approximately $750
# higher median expenditure.
#
# The multiplier is large because exp(.) maps from the log scale back to
# the original dollar scale.  The AME gives stakeholders a tangible,
# dollar-valued measure of the insurance effect at the median.
#
# You could repeat this calculation at other quantiles to see how the
# dollar-level effect differs for low vs. high spenders.
# -----------------------------------------------------------------------

# ============================================================
# PART 2: QUANTILE REGRESSION FOR GENERATED HETEROSKEDASTIC DATA
# ============================================================
# -----------------------------------------------------------------------
# SIMULATION: WHY DO QR COEFFICIENTS VARY ACROSS QUANTILES?
# -----------------------------------------------------------------------
# This simulation creates data where we KNOW the true data-generating
# process (DGP), so we can verify that QR correctly recovers the
# quantile-specific parameters.
#
# The DGP is:
#     y = 1 + 1*x2 + 1*x3 + u
#     u = (0.1 + 0.5*x2) * e       where  e ~ N(0, 25)
#     x2 ~ chi-squared(1),  x3 ~ N(0, 25)
#
# The key feature is that the error term u is MULTIPLICATIVELY
# heteroskedastic: its standard deviation is proportional to (0.1 + 0.5*x2).
# This means:
#   - The variance of y increases with x2.
#   - The CONDITIONAL QUANTILE of y given (x2, x3) is:
#       Q_tau(y | x2, x3) = 1 + [1 + 0.5 * F^{-1}_e(tau)] * x2
#                              + 1 * x3
#                              + 0.1 * F^{-1}_e(tau)
#     where F^{-1}_e(tau) is the tau-th quantile of the error e.
#
# So the coefficient on x2 DEPENDS ON tau:
#     beta_x2(tau) = 1 + 0.5 * Q_e(tau)
#
# Meanwhile the coefficient on x3 is 1.0 at every quantile (because x3
# does not interact with the error).
#
# This demonstrates the fundamental insight: QR coefficients vary across
# quantiles when the error interacts with a regressor (a form of
# heteroskedasticity / heterogeneous effects).  OLS, by contrast, only
# recovers the mean coefficient which is 1.0 for both x2 and x3.
#
# Code details:
#   - np.random.seed(10101) ensures reproducibility (same draws each run).
#   - np.random.chisquare(1, n) draws x2 from the chi-squared(1)
#     distribution (always positive, right-skewed).
#   - 5 * np.random.randn(n) draws Normal(0, 25) for both x3 and e.
#   - u = (0.1 + 0.5*x2) * e creates the heteroskedastic composite error.
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2: QR FOR GENERATED HETEROSKEDASTIC DATA")
print("=" * 60)

np.random.seed(10101)
n = 10000
x2 = np.random.chisquare(1, n)
x3 = 5 * np.random.randn(n)
e = 5 * np.random.randn(n)
u = (0.1 + 0.5 * x2) * e
y_gen = 1 + 1 * x2 + 1 * x3 + u

X_gen = sm.add_constant(np.column_stack([x2, x3]))

# -----------------------------------------------------------------------
# ESTIMATION ON SIMULATED DATA
# -----------------------------------------------------------------------
# We estimate QR at tau = 0.25, 0.50, 0.75 and report the coefficient
# on x2 and x3.
#   - beta_x3 should be close to 1.0 at all three quantiles (since x3
#     does not interact with the error).
#   - beta_x2 should vary: it should be close to 1 + 0.5 * Q_e(tau)
#     at each quantile tau.
# -----------------------------------------------------------------------
print("\n--- Quantile Regression for Generated Data ---")
gen_results = {}
for q in [0.25, 0.50, 0.75]:
    model_gen = QuantReg(y_gen, X_gen).fit(q=q)
    gen_results[q] = model_gen.params
    print(f"  q={q}: beta_x2 = {model_gen.params[1]:.4f}, beta_x3 = {model_gen.params[2]:.4f}")

# Predicted coefficients
# -----------------------------------------------------------------------
# COMPARISON: ESTIMATED vs. THEORETICALLY PREDICTED COEFFICIENTS
# -----------------------------------------------------------------------
# We compute the empirical quantiles of the error draws e at tau = 0.25,
# 0.50, 0.75 using np.quantile.  Then the predicted coefficient on x2 is:
#
#     beta_x2(tau) = 1 + 0.5 * Q_e(tau)
#
# For e ~ N(0, 25):
#   - Q_e(0.25) ~ -3.37  --> predicted beta_x2(0.25) ~ -0.69
#   - Q_e(0.50) ~  0.00  --> predicted beta_x2(0.50) ~  1.00
#   - Q_e(0.75) ~ +3.37  --> predicted beta_x2(0.75) ~  2.69
#
# The estimated QR coefficients should be close to these predicted values
# (with n=10,000 they should be quite close).
# -----------------------------------------------------------------------
e_quantiles = np.quantile(e, [0.25, 0.50, 0.75])
print(f"\n  Predicted coefficient of x2:")
for q, eq in zip([0.25, 0.50, 0.75], e_quantiles):
    pred = 1 + 0.5 * eq
    print(f"    q={q}: predicted = {pred:.4f}, estimated = {gen_results[q][1]:.4f}")

# -----------------------------------------------------------------------
# INTERPRETATION -- Simulated Data Results:
# 1) The coefficient on x3 should be very close to 1.0 at all quantiles,
#    confirming that QR recovers the homogeneous effect correctly.
#
# 2) The coefficient on x2 should vary substantially across quantiles:
#    negative at q=0.25, around 1 at q=0.50, and well above 1 at q=0.75.
#    This confirms that QR captures the heterogeneous effect induced by
#    the x2-error interaction.
#
# 3) OLS would give a single coefficient of approximately 1.0 for x2
#    (the conditional mean effect).  It completely misses the fact that
#    x2 has a negative effect on the lower quantiles and a strongly
#    positive effect on the upper quantiles.
#
# This simulation is pedagogically powerful: it shows that QR is not
# just a different estimator of the same parameter -- it estimates
# DIFFERENT parameters (the quantile-specific slopes) that can tell a
# richer story about the relationship between X and Y.
# -----------------------------------------------------------------------

# ============================================================
# PART 3: COUNT DATA MODELS
# ============================================================
# -----------------------------------------------------------------------
# COUNT DATA REGRESSION: POISSON AND NEGATIVE BINOMIAL
# -----------------------------------------------------------------------
# Count data (0, 1, 2, ...) arises frequently in microeconometrics:
# doctor visits, number of children, accident counts, patent filings, etc.
# OLS is inappropriate for counts because:
#   1) The outcome is non-negative and discrete -- OLS can predict
#      negative or fractional values.
#   2) The variance of counts typically depends on the mean (violating
#      the constant-variance assumption).
#
# The Poisson regression model specifies:
#     Y_i | X_i ~ Poisson( mu_i )
#     where mu_i = exp( X_i' * beta )
#
# This implies E[Y|X] = Var[Y|X] = exp(X'beta).  The equality of mean
# and variance is called "equidispersion" and is the key restriction of
# the Poisson model.
#
# In practice, count data is almost always OVERDISPERSED: Var[Y|X] > E[Y|X].
# The Negative Binomial (NB) model relaxes equidispersion by adding a
# dispersion parameter alpha:
#     Var[Y|X] = E[Y|X] + alpha * E[Y|X]^2    (NB2 parameterization)
#
# When alpha = 0 the NB reduces to Poisson.  When alpha > 0, the
# variance exceeds the mean, accommodating overdispersion.
#
# An important practical point: even if the Poisson distributional
# assumption is wrong, the Poisson pseudo-MLE estimator is still
# consistent for beta as long as the conditional mean E[Y|X] = exp(X'beta)
# is correctly specified.  This is the "quasi-Poisson" or "Poisson PML"
# result (Gourieroux, Monfort, Trognon 1984; Wooldridge 2010).  Using
# robust (sandwich / HC1) standard errors makes inference valid even
# under overdispersion.
#
# In this Part we model doctor visits (docvis) as a function of:
#   - private : private health insurance (0/1)
#   - totchr  : total number of chronic conditions
#   - age     : age in years
#   - female  : gender indicator (1 = female)
#   - white   : race indicator (1 = white)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3: COUNT DATA (DOCTOR VISITS)")
print("=" * 60)

# -----------------------------------------------------------------------
# DATA LOADING AND SUMMARY STATISTICS
# -----------------------------------------------------------------------
# We load the doctor visits dataset and print summary statistics for the
# key variables.  Pay close attention to:
#   - docvis: the count outcome.  Check its mean, median, min, max, and
#     standard deviation.  Counts often have a mass point at zero and a
#     long right tail.
#   - Dummy variable means tell you the sample composition (e.g., what
#     fraction have private insurance, are female, etc.).
# -----------------------------------------------------------------------
df_count = pd.read_excel("Chap5_2_data.xlsx")
print(f"\nSample size: {len(df_count)}")
count_vars = ['docvis', 'private', 'totchr', 'age', 'female', 'white']
print("\n--- Summary Statistics ---")
print(df_count[count_vars].describe().round(3))

# Overdispersion check
# -----------------------------------------------------------------------
# OVERDISPERSION DIAGNOSTIC
# -----------------------------------------------------------------------
# The simplest overdispersion check is the ratio Var(Y) / Mean(Y).
# Under Poisson: this ratio should be close to 1.
# If the ratio >> 1, the data are overdispersed and:
#   (a) Poisson standard errors (without robust correction) will be too
#       small, leading to spuriously significant results.
#   (b) A Negative Binomial model may fit better.
#
# Note: This is an UNCONDITIONAL check (not conditioning on X).  The
# conditional variance could behave differently.  But it is a useful
# quick diagnostic.
# -----------------------------------------------------------------------
mean_y = df_count['docvis'].mean()
var_y = df_count['docvis'].var()
print(f"\n  Mean(docvis) = {mean_y:.3f}")
print(f"  Var(docvis)  = {var_y:.3f}")
print(f"  Var/Mean     = {var_y/mean_y:.3f} (>1 means overdispersion)")

# -----------------------------------------------------------------------
# INTERPRETATION -- Overdispersion Check:
# If Var/Mean is, say, 5 or 10, the data exhibit strong overdispersion.
# This means:
#   - The Poisson model's equidispersion assumption is violated.
#   - Poisson MLE standard errors (without robust correction) would be
#     much too small.  Using robust SE (as we do below with cov_type='HC1')
#     corrects this problem.
#   - The Negative Binomial model may provide a better distributional fit,
#     though for coefficient estimation the robust Poisson is often
#     sufficient.
# -----------------------------------------------------------------------

# --- Poisson regression ---
# -----------------------------------------------------------------------
# POISSON REGRESSION
# -----------------------------------------------------------------------
# We estimate:
#     E[docvis | X] = exp( beta_0 + beta_1*private + beta_2*totchr
#                          + beta_3*age + beta_4*female + beta_5*white )
#
# Code details:
#   - sm.GLM(..., family=sm.families.Poisson()) fits a Poisson GLM via
#     iteratively reweighted least squares (IRLS), which is equivalent to
#     maximum likelihood for the exponential family.
#   - .fit(cov_type='HC1') requests robust (Huber-White sandwich) standard
#     errors.  This is crucial because, as noted above, the Poisson
#     equidispersion assumption almost certainly fails.  The robust SE
#     remain valid as long as the mean specification is correct.
#
# Coefficient interpretation:
#   - In the Poisson model, beta_j is a SEMI-ELASTICITY with respect to
#     the conditional mean:
#         d ln E[Y|X] / dX_j = beta_j
#     A one-unit increase in X_j multiplies the expected count by
#     exp(beta_j).  For small beta_j, this is approximately a
#     100*beta_j percent change.
#   - For a dummy variable (e.g., private), the proportional change is
#     exactly exp(beta_private) - 1.
# -----------------------------------------------------------------------
print("\n--- Poisson Regression ---")
model_pois = sm.GLM(
    df_count['docvis'],
    sm.add_constant(df_count[['private', 'totchr', 'age', 'female', 'white']]),
    family=sm.families.Poisson()
).fit(cov_type='HC1')
print(model_pois.summary2().tables[1].round(4))

# -----------------------------------------------------------------------
# INTERPRETATION -- Poisson Results:
# Look at the sign, magnitude, and significance of each coefficient:
#   - private > 0: individuals with private insurance visit the doctor
#     more (possible moral hazard or access effect).
#   - totchr > 0: more chronic conditions --> more doctor visits (as
#     expected -- sicker people see doctors more).
#   - female > 0: women tend to have more doctor visits (consistent with
#     health utilization literature).
#
# Remember these are log-link coefficients.  To convert to a percentage
# effect: exp(beta) - 1.  For example, if beta_private = 0.30, then
# exp(0.30) - 1 = 0.35, meaning private insurance is associated with
# 35% more doctor visits.
# -----------------------------------------------------------------------

# --- Negative Binomial regression ---
# -----------------------------------------------------------------------
# NEGATIVE BINOMIAL REGRESSION
# -----------------------------------------------------------------------
# The NB model relaxes the Poisson equidispersion restriction by adding
# a dispersion parameter.  In statsmodels' GLM interface:
#   - sm.families.NegativeBinomial(alpha=1.0) specifies the NB2
#     parameterization with a fixed dispersion parameter alpha = 1.0.
#     (In a fully specified NB model, alpha is estimated jointly with
#     beta; the GLM interface requires it to be pre-specified.  For full
#     MLE with estimated alpha, use sm.NegativeBinomial() from the
#     discrete module instead.)
#   - cov_type='HC1' provides robust standard errors.
#
# With alpha = 1.0, the NB2 variance function is:
#     Var[Y|X] = mu + alpha * mu^2 = mu + mu^2
#
# If the true overdispersion is much higher or lower than alpha=1.0,
# the variance function will be misspecified.  However, with robust SE,
# inference on beta remains valid as long as the mean is correct
# (same quasi-likelihood logic as Poisson PML).
# -----------------------------------------------------------------------
print("\n--- Negative Binomial Regression ---")
model_nb = sm.GLM(
    df_count['docvis'],
    sm.add_constant(df_count[['private', 'totchr', 'age', 'female', 'white']]),
    family=sm.families.NegativeBinomial(alpha=1.0)
).fit(cov_type='HC1')
print(model_nb.summary2().tables[1].round(4))

# -----------------------------------------------------------------------
# INTERPRETATION -- Negative Binomial Results:
# Compare NB coefficients and standard errors to the Poisson results:
#   - Coefficients should be similar if the conditional mean specification
#     is the same (both use the log link).  Differences arise because the
#     NB likelihood weights observations differently.
#   - NB standard errors are often LARGER than robust Poisson SE, because
#     the NB explicitly models the extra-Poisson variation.
#
# If coefficients are very similar between Poisson (robust) and NB, this
# is reassuring: the mean model appears stable across different variance
# assumptions.
#
# In practice, many applied researchers report Poisson with robust SE as
# their primary specification and NB as a robustness check.
# -----------------------------------------------------------------------

# --- Marginal effects (at the mean) ---
# -----------------------------------------------------------------------
# MARGINAL EFFECTS AT THE MEAN (MEM)
# -----------------------------------------------------------------------
# In nonlinear models like Poisson, the raw coefficient beta_j is NOT
# the marginal effect dE[Y|X]/dX_j.  The marginal effect depends on
# the level of ALL covariates because:
#
#     E[Y|X] = exp(X'beta)
#     dE[Y|X]/dX_j = beta_j * exp(X'beta) = beta_j * E[Y|X]
#
# One common summary is the Marginal Effect at the Mean (MEM): evaluate
# the marginal effect at the sample mean of every covariate.
#
#     MEM_j = beta_j * exp( Xbar' * beta )
#
# Code details:
#   - X_mean = ... .mean() computes the sample mean of each covariate
#     (including the constant, whose mean is 1).
#   - mu_mean = np.exp(model_pois.params @ X_mean) computes the predicted
#     count at the mean covariate values.  The @ operator is matrix
#     multiplication (dot product here).
#   - me = beta_j * mu_mean gives the MEM for covariate j.
#
# MEM tells you: "For an individual at the sample average of all
# covariates, a one-unit increase in X_j raises the expected number
# of doctor visits by MEM_j."
#
# Alternative: Average Marginal Effect (AME) = (1/n) SUM_i [beta_j * mu_i].
# AME averages the marginal effect over all individuals, which is often
# more representative.  MEM and AME are usually similar but can differ
# when the covariate distribution is skewed.
# -----------------------------------------------------------------------
print("\n--- Marginal Effects at the Mean (Poisson) ---")
X_mean = sm.add_constant(df_count[['private', 'totchr', 'age', 'female', 'white']]).mean()
mu_mean = np.exp(model_pois.params @ X_mean)
print(f"  Predicted count at means: {mu_mean:.3f}")
for var in ['private', 'totchr', 'age', 'female', 'white']:
    me = model_pois.params[var] * mu_mean
    print(f"  dE[Y]/d({var}) = {me:.4f}")

# -----------------------------------------------------------------------
# INTERPRETATION -- Marginal Effects:
# The predicted count at the means (mu_mean) tells you the expected
# number of doctor visits for a "representative" individual whose
# covariates are all at their sample means.  Note that this hypothetical
# individual may not actually exist (e.g., if private has mean 0.45,
# the "mean individual" has 0.45 of an insurance policy, which is not
# real).
#
# The marginal effect of totchr tells you how many additional doctor
# visits one extra chronic condition generates for this representative
# individual.  If beta_totchr = 0.20 and mu_mean = 4.0, then
# ME_totchr = 0.80: one more chronic condition adds about 0.8 visits.
#
# For dummy variables (private, female, white), the MEM is an
# approximation.  A more precise approach for dummies is to compute:
#     Delta = exp(X_mean' beta with D=1) - exp(X_mean' beta with D=0)
# which is the discrete change in the expected count.  For small
# coefficients the MEM approximation is close.
# -----------------------------------------------------------------------

# --- Quantile count regression (jittered) ---
# -----------------------------------------------------------------------
# JITTERED QUANTILE REGRESSION FOR COUNT DATA
# -----------------------------------------------------------------------
# Standard quantile regression is designed for continuous outcomes.  When
# the outcome is a count (integer-valued), the conditional quantile
# function is a step function, and the standard QR estimator can behave
# erratically (multiple solutions, poor inference).
#
# The jittering approach (Machado and Santos Silva 2005) is a simple and
# effective workaround:
#
#     Y_jittered = Y + U,    where U ~ Uniform(0, 1)
#
# Adding Uniform(0,1) noise makes the outcome continuous while preserving
# the ordering (since the noise is less than 1).  We then apply standard
# QR to Y_jittered.
#
# This lets us explore heterogeneous covariate effects across the
# distribution of doctor visits -- analogous to what we did for medical
# expenditures in Part 1, but now with a count outcome.
#
# Code details:
#   - np.random.seed(10101) ensures the same jitter draws each run.
#   - df_count['docvis_jitter'] = df_count['docvis'] + np.random.uniform(...)
#     adds the uniform noise.
#   - We then run standard QuantReg on the jittered outcome at q=0.50
#     and q=0.75.
#
# Caveat: Because we are adding random noise, the exact results depend
# on the random seed.  For more robust inference, one can repeat the
# jittering many times and average the results (a "smooth jittering"
# approach).  Here we use a single draw for simplicity.
#
# We focus on q=0.50 and q=0.75 rather than q=0.25 because many
# individuals have zero doctor visits, so the lower quantiles may be
# at or near zero, making the regression uninformative at those quantiles.
# -----------------------------------------------------------------------
print("\n--- Quantile Regression on Jittered Counts ---")
np.random.seed(10101)
df_count['docvis_jitter'] = df_count['docvis'] + np.random.uniform(size=len(df_count))

X_count = sm.add_constant(df_count[['private', 'totchr', 'age', 'female', 'white']])
for q in [0.50, 0.75]:
    model_qc = QuantReg(df_count['docvis_jitter'], X_count).fit(q=q)
    print(f"\n  Quantile Count (jittered) at q={q}:")
    for var in ['private', 'totchr', 'age', 'female', 'white']:
        print(f"    {var}: {model_qc.params[var]:.4f}")

# -----------------------------------------------------------------------
# INTERPRETATION -- Jittered Quantile Count Regression:
# These coefficients tell you how each covariate shifts the tau-th
# conditional quantile of (jittered) doctor visits.
#
# Compare q=0.50 and q=0.75 coefficients:
#   - If totchr has a larger coefficient at q=0.75 than q=0.50, chronic
#     conditions generate even more doctor visits for those who already
#     visit frequently -- a "rich get richer" effect in health care use.
#   - If private insurance has a larger effect at q=0.75, insurance
#     encourages high utilizers to visit even more (intensive-margin
#     moral hazard concentrated among heavy users).
#
# Comparison with the Poisson/NB results:
#   - Poisson/NB model the conditional MEAN, summarizing the entire
#     distribution in a single coefficient per variable.
#   - Jittered QR examines specific QUANTILES, revealing whether the
#     effects are larger or smaller at different points of the
#     distribution.
#   - If QR coefficients are similar at q=0.50 and q=0.75, the
#     conditional distribution shifts uniformly and the Poisson mean
#     model captures the key variation.
#   - If they differ, there are heterogeneous effects that Poisson
#     averages over.
# -----------------------------------------------------------------------

print("\n" + "=" * 60)
print("END OF CHAPTER 5")
print("=" * 60)
