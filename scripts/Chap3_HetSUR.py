"""
=============================================================================
EE627 Microeconometrics - Chapter 3
Heteroskedasticity, SUR, and Survey Data
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

Datasets: Chap3_surdata.xlsx, Chap3_nhanes2.xlsx
This script replicates "Chap 3 dofile.do" in Python.
Designed to run on Google Colab.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib openpyxl

=============================================================================
OVERVIEW OF KEY TOPICS
=============================================================================

This script covers three core topics from Chapter 3 of the graduate
microeconometrics course. Each topic addresses a violation of the classical
linear model assumption of spherical errors (Var(u|X) = sigma^2 * I):

PART 1 -- HETEROSKEDASTICITY: FGLS AND WLS
    When the error variance is not constant across observations (i.e.,
    Var(u_i | x_i) = sigma_i^2 varies with x), OLS remains consistent
    but is no longer efficient, and the default standard errors are WRONG.
    This part demonstrates:
      a) Generating data with a known heteroskedastic DGP
      b) Comparing OLS with default vs. robust (HC1) standard errors
      c) The Breusch-Pagan test for heteroskedasticity
      d) Diagnostic scatter plots of |residuals| against regressors
      e) Feasible Generalized Least Squares (FGLS) -- modeling the variance
         function to obtain efficient estimates
      f) Weighted Least Squares (WLS) = FGLS + robust SEs -- the recommended
         estimator that is efficient AND inference-robust
      g) A side-by-side comparison of all four estimators

PART 2 -- SEEMINGLY UNRELATED REGRESSIONS (SUR)
    When we have a system of equations for the same cross-sectional units
    (e.g., drug expenditure and other medical expenditure for the same
    individuals), the error terms across equations are likely correlated.
    OLS equation-by-equation ignores this cross-equation correlation. SUR
    (Zellner, 1962) exploits this correlation to produce more efficient
    estimates -- the efficiency gain is larger when:
      (i)  the cross-equation error correlation is high, and
      (ii) the regressors differ across equations.
    This part demonstrates:
      a) Equation-by-equation OLS with robust SEs (baseline)
      b) Measuring the residual cross-equation correlation
      c) SUR estimation (using linearmodels if available)

    Stata equivalent: sureg (eq1) (eq2), corr

PART 3 -- SURVEY DATA: WEIGHTING, CLUSTERING, AND STRATIFICATION
    Most micro-datasets come from complex survey designs that involve:
      - Probability weights (pweights) -- because some groups are
        oversampled or undersampled relative to their population share
      - Stratification -- dividing the population into strata and sampling
        within each stratum to reduce sampling variance
      - Clustering (Primary Sampling Units / PSUs) -- because individuals
        within the same PSU share common unobservables, inducing intra-
        cluster correlation that inflates standard errors
    Ignoring any of these design features leads to incorrect inference.
    This part demonstrates:
      a) Unweighted OLS (ignores the survey design entirely)
      b) Weighted OLS using finalwgt as pweights (corrects point estimates)
      c) Weighted OLS with cluster-robust SEs on PSU (corrects both point
         estimates and standard errors)
      d) Comparison of population means: weighted vs. unweighted

    Stata equivalents: svyset, svy: regress, svy: mean
=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

np.set_printoptions(precision=4)

print("=" * 60)
print("CHAPTER 3: HETEROSKEDASTICITY, SUR, AND SURVEY DATA")
print("=" * 60)

# ============================================================
# PART 1: MODELING HETEROSKEDASTIC DATA (Generated Data)
# ============================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONCEPTUAL BACKGROUND
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The classical linear regression model assumes homoskedasticity:
#
#     Var(u_i | X_i) = sigma^2   for all i
#
# When this assumption fails -- i.e., the error variance depends
# on the regressors -- we have HETEROSKEDASTICITY:
#
#     Var(u_i | X_i) = sigma_i^2 = h(X_i)
#
# Consequences of heteroskedasticity for OLS:
#   1. OLS estimates (beta-hat) are still UNBIASED and CONSISTENT.
#   2. The usual OLS standard errors are BIASED (typically too small),
#      so t-tests and confidence intervals are INVALID.
#   3. OLS is no longer the BLUE (Best Linear Unbiased Estimator);
#      GLS/WLS can be more efficient.
#
# Solutions:
#   (A) Use OLS but correct the SEs -> robust (Huber-White) SEs
#       - Does NOT improve efficiency, just fixes inference
#   (B) Use FGLS/WLS -> model the variance and re-weight
#       - Potentially MORE EFFICIENT than OLS
#       - Requires a correct specification of the variance function
#   (C) Use WLS = FGLS + robust SEs -> best of both worlds (RECOMMENDED)
#       - More efficient AND robust to variance misspecification
#
# DATA GENERATING PROCESS (DGP):
#   y_i = 1 + 1*x2_i + 1*x3_i + u_i
#   u_i = sqrt(exp(-1 + 0.2*x2_i)) * e_i
#   where x2, x3, e ~ N(0, 25)  [i.e., 5 * N(0,1)]
#
# This implies:
#   Var(u_i | x_i) = exp(-1 + 0.2*x2_i) * Var(e_i)
#                   = exp(-1 + 0.2*x2_i) * 25
#
# The variance INCREASES EXPONENTIALLY with x2, while x3 plays
# no role in the heteroskedasticity. The true coefficients are
# all equal to 1 (intercept = 1, beta_x2 = 1, beta_x3 = 1).
#
# Stata equivalent of this section:
#   set seed 10101
#   quietly set obs 500
#   generate double x2 = 5*rnormal(0)
#   ... etc.
#   regress y x2 x3         // OLS with default SEs
#   regress y x2 x3, robust // OLS with robust SEs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("\n" + "=" * 60)
print("PART 1: HETEROSKEDASTICITY - FGLS AND WLS")
print("=" * 60)

# Generate data matching the Stata DGP:
# y = 1 + 1*x2 + 1*x3 + u
# u = sqrt(exp(-1 + 0.2*x2)) * e
# Var(u|x) = exp(-1 + 0.2*x2)  -- heteroskedastic!
np.random.seed(10101)
n = 500
x2 = 5 * np.random.normal(size=n)
x3 = 5 * np.random.normal(size=n)
e = 5 * np.random.normal(size=n)
u = np.sqrt(np.exp(-1 + 0.2 * x2)) * e
y = 1 + 1 * x2 + 1 * x3 + u

df_het = pd.DataFrame({'y': y, 'x2': x2, 'x3': x3})
print("\nGenerated data summary:")
print(df_het.describe().round(3))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OLS WITH DEFAULT STANDARD ERRORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# sm.add_constant() adds a column of ones to the regressor matrix,
# equivalent to Stata's automatic inclusion of _cons in regress.
#
# sm.OLS(y, X).fit() runs OLS and computes default (non-robust) SEs.
# These default SEs assume homoskedasticity. Since we KNOW the DGP
# is heteroskedastic, these SEs are WRONG -- they understate the
# true sampling variability of beta-hat for coefficients associated
# with x2 (the variable driving the heteroskedasticity).
#
# Stata equivalent: regress y x2 x3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- OLS with default SEs ---
X = sm.add_constant(df_het[['x2', 'x3']])
model_default = sm.OLS(y, X).fit()
print("\n--- OLS with Default SEs (INCORRECT for inference) ---")
print(model_default.summary2().tables[1].round(4))

# INTERPRETATION: The coefficient estimates should be close to the
# true values (1, 1, 1). However, the standard errors are computed
# under the assumption of homoskedasticity. For x2, the default SE
# is likely SMALLER than it should be, making the t-statistic too
# large and p-value too small -- overstating significance.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# OLS WITH ROBUST (HUBER-WHITE) STANDARD ERRORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cov_type='HC1' requests the Huber-White heteroskedasticity-
# consistent covariance matrix estimator with a small-sample
# degrees-of-freedom correction (N / (N-k)), which matches
# Stata's ", robust" option.
#
# HC1 (White, 1980) is:
#   Var_robust(b) = (X'X)^{-1} [sum_i uhat_i^2 * x_i x_i'] (X'X)^{-1}
#                   * N/(N-k)
#
# The coefficient estimates are IDENTICAL to default OLS (same
# beta-hat), only the standard errors change. The robust SEs are
# valid whether or not there is heteroskedasticity.
#
# Stata equivalent: regress y x2 x3, robust
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- OLS with robust SEs ---
model_robust = sm.OLS(y, X).fit(cov_type='HC1')
print("\n--- OLS with Robust SEs (HC1) ---")
print(model_robust.summary2().tables[1].round(4))

# INTERPRETATION: The coefficient estimates are exactly the same as
# above, but the SEs now correctly account for heteroskedasticity.
# Compare the SE for x2: the robust SE should be LARGER than the
# default SE (because the default SE was too optimistic). The SE
# for x3 should be similar across default and robust, because x3
# does not drive the heteroskedasticity in this DGP.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BREUSCH-PAGAN TEST FOR HETEROSKEDASTICITY
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The Breusch-Pagan (BP) test examines whether the squared OLS
# residuals are related to the regressors. Under the null:
#
#   H0: Var(u_i | x_i) = sigma^2  (homoskedasticity)
#   H1: Var(u_i | x_i) = sigma^2 * h(x_i'alpha), alpha != 0
#
# The test regresses uhat^2 on X and computes an LM statistic
# that is chi-squared distributed under H0.
#
# het_breuschpagan() returns: (LM stat, p-value, F-stat, F p-value)
#
# If p-value < 0.05, we reject H0 and conclude that heteroskedasticity
# is present. In this DGP, we KNOW there is heteroskedasticity, so
# we expect a strong rejection.
#
# Stata equivalent: estat hettest x2 x3, iid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Heteroskedasticity test ---
print("\n--- Breusch-Pagan Test ---")
bp_stat, bp_pval, _, _ = het_breuschpagan(model_default.resid, X)
print(f"  LM stat = {bp_stat:.4f}, p-value = {bp_pval:.6f}")
print(f"  Result: {'REJECT H0 (heteroskedastic)' if bp_pval < 0.05 else 'Fail to reject H0'}")

# INTERPRETATION: A very small p-value (near 0) means we strongly
# reject homoskedasticity. This confirms that at least one regressor
# is associated with non-constant variance. The formal test validates
# what the diagnostic plots (below) show visually.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HETEROSKEDASTICITY DIAGNOSTIC PLOTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A simple visual diagnostic for heteroskedasticity: plot the
# absolute value of OLS residuals |uhat| against each regressor.
#
# If the errors are homoskedastic, |uhat| should show no systematic
# pattern against any x variable -- the scatter should be a
# "horizontal band" of roughly constant width.
#
# If heteroskedasticity is present and driven by a particular
# variable, we will see |uhat| increasing (or decreasing) with
# that variable -- a "fan shape" or "funnel shape."
#
# Here we fit a polynomial curve to help visualize the trend.
# - Left panel: |uhat| vs x2 -> expect an INCREASING pattern
#   because Var(u|x) = exp(-1 + 0.2*x2), so variance rises with x2
# - Right panel: |uhat| vs x3 -> expect a FLAT pattern because
#   x3 does not enter the variance function
#
# Stata equivalent:
#   predict double uhat, resid
#   generate double absu = abs(uhat)
#   twoway (scatter absu x2) (lowess absu x2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Heteroskedasticity diagnostic plot ---
uhat = model_default.resid
absu = np.abs(uhat)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(x2, absu, alpha=0.3, s=10, color='#4472C4')
z = np.polyfit(x2, absu, 3)
p_fit = np.poly1d(z)
x2_sorted = np.sort(x2)
axes[0].plot(x2_sorted, p_fit(x2_sorted), 'r-', linewidth=2, label='Polynomial fit')
axes[0].set_xlabel('x2')
axes[0].set_ylabel('|residual|')
axes[0].set_title('|Residual| vs x2 (Increasing variance)', fontweight='bold')
axes[0].legend()

axes[1].scatter(x3, absu, alpha=0.3, s=10, color='#ED7D31')
z3 = np.polyfit(x3, absu, 3)
p_fit3 = np.poly1d(z3)
x3_sorted = np.sort(x3)
axes[1].plot(x3_sorted, p_fit3(x3_sorted), 'r-', linewidth=2, label='Polynomial fit')
axes[1].set_xlabel('x3')
axes[1].set_ylabel('|residual|')
axes[1].set_title('|Residual| vs x3 (Constant variance)', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_ch3_hetdiag.png', dpi=150)
plt.show()

# INTERPRETATION: In the left panel, you should see the spread of
# residuals INCREASING as x2 moves to the right -- this is the
# visual signature of heteroskedasticity driven by x2. The red
# polynomial fit should slope upward. In the right panel, the
# spread should be roughly constant -- no fan shape -- confirming
# that x3 does not contribute to heteroskedasticity.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FEASIBLE GENERALIZED LEAST SQUARES (FGLS)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLS is the efficient estimator when we know Var(u|X) = Omega:
#
#   beta_GLS = (X' Omega^{-1} X)^{-1} X' Omega^{-1} y
#
# In practice, we never know the true Omega, so we ESTIMATE it
# from the data in two steps -- this is FGLS:
#
# STEP 1: Run OLS and save the residuals uhat_i.
#
# STEP 2: Model the variance function. We assume a multiplicative
#   heteroskedasticity model:
#     Var(u_i | x_i) = exp(alpha_0 + alpha_1 * x2_i)
#
#   To estimate alpha_0, alpha_1, we can take logs:
#     log(uhat_i^2) = alpha_0 + alpha_1 * x2_i + noise
#   and run OLS of log(uhat^2) on a constant and x2.
#
#   This is an approximation to the NLS approach used in Stata:
#     nl (uhatsq = exp({xb: x2 one}))
#
# STEP 3: Compute the estimated variance: varu_i = exp(Zhat_i)
#
# STEP 4: Run WLS with weights w_i = 1/varu_i. Observations with
#   large estimated variance get DOWN-WEIGHTED (they carry less
#   information), while observations with small variance get
#   UP-WEIGHTED (they are more precise).
#
#   In Python: sm.WLS(y, X, weights=1/varu).fit()
#   In Stata:  regress y x2 x3 [aweight=1/varu]
#
# FGLS is asymptotically efficient if the variance function is
# correctly specified. However, the FGLS default SEs assume the
# variance model is correct, which is a strong assumption.
#
# Stata equivalent:
#   nl (uhatsq = exp({xb: x2 one})), nolog
#   predict double varu, yhat
#   regress y x2 x3 [aweight=1/varu]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- FGLS Estimation ---
print("\n--- FGLS (Feasible GLS) ---")

# Step 1: Get OLS residuals
uhat_sq = uhat ** 2

# Step 2: Model variance function: E[u^2|x] = exp(alpha0 + alpha1*x2)
# Using NLS (nonlinear least squares) via OLS on log(uhat^2)
log_uhat_sq = np.log(uhat_sq + 1e-10)  # add small constant to avoid log(0)
Z = sm.add_constant(x2)
var_model = sm.OLS(log_uhat_sq, Z).fit()
print(f"  Variance model: log(u^2) = {var_model.params[0]:.3f} + {var_model.params[1]:.3f}*x2")
print(f"  (True: -1 + 0.2*x2)")

# INTERPRETATION of the variance model: The estimated intercept should
# be close to -1 and the slope close to 0.2 (the true values in our
# DGP). If these estimates are far from the truth, the FGLS weights
# will be poorly calibrated, reducing the efficiency gains. In real
# applications, you do not know the true variance function, so you
# would try different specifications and compare results.

# Step 3: Estimated variance
varu = np.exp(var_model.fittedvalues)

# Step 4: FGLS (WLS with default SEs)
model_fgls = sm.WLS(y, X, weights=1.0 / varu).fit()
print(f"\n  FGLS estimates:")
print(model_fgls.summary2().tables[1].round(4))

# INTERPRETATION: The FGLS coefficient estimates should be close to
# (1, 1, 1). Compared to OLS, the SEs may be smaller -- especially
# for x2 -- because FGLS uses the variance structure to improve
# efficiency. However, these SEs still assume the variance model is
# correctly specified.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WLS = FGLS + ROBUST STANDARD ERRORS (RECOMMENDED APPROACH)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The WLS estimator combines the efficiency of FGLS (using weights)
# with the robustness of Huber-White SEs. This is the RECOMMENDED
# estimator because:
#
#   1. If the variance model is correctly specified, WLS is as
#      efficient as FGLS and the SEs are valid.
#   2. If the variance model is MISspecified, WLS may lose some
#      efficiency but the robust SEs remain valid.
#
# In other words, you get "insurance" against variance misspecification
# at essentially no cost.
#
# Stata equivalent: regress y x2 x3 [aweight=1/varu], robust
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 5: WLS = FGLS + robust SEs (RECOMMENDED)
model_wls = sm.WLS(y, X, weights=1.0 / varu).fit(cov_type='HC1')
print(f"\n--- WLS (FGLS + Robust SEs) - RECOMMENDED ---")
print(model_wls.summary2().tables[1].round(4))

# INTERPRETATION: The coefficient estimates are identical to FGLS
# (same weights -> same beta-hat). The SEs may differ slightly
# from FGLS because they use the sandwich formula instead of
# assuming the variance model is exact. In practice, the WLS SEs
# are the ones you should report and use for hypothesis testing.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COMPARISON TABLE: ALL FOUR ESTIMATORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This table lets you see at a glance how the choice of estimator
# and SE type affects your results.
#
# Key things to look for:
#   - All four estimators should give SIMILAR point estimates
#     (because OLS and FGLS/WLS are both consistent for beta).
#   - SE_default (column 2) for x2 should be SMALLER than SE_robust
#     (column 4), showing that default SEs are misleadingly small.
#   - SE_fgls and SE_wls should generally be SMALLER than SE_robust,
#     reflecting the efficiency gain from modeling the variance.
#   - If the variance model is well-specified, SE_fgls and SE_wls
#     will be similar. If misspecified, they may differ.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Comparison table ---
print("\n--- Comparison of Estimators ---")
comp = pd.DataFrame({
    'OLS_default': model_default.params.round(4),
    'SE_default': model_default.bse.round(4),
    'OLS_robust': model_robust.params.round(4),
    'SE_robust': model_robust.bse.round(4),
    'FGLS': model_fgls.params.round(4),
    'SE_fgls': model_fgls.bse.round(4),
    'WLS': model_wls.params.round(4),
    'SE_wls': model_wls.bse.round(4),
})
print(comp)

# INTERPRETATION: This comparison table is the punchline of Part 1.
# Students should verify that:
#   (a) Point estimates are similar across all columns (consistency).
#   (b) SE_default for x2 is misleadingly small compared to SE_robust.
#   (c) FGLS/WLS achieve smaller SEs than OLS-robust, demonstrating
#       the efficiency gain from correctly modeling heteroskedasticity.
#   (d) WLS (FGLS + robust SEs) is the recommended "belt and suspenders"
#       approach for applied work.

# ============================================================
# PART 2: SEEMINGLY UNRELATED REGRESSIONS (SUR)
# ============================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONCEPTUAL BACKGROUND
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Seemingly Unrelated Regressions (SUR), introduced by Zellner (1962),
# is a method for estimating a SYSTEM of equations that share the same
# observations but may have different regressors.
#
# Consider two equations for the same N individuals:
#
#   Eq 1: ldrugexp_i = X1_i * beta1 + u1_i   (log drug expenditure)
#   Eq 2: ltotothr_i = X2_i * beta2 + u2_i   (log other medical exp.)
#
# The equations are "seemingly unrelated" because they have different
# dependent variables and (potentially) different regressors. However,
# the errors u1_i and u2_i are likely CORRELATED because unobservable
# individual characteristics (e.g., health consciousness, risk aversion)
# affect both types of spending.
#
# If we estimate each equation separately by OLS, we ignore this
# cross-equation error correlation and lose efficiency.
#
# SUR stacks the system and applies GLS using the estimated cross-
# equation covariance matrix (Sigma):
#
#   beta_SUR = [X'(Sigma^{-1} ⊗ I)X]^{-1} X'(Sigma^{-1} ⊗ I)y
#
# where ⊗ is the Kronecker product.
#
# THE EFFICIENCY GAIN FROM SUR:
#   SUR is more efficient than equation-by-equation OLS when:
#   (1) The cross-equation error correlation |rho| is HIGH.
#   (2) The regressor sets X1 and X2 are DIFFERENT.
#
#   If X1 = X2 (identical regressors in both equations), SUR reduces
#   to OLS -- there is NO efficiency gain. Intuitively, with identical
#   regressors, each equation already uses all available information.
#
#   If rho = 0 (no cross-equation correlation), SUR also reduces to
#   OLS -- there is nothing to exploit.
#
# In this example, the equations share some regressors (age, age2,
# actlim, totchr, private) but differ in others (medicaid appears
# only in eq1; educyr appears only in eq2). This difference in
# regressor sets is what allows SUR to improve on OLS.
#
# Stata equivalent:
#   sureg (ldrugexp age age2 actlim totchr medicaid private) ///
#         (ltotothr age age2 educyr actlim totchr private), corr
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("\n" + "=" * 60)
print("PART 2: SEEMINGLY UNRELATED REGRESSIONS (SUR)")
print("=" * 60)

df_sur = pd.read_excel("Chap3_surdata.xlsx")
print("\nSUR data summary:")
sur_vars = ['ldrugexp', 'ltotothr', 'age', 'age2', 'educyr', 'actlim', 'totchr', 'medicaid', 'private']
print(df_sur[sur_vars].describe().round(3))

# Note: Python's statsmodels does not have a direct SUR command.
# We use equation-by-equation OLS as a starting point.
# For true SUR, use the systemfit or linearmodels package.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EQUATION-BY-EQUATION OLS (Baseline for SUR comparison)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Before running SUR, we estimate each equation separately by OLS
# with robust (HC1) standard errors. This serves as the baseline.
#
# Equation 1 models LOG DRUG EXPENDITURE as a function of:
#   - age, age2: age and its square (to capture nonlinear age effects)
#   - actlim: activity limitation (a measure of disability/health)
#   - totchr: total chronic conditions
#   - medicaid: Medicaid coverage indicator (1 = covered)
#   - private: private insurance indicator (1 = covered)
#
# Equation 2 models LOG TOTAL OTHER MEDICAL EXPENDITURE as a function of:
#   - age, age2: age effects
#   - educyr: years of education (appears only in eq2, not eq1)
#   - actlim, totchr: health measures
#   - private: private insurance indicator
#
# Note: We restrict to observations that are non-missing in ALL
# variables for BOTH equations. This is critical because SUR requires
# the same sample for both equations (balanced system). In Stata,
# sureg automatically handles this by using the intersection of
# non-missing observations across all equations.
#
# df.dropna() removes any row with a missing value in the selected
# columns, equivalent to Stata's: summarize ldrugexp if ldrugexp!=.
# & ltotothr!=.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Equation 1: ldrugexp = f(age, age2, actlim, totchr, medicaid, private)
# Equation 2: ltotothr = f(age, age2, educyr, actlim, totchr, private)

df_sur_clean = df_sur[['ldrugexp', 'ltotothr', 'age', 'age2', 'educyr',
                        'actlim', 'totchr', 'medicaid', 'private']].dropna()

print(f"\nUsable observations (non-missing in both equations): {len(df_sur_clean)}")

# Equation 1
X1 = sm.add_constant(df_sur_clean[['age', 'age2', 'actlim', 'totchr', 'medicaid', 'private']])
y1 = df_sur_clean['ldrugexp']
eq1 = sm.OLS(y1, X1).fit(cov_type='HC1')

# Equation 2
X2 = sm.add_constant(df_sur_clean[['age', 'age2', 'educyr', 'actlim', 'totchr', 'private']])
y2 = df_sur_clean['ltotothr']
eq2 = sm.OLS(y2, X2).fit(cov_type='HC1')

print("\n--- Equation 1: Log Drug Expenditure ---")
print(eq1.summary2().tables[1].round(4))

# INTERPRETATION of Equation 1:
# - age/age2: Together they capture a quadratic age profile. A positive
#   age coefficient and negative age2 coefficient implies an inverted-U
#   shape (spending rises with age, then levels off or falls).
# - actlim: A positive coefficient means activity limitation is
#   associated with HIGHER drug spending (sicker people spend more).
# - totchr: More chronic conditions -> higher drug spending.
# - medicaid: Medicaid coverage may increase observed drug spending
#   (because Medicaid lowers the out-of-pocket price of drugs).
# - private: Private insurance similarly may increase drug spending.

print("\n--- Equation 2: Log Other Medical Expenditure ---")
print(eq2.summary2().tables[1].round(4))

# INTERPRETATION of Equation 2:
# - The same health/insurance variables appear with similar signs.
# - educyr (education years): May be positively associated with
#   other medical spending (more educated people may seek more care),
#   or the relationship may reflect omitted income effects.
# - Comparing eq1 and eq2 coefficients is informative: if private
#   insurance has a DIFFERENT effect on drug vs. other spending,
#   that itself is an interesting economic finding.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CROSS-EQUATION RESIDUAL CORRELATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The key to SUR's efficiency gain is the cross-equation correlation
# of residuals: rho = Corr(uhat1_i, uhat2_i).
#
# np.corrcoef() computes the Pearson correlation matrix. We extract
# the off-diagonal element [0,1] which is the correlation between
# the two residual vectors.
#
# Rules of thumb:
#   - |rho| > 0.3: SUR likely provides meaningful efficiency gains
#   - |rho| < 0.1: SUR is essentially equivalent to OLS
#   - The gain also depends on how different X1 and X2 are
#
# In Stata, sureg ... , corr displays this correlation matrix
# automatically.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Cross-equation residual correlation
resid_corr = np.corrcoef(eq1.resid[:len(eq2.resid)], eq2.resid[:len(eq1.resid)])[0, 1]
print(f"\n  Residual correlation between equations: {resid_corr:.4f}")
print(f"  (If this is high, SUR gains efficiency over equation-by-equation OLS)")

# INTERPRETATION: If the residual correlation is, say, 0.3 or higher
# in absolute value, SUR can deliver noticeably smaller standard errors
# than equation-by-equation OLS. If it is close to zero, there is
# little to gain from joint estimation, and OLS is fine. The sign of
# the correlation tells us whether unobserved factors that increase
# drug spending also increase (positive) or decrease (negative) other
# medical spending.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SUR ESTIMATION (if linearmodels is available)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The linearmodels package provides a SUR estimator that implements
# the Zellner (1962) FGLS procedure. If the package is not installed,
# we fall back to the equation-by-equation OLS results above.
#
# For full SUR functionality equivalent to Stata's sureg, one can also
# use bootstrap standard errors for robustness (as the Stata do-file
# does with: bootstrap, reps(400) seed(10101) nodots: sureg ...).
#
# Key SUR tests available in Stata but harder in Python:
#   - test age age2  (joint test of age effects across both equations)
#   - test [ldrugexp]private = [ltotothr]private  (cross-equation test)
#   - constraint 1 [ldrugexp]private = [ltotothr]private (restricted SUR)
#
# These tests exploit the joint covariance matrix of beta1 and beta2,
# which is only available from SUR (not from separate OLS).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Try SUR using linearmodels if available
try:
    from linearmodels.system import SUR as SystemSUR
    print("\n--- SUR Estimation (using linearmodels) ---")
    sur_data = {
        'eq1': {'dependent': y1, 'exog': X1},
        'eq2': {'dependent': y2, 'exog': X2},
    }
    sur_model = SystemSUR(sur_data)
    sur_result = sur_model.fit(cov_type='unadjusted')
    print(sur_result.summary)
except ImportError:
    print("\n  Note: For full SUR estimation, install linearmodels:")
    print("  pip install linearmodels")

# ============================================================
# PART 3: SURVEY DATA (NHANES II)
# ============================================================
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONCEPTUAL BACKGROUND
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Most microeconomic datasets are collected through COMPLEX SURVEY
# DESIGNS, not simple random samples. The NHANES II (National Health
# and Nutrition Examination Survey, 1976-1980) is a classic example.
#
# A complex survey design has three features that affect estimation:
#
# 1. PROBABILITY WEIGHTS (pweights / sampling weights):
#    - Each observation i has a weight w_i = 1/Pr(being selected).
#    - In NHANES II, the variable "finalwgt" is the sampling weight.
#    - Some groups (e.g., minorities, elderly) are deliberately
#      OVERSAMPLED to ensure adequate sample size for subgroup analysis.
#      Their weights are SMALLER than 1 (because they had a higher
#      probability of selection than their population share implies).
#    - If we ignore weights, our estimates reflect the SAMPLE composition,
#      not the POPULATION composition. This introduces bias if the
#      sampling probability is related to the outcome variable.
#    - In Stata: [pweight=finalwgt]
#    - In Python: sm.WLS(..., weights=finalwgt)
#
# 2. STRATIFICATION:
#    - The population is divided into STRATA (e.g., geographic regions,
#      urban/rural), and independent samples are drawn within each stratum.
#    - This REDUCES sampling variance (compared to a simple random sample
#      of the same size) because each stratum is guaranteed representation.
#    - In NHANES II, the variable "strata" identifies the stratum.
#    - In Stata: svyset ..., strata(strata)
#    - Python's statsmodels does not directly support stratification;
#      we approximate by clustering within strata.
#
# 3. CLUSTERING (Primary Sampling Units / PSUs):
#    - Within each stratum, a random sample of PSUs is drawn (e.g.,
#      counties), and then all or a subsample of individuals within
#      each selected PSU are surveyed.
#    - Individuals within the same PSU share common unobservables
#      (e.g., local healthcare quality, environmental factors), creating
#      WITHIN-CLUSTER CORRELATION (intra-class correlation).
#    - This correlation means the observations are NOT independent,
#      which INFLATES standard errors relative to what you would get
#      if the same number of observations were an i.i.d. sample.
#    - In NHANES II, the variable "psu" identifies the PSU within stratum.
#    - In Stata: svyset psu [pweight=finalwgt], strata(strata)
#      then svy: regress hgb age female
#    - In Python: cov_type='cluster', cov_kwds={'groups': uniqpsu}
#
# IGNORING THE SURVEY DESIGN:
#   - Ignoring weights -> biased point estimates (if sampling is
#     related to the outcome)
#   - Ignoring clustering -> standard errors that are too SMALL
#     (overstating precision, too many false positives)
#   - Ignoring stratification -> standard errors that are too LARGE
#     (being overly conservative, though this is less problematic)
#
# The variable "hgb" is hemoglobin level (g/dL), a measure of blood
# health. We regress it on age and a female indicator.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("\n" + "=" * 60)
print("PART 3: SURVEY DATA (NHANES II)")
print("=" * 60)

df_nhanes = pd.read_excel("Chap3_nhanes2.xlsx")
df_nhanes = df_nhanes[(df_nhanes['age'] >= 21) & (df_nhanes['age'] <= 65)]

print(f"\nNHANES II data: {len(df_nhanes)} observations (age 21-65)")
print(f"\nSurvey design variables:")
for v in ['sampl', 'finalwgt', 'strata', 'psu']:
    if v in df_nhanes.columns:
        print(f"  {v}: {df_nhanes[v].describe().round(2).to_dict()}")

# INTERPRETATION of design variables:
# - sampl: sample identifier (indicates which sub-sample the person belongs to)
# - finalwgt: the sampling weight (1/probability of selection). Large
#   variation in weights suggests substantial departure from self-weighting.
# - strata: stratum identifier. Number of distinct strata = number of
#   independent sub-populations sampled.
# - psu: PSU identifier within stratum. Typically coded 1 or 2 within
#   each stratum (paired PSU design, common in NHANES).

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNWEIGHTED OLS REGRESSION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the "naive" estimator that ignores the entire survey design.
# sm.OLS(y, X).fit() with no weights and default SEs.
#
# If the survey design is "self-weighting" (all weights are equal),
# unweighted OLS gives the same results as weighted OLS. But NHANES II
# is NOT self-weighting, so unweighted estimates will differ from
# the population-representative weighted estimates.
#
# Stata equivalent: regress hgb age female
#   (without [pweight=...] or svy: prefix)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Unweighted regression ---
print("\n--- Unweighted OLS: hgb on age, female ---")
nhanes_vars = ['hgb', 'age', 'female']
df_nh = df_nhanes[nhanes_vars + ['finalwgt', 'strata', 'psu']].dropna()

X_nh = sm.add_constant(df_nh[['age', 'female']])
y_nh = df_nh['hgb']
model_unwt = sm.OLS(y_nh, X_nh).fit()
print(model_unwt.summary2().tables[1].round(4))

# INTERPRETATION: The unweighted estimates reflect the SAMPLE
# composition, which may differ from the population due to oversampling.
# The standard errors assume simple random sampling (i.i.d.), ignoring
# clustering. These SEs are likely TOO SMALL because they do not
# account for within-PSU correlation.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WEIGHTED OLS (using probability weights)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Weighted Least Squares using finalwgt as probability weights
# corrects the point estimates for the non-self-weighting design.
#
# In sm.WLS(), the `weights` parameter multiplies each observation's
# contribution to the objective function. With pweights, we set
# weights = finalwgt, so individuals with higher sampling weight
# (underrepresented in the sample) count MORE in estimation.
#
# We use cov_type='HC1' (robust SEs) because pweights alone do not
# account for clustering. The HC1 SEs are valid under heteroskedasticity
# but still assume independence -- they do not fully correct for
# within-PSU correlation.
#
# Stata equivalent:
#   regress hgb age female [pweight=finalwgt]
#   (this gives weighted estimates with heteroskedasticity-robust SEs,
#    but does NOT account for clustering or stratification)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Weighted regression (pweight) ---
print("\n--- Weighted OLS (pweight = finalwgt) ---")
model_wt_nh = sm.WLS(y_nh, X_nh, weights=df_nh['finalwgt']).fit(cov_type='HC1')
print(model_wt_nh.summary2().tables[1].round(4))

# INTERPRETATION: The coefficients may differ from the unweighted
# estimates, especially for the intercept and the female coefficient.
# Differences indicate that the sampling design overrepresents groups
# with systematically different hemoglobin levels. The SEs are
# heteroskedasticity-robust but do NOT yet account for clustering.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WEIGHTED OLS WITH CLUSTER-ROBUST STANDARD ERRORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the closest Python approximation to Stata's svy: regress.
#
# We create a unique PSU identifier by combining strata and psu:
#   uniqpsu = 2*strata + psu
# This ensures that PSU "1" in stratum 1 is different from PSU "1"
# in stratum 2 (since strata are independent sampling units).
#
# Cluster-robust SEs (also called "Liang-Zeger" or "Rogers" SEs)
# allow for arbitrary within-cluster correlation. The formula is:
#
#   Var_cluster(b) = (X'WX)^{-1} [sum_g X_g' W_g uhat_g uhat_g' W_g X_g] (X'WX)^{-1}
#
# where g indexes clusters (PSUs) and W_g is the diagonal weight matrix.
#
# These SEs will typically be LARGER than the unclustered SEs because
# they account for the positive within-PSU correlation. The "effective
# sample size" is closer to the number of PSUs than to the number of
# individuals.
#
# Note: This approach does not fully replicate Stata's svy: prefix
# because it does not incorporate the finite population correction
# or the stratification adjustment. However, for most practical
# purposes, clustering on PSU is the single most important correction.
#
# Stata equivalent:
#   generate uniqpsu = 2*strata + psu
#   regress hgb age female [pweight=finalwgt], vce(cluster uniqpsu)
#
# Or, equivalently (and preferably):
#   svyset psu [pweight=finalwgt], strata(strata)
#   svy: regress hgb age female
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Weighted with cluster-robust SEs ---
print("\n--- Weighted OLS with Cluster-Robust SEs (cluster on PSU) ---")
# Create unique PSU identifier
df_nh['uniqpsu'] = 2 * df_nh['strata'] + df_nh['psu']
model_svyreg = sm.WLS(y_nh, X_nh, weights=df_nh['finalwgt']).fit(
    cov_type='cluster', cov_kwds={'groups': df_nh['uniqpsu']}
)
print(model_svyreg.summary2().tables[1].round(4))

# INTERPRETATION: This is the most complete estimator in Part 3.
# The coefficients are the same as the weighted OLS above (same
# weights -> same beta-hat). The SEs should be LARGER than the
# unclustered weighted SEs because they account for intra-PSU
# correlation. This is the Python analog of svy: regress in Stata.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# POPULATION MEAN ESTIMATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# A basic but important task in survey analysis: estimating the
# population mean of a variable.
#
# Unweighted mean = (1/N) * sum_i y_i
#   -> reflects the SAMPLE composition
#
# Weighted mean = sum_i (w_i * y_i) / sum_i (w_i)
#   -> reflects the POPULATION composition
#   -> np.average(y, weights=w) computes this
#
# If the two means differ substantially, it means the sampling
# design is not self-weighting: some groups are over- or under-
# represented. The weighted mean is the consistent estimator of
# the population mean.
#
# Stata equivalents:
#   mean hgb                  // unweighted
#   svy: mean hgb             // weighted with full survey design
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Population mean ---
print("\n--- Population Mean of hgb ---")
hgb = df_nh['hgb']
wt = df_nh['finalwgt']
wtd_mean = np.average(hgb, weights=wt)
unwt_mean = hgb.mean()
print(f"  Unweighted mean: {unwt_mean:.4f}")
print(f"  Weighted mean:   {wtd_mean:.4f}")

# INTERPRETATION: If the weighted and unweighted means differ, it
# indicates that the sampling weights matter for this variable.
# For hemoglobin, the difference arises because groups with
# different average hemoglobin levels (e.g., by race, sex, age)
# are sampled at different rates. Always report the weighted mean
# as the population estimate.

print("\n  Note: Differences between weighted and unweighted estimates")
print("  indicate that the sampling design is not self-weighting.")
print("  Use svy: prefix in Stata, or WLS with pweights in Python.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FINAL COMPARISON: UNWEIGHTED vs WEIGHTED vs SURVEY REGRESSION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This table shows how each successive correction changes the results:
#
# (1) Unweighted: Ignores everything. May have biased coefficients
#     and incorrect (too small) SEs.
#
# (2) Weighted: Corrects point estimates for non-self-weighting
#     design. SEs are robust to heteroskedasticity but not clustering.
#
# (3) Survey (weighted + clustered): Corrects both coefficients
#     (via weights) and SEs (via clustering). This is the gold
#     standard for survey data analysis.
#
# Students should compare:
#   - How much do the coefficients change between (1) and (2)?
#     -> This tells you how much the sampling design affects
#        the point estimates.
#   - How much do the SEs change between (2) and (3)?
#     -> This tells you how much within-PSU correlation inflates
#        the standard errors. If the SEs double, the effective
#        sample size is roughly 1/4 of the nominal sample size.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- Comparison ---
print("\n--- Comparison: Unweighted vs Weighted vs Survey ---")
comp_survey = pd.DataFrame({
    'Unweighted': model_unwt.params.round(4),
    'SE_unwt': model_unwt.bse.round(4),
    'Weighted': model_wt_nh.params.round(4),
    'SE_wt': model_wt_nh.bse.round(4),
    'Survey': model_svyreg.params.round(4),
    'SE_survey': model_svyreg.bse.round(4),
})
print(comp_survey)

# INTERPRETATION: The final comparison table is the punchline of Part 3.
# Key takeaways:
#   (a) If Unweighted and Weighted coefficients differ noticeably,
#       the survey is not self-weighting and weights MUST be used
#       for population inference.
#   (b) If SE_survey >> SE_wt, there is substantial intra-cluster
#       correlation, and ignoring clustering leads to overstated
#       precision (false sense of statistical significance).
#   (c) In applied work with survey data, ALWAYS use the full survey
#       design (weights + clustering + stratification). In Stata,
#       this is svyset + svy:. In Python, the closest approximation
#       is WLS with cluster-robust SEs, as shown here.

print("\n" + "=" * 60)
print("END OF CHAPTER 3")
print("=" * 60)
