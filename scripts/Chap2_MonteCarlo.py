# MIT License
# Copyright (c) 2026 Tiraphap Fakthong
# See LICENSE file for full license text.
"""
=============================================================================
EC627 Microeconometrics - Chapter 2
Monte Carlo Simulation Methods
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

This script replicates the Stata do-file "chap 2.do" in Python.
Designed to run on Google Colab. No external dataset needed.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib

=============================================================================
OVERVIEW OF KEY TOPICS COVERED IN THIS CHAPTER
=============================================================================

This chapter introduces Monte Carlo simulation as a fundamental tool for
understanding the finite-sample (small-sample) properties of econometric
estimators. Unlike asymptotic theory (which tells us what happens as N -> infinity),
Monte Carlo simulation lets us study how estimators actually behave with
realistic sample sizes.

Topics covered:

1. PSEUDO-RANDOM NUMBER GENERATORS (Section 1)
   - How computers generate "random" numbers (they are deterministic but appear random)
   - Setting seeds for reproducibility (critical for replicable research)
   - Verifying that draws are i.i.d. (independent and identically distributed)
   - Stata-to-Python equivalence: set seed -> np.random.seed; runiform() -> np.random.uniform()

2. GENERATING FROM VARIOUS DISTRIBUTIONS (Section 2)
   - Drawing from t, chi-squared, and F distributions
   - Constructing the F distribution from chi-squared random variables
   - Understanding the relationship between these distributions and their parameters

3. DISCRETE RANDOM VARIABLES (Section 3)
   - Poisson distribution for count data
   - Negative Binomial as a Poisson-Gamma mixture (overdispersion)
   - Why the variance-to-mean ratio matters for count data models

4. CENTRAL LIMIT THEOREM (CLT) DEMONSTRATION (Section 4)
   - The CLT says: sqrt(N)*(xbar - mu) -> N(0, sigma^2) as N -> infinity
   - Even starting from a Uniform(0,1) distribution (which looks nothing like
     a normal), the distribution of the sample mean becomes approximately normal
   - We verify this numerically with 10,000 replications

5. FINITE-SAMPLE PROPERTIES OF OLS (Section 5)
   - Unbiasedness: E[beta_hat] = beta (the true value)
   - Standard errors track actual sampling variability: E[SE] ~ SD(beta_hat)
   - t-test has correct size: rejection rate under H0 should be ~5%
   - Uses non-normal (chi-squared) errors to show these properties hold
     regardless of error distribution (by the CLT, as long as N is large enough)

6. MEASUREMENT ERROR / ERRORS-IN-VARIABLES (Section 6)
   - When we observe x = x* + v instead of the true x*, OLS is biased
   - Attenuation bias: the coefficient is biased TOWARD ZERO
   - The bias formula: plim(beta_hat) = beta * Var(x*) / [Var(x*) + Var(v)]
   - This is ALWAYS between 0 and beta, so the coefficient is always "attenuated"

7. ENDOGENOUS REGRESSOR (Section 7)
   - When Cov(x, u) != 0, OLS is biased AND inconsistent
   - Unlike measurement error, the bias does NOT vanish with larger samples
   - The direction and magnitude of bias depend on Cov(x, u) / Var(x)
   - Solution: Instrumental Variables (covered in Chapter 4)

=============================================================================
WHY MONTE CARLO SIMULATION MATTERS IN ECONOMETRICS
=============================================================================

In econometric theory, we derive properties of estimators using two frameworks:
  (a) Finite-sample (exact) properties: E[beta_hat] = beta (unbiasedness)
  (b) Asymptotic properties: plim(beta_hat) = beta (consistency)

Analytical derivations can be complex or intractable for many estimators.
Monte Carlo simulation provides a numerical alternative:
  1. Specify a known Data Generating Process (DGP) with known parameters
  2. Generate many artificial datasets from this DGP
  3. Apply the estimator to each dataset
  4. Examine the distribution of estimates across replications

If the estimator is unbiased, the average estimate should equal the true parameter.
If the estimator is consistent, this average should converge to the true value
as sample size N increases.

=============================================================================
STATA-TO-PYTHON QUICK REFERENCE
=============================================================================
  Stata                           Python (NumPy / SciPy / statsmodels)
  -----                           ------------------------------------
  set seed 10101                  np.random.seed(10101)
  runiform()                      np.random.uniform()
  rnormal()                       np.random.normal()
  rnormal(mu, sigma)              np.random.normal(mu, sigma)
  rchi2(df)                       np.random.chisquare(df) or stats.chi2.rvs(df)
  rt(df)                          stats.t.rvs(df)
  rpoisson(mu)                    np.random.poisson(lam=mu)
  rgamma(a, b)                    np.random.gamma(shape=a, scale=b)
  regress y x                     sm.OLS(y, sm.add_constant(x)).fit()
  _b[x]                           model.params[1]
  _se[x]                          model.bse[1]
  summarize x                     pd.Series(x).describe()
  simulate ... , reps(N): prog    [prog() for _ in range(N)]  (loop-based)
  histogram x                     plt.hist(x)
  kdensity x                      x.plot.kde()  or  stats.gaussian_kde(x)
=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
#
# We import the core libraries needed throughout this script:
#
#   pandas      - DataFrames for organized tabular output (like Stata's data editor)
#   numpy       - Numerical arrays and random number generation (like Stata's mata)
#   matplotlib  - Plotting library (like Stata's graph commands)
#   scipy.stats - Statistical distributions (like Stata's r-class random functions)
#   statsmodels - Regression and econometric models (like Stata's regress, etc.)
#
# Stata equivalent of this setup block:
#   set more off      -> not needed in Python (output streams continuously)
#   version 11        -> not needed (Python packages handle versioning)
#   clear all         -> not needed (each script starts fresh)
#   set memory 10m    -> not needed (Python manages memory dynamically)
#   set scheme s1mono -> matplotlib has its own styling; we use defaults here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Set NumPy to display 4 decimal places for cleaner output.
# This is analogous to Stata's "format %9.4f" but applied globally to NumPy arrays.
np.set_printoptions(precision=4)

print("=" * 60)
print("CHAPTER 2: MONTE CARLO SIMULATION METHODS")
print("=" * 60)

# ============================================================
# 1. PSEUDO-RANDOM NUMBER GENERATORS
# ============================================================
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# Computers cannot generate truly random numbers. Instead, they use
# deterministic algorithms called Pseudo-Random Number Generators (PRNGs)
# that produce sequences of numbers that *appear* random. These sequences
# satisfy statistical tests for uniformity and independence.
#
# A "seed" initializes the PRNG to a specific starting state. Setting the
# same seed always produces the same sequence of "random" numbers. This is
# essential for REPRODUCIBILITY in research -- other researchers can verify
# your Monte Carlo results by using the same seed.
#
# The fundamental building block is the Uniform(0,1) distribution. All other
# random variables (normal, t, chi-squared, etc.) are derived from uniform
# draws through mathematical transformations (e.g., the inverse CDF method
# or the Box-Muller transform for normals).
#
# KEY PROPERTIES OF Uniform(0,1):
#   - E[X]   = 0.5                  (the mean)
#   - Var(X) = 1/12 ~ 0.0833       (the variance)
#   - SD(X)  = 1/sqrt(12) ~ 0.2887 (the standard deviation)
#   - Min    = 0, Max = 1
#
# WHAT THE CODE DOES:
# -------------------
# First, we generate a single uniform draw, then 10,000 draws. By the Law
# of Large Numbers (LLN), the sample mean of 10,000 i.i.d. draws should be
# very close to the population mean (0.5). We then check independence by
# computing autocorrelations at lags 1, 2, and 3.
#
# STATA EQUIVALENTS:
#   set seed 10101           -> np.random.seed(10101)
#   scalar u = runiform()    -> u = np.random.uniform()
#   set obs 10000            -> (specified via size= parameter)
#   generate x = runiform()  -> x = np.random.uniform(size=10000)
#   pwcorr x L1.x L2.x L3.x -> manual autocorrelation via np.corrcoef
#
print("\n--- 1. Pseudo-Random Number Generators ---")

# Stata: set seed 10101; scalar u = runiform(); display u
np.random.seed(10101)
u = np.random.uniform()
print(f"Single uniform draw: {u:.6f}")

# Stata: set obs 10000; generate x = runiform()
np.random.seed(10101)
x = np.random.uniform(size=10000)
print(f"\n10,000 uniform draws:")
print(f"  Mean: {x.mean():.4f} (expected: 0.5)")
print(f"  Std:  {x.std():.4f} (expected: {1/np.sqrt(12):.4f})")
print(f"  Min:  {x.min():.4f}")
print(f"  Max:  {x.max():.4f}")

# INTERPRETATION OF UNIFORM DRAW RESULTS:
# ----------------------------------------
# With 10,000 draws, the sample mean should be very close to 0.5 (within about
# 0.005 by the LLN). The standard deviation should be close to 1/sqrt(12) = 0.2887.
# If these numbers are far off, something is wrong with the PRNG.

# Check independence: autocorrelations
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# For our simulation results to be valid, the draws must be INDEPENDENT.
# We test this by computing autocorrelations: the correlation between x_t
# and x_{t-k} for lags k = 1, 2, 3. For truly independent draws, all
# autocorrelations should be approximately zero.
#
# In Stata, you would use:
#   generate t = _n
#   tsset t
#   pwcorr x L1.x L2.x L3.x
# Here in Python, we manually compute the correlation between the series
# and its lagged version using np.corrcoef.
#
# WHAT TO LOOK FOR:
# Autocorrelations should be very close to 0 (typically |r| < 0.02 for
# N = 10,000). Large autocorrelations would indicate the PRNG is producing
# dependent sequences, which would invalidate Monte Carlo results.
print("\n--- Autocorrelations (independence check) ---")
for lag in [1, 2, 3]:
    r = np.corrcoef(x[lag:], x[:-lag])[0, 1]
    print(f"  Lag {lag}: r = {r:.6f} (should be ~0)")

# INTERPRETATION OF AUTOCORRELATION RESULTS:
# -------------------------------------------
# All autocorrelations should be essentially zero (e.g., |r| < 0.02).
# This confirms that our PRNG produces (approximately) independent draws.
# Combined with the summary statistics confirming the correct distribution,
# we have verified that our draws are i.i.d. Uniform(0,1) -- the foundation
# for all subsequent simulations in this chapter.

# ============================================================
# 2. GENERATING FROM VARIOUS DISTRIBUTIONS
# ============================================================
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# In econometrics, we frequently need draws from distributions other than
# the uniform. The three most important are:
#
# (a) Student's t-distribution with df degrees of freedom: t(df)
#     - Arises when we estimate the mean of a normal population with unknown
#       variance. The t-statistic follows this distribution under H0.
#     - As df -> infinity, t(df) -> N(0,1) (standard normal).
#     - For df=10: E[X] = 0, Var(X) = df/(df-2) = 10/8 = 1.25
#
# (b) Chi-squared distribution with df degrees of freedom: chi2(df)
#     - Sum of df independent squared standard normals: X = Z1^2 + ... + Zdf^2
#     - Arises in tests of variance and goodness-of-fit tests.
#     - E[X] = df = 10, Var(X) = 2*df = 20
#
# (c) F-distribution with (df1, df2) degrees of freedom: F(df1, df2)
#     - Ratio of two independent chi-squared r.v.s, each divided by their df:
#       F = (chi2(df1)/df1) / (chi2(df2)/df2)
#     - Arises in F-tests for joint significance of multiple coefficients.
#     - E[X] = df2/(df2-2) = 5/3 = 1.667 (for df2 > 2)
#
# WHAT THE CODE DOES:
# -------------------
# We draw 2,000 observations from each distribution and verify that their
# sample statistics match the theoretical values listed above. We also
# construct the F(10,5) distribution manually from two chi-squared draws
# to illustrate the definitional relationship.
#
# STATA EQUIVALENTS:
#   generate xt = rt(10)         -> stats.t.rvs(df=10, size=n)
#   generate xc = rchi2(10)      -> stats.chi2.rvs(df=10, size=n)
#   generate xfn = rchi2(10)/10  -> stats.chi2.rvs(df=10, size=n) / 10
#   generate xfd = rchi2(5)/5    -> stats.chi2.rvs(df=5, size=n) / 5
#   generate xf = xfn/xfd        -> xfn / xfd
#
print("\n--- 2. Multiple Distributions ---")

np.random.seed(10101)
n = 2000

# Stata equivalents
xt = stats.t.rvs(df=10, size=n)           # t(10)
xc = stats.chi2.rvs(df=10, size=n)        # Chi-squared(10)
xfn = stats.chi2.rvs(df=10, size=n) / 10  # Numerator of F(10,5)
xfd = stats.chi2.rvs(df=5, size=n) / 5    # Denominator of F(10,5)
xf = xfn / xfd                             # F(10,5)

results = pd.DataFrame({
    't(10)': xt, 'chi2(10)': xc, 'F(10,5)': xf
})
print(results.describe().round(3))

# INTERPRETATION OF DISTRIBUTION RESULTS:
# ----------------------------------------
# Check the sample statistics against theoretical values:
#
#   t(10):    E[X] = 0,      SD(X) = sqrt(10/8) = 1.118
#   chi2(10): E[X] = 10,     SD(X) = sqrt(20)   = 4.472
#   F(10,5):  E[X] = 5/3 = 1.667, but high variance and right-skewed
#
# The sample mean and standard deviation should be close to these theoretical
# values. Small deviations are expected due to sampling variability. With
# n=2,000, we expect the sample mean to be within roughly 2*SD/sqrt(2000)
# of the true mean.

# Histogram + KDE (Kernel Density Estimate)
#
# We plot histograms overlaid with kernel density estimates to visually
# verify the shape of each distribution. The t(10) should look nearly
# normal but with slightly heavier tails. The chi2(10) should be right-skewed.
# The F(10,5) should also be right-skewed with a long right tail.
#
# Stata equivalent:
#   twoway (histogram xc, width(1)) (kdensity xc, lwidth(thick))
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, color in zip(axes, results.columns, ['#4472C4', '#ED7D31', '#70AD47']):
    results[col].hist(bins=50, ax=ax, density=True, alpha=0.7, color=color, edgecolor='white')
    results[col].plot.kde(ax=ax, linewidth=2, color='black')
    ax.set_title(col, fontweight='bold')
plt.suptitle('Random Draws from t, Chi-squared, and F distributions', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_ch2_distributions.png', dpi=150)
plt.show()

# ============================================================
# 3. DISCRETE RANDOM VARIABLES
# ============================================================
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# Count data (non-negative integers: 0, 1, 2, ...) are extremely common
# in microeconometrics: number of doctor visits, number of children, number
# of patents filed, etc. Two key distributions for count data are:
#
# (a) Poisson distribution with parameter mu (lambda):
#     - P(Y = k) = exp(-mu) * mu^k / k!
#     - E[Y] = mu,  Var(Y) = mu  (mean equals variance: "equidispersion")
#     - This equidispersion restriction is very strong and often violated
#       in real data.
#
# (b) Negative Binomial distribution:
#     - Arises as a Poisson-Gamma MIXTURE: Y|lambda ~ Poisson(lambda),
#       where lambda itself is random: lambda = mu * v, v ~ Gamma(1,1)
#     - This adds "unobserved heterogeneity" -- individuals with the same
#       observed characteristics have different underlying rates.
#     - E[Y] = mu (same as Poisson), but Var(Y) > mu (OVERDISPERSION)
#     - Var(Y)/E[Y] > 1 is the hallmark of overdispersion.
#
# WHY THIS MATTERS:
# -----------------
# If the data are overdispersed (Var > Mean) but you estimate a Poisson
# model, your standard errors will be TOO SMALL, leading to spuriously
# significant results. The Negative Binomial model accommodates overdispersion.
# This is covered in detail in Chapter 5.
#
# WHAT THE CODE DOES:
# -------------------
# We generate three types of count data:
#   1. Poisson(5) -- homogeneous, equidispersed
#   2. Poisson(xb) where xb ~ Uniform(4,6) -- heterogeneous means, but
#      conditionally equidispersed
#   3. Poisson(xb * xg) where xg ~ Gamma(1,1) -- this is a Poisson-Gamma
#      mixture, producing a Negative Binomial with overdispersion
#
# STATA EQUIVALENTS:
#   generate xp  = rpoisson(5)    -> np.random.poisson(lam=5, size=n)
#   generate xb  = 4 + 2*runiform() -> 4 + 2*np.random.uniform(size=n)
#   generate xg  = rgamma(1,1)    -> np.random.gamma(shape=1, scale=1, size=n)
#   generate xp1 = rpoisson(xb)   -> np.random.poisson(lam=xb)
#   generate xp2 = rpoisson(xbh)  -> np.random.poisson(lam=xbh)
#
print("\n--- 3. Discrete Random Variables ---")

np.random.seed(10101)

# Poisson draws with constant mean mu=5
# For Poisson(5): E[Y] = 5, Var(Y) = 5, so Var/Mean = 1.0 (equidispersed)
xp = np.random.poisson(lam=5, size=n)

# Negative Binomial (Poisson-Gamma mixture)
# Step 1: Generate heterogeneous means. xb ~ Uniform(4,6) so E[xb] = 5.
xb = 4 + 2 * np.random.uniform(size=n)          # mu varies between 4 and 6

# Step 2: Generate gamma-distributed multiplicative heterogeneity.
# xg ~ Gamma(shape=1, scale=1), so E[xg] = 1 and Var(xg) = 1.
# This means xbh = xb * xg has the same mean as xb (since E[xg]=1),
# but much more variability.
xg = np.random.gamma(shape=1, scale=1, size=n)   # gamma heterogeneity
xbh = xb * xg                                     # multiplicative heterogeneity

# Step 3: Draw from Poisson with the heterogeneous means.
# xp1 uses xb directly (varying mean but no gamma heterogeneity).
# xp2 uses xbh = xb * xg (Poisson-Gamma = Negative Binomial).
xp1 = np.random.poisson(lam=xb)                  # Poisson with varying mean
xp2 = np.random.poisson(lam=xbh)                 # NegBin (Poisson-Gamma)

disc_df = pd.DataFrame({'Poisson(5)': xp, 'Poisson(xb)': xp1, 'NB(xb)': xp2})
print(disc_df.describe().round(3))
print(f"\nOverdispersion check for NB(xb):")
print(f"  Var/Mean ratio = {xp2.var()/xp2.mean():.2f} (>1 means overdispersion)")

# INTERPRETATION OF DISCRETE DISTRIBUTION RESULTS:
# --------------------------------------------------
# Compare the three columns:
#
# 1. Poisson(5):   Mean ~ 5, Var ~ 5,   Var/Mean ~ 1.0  (equidispersed)
# 2. Poisson(xb):  Mean ~ 5, Var ~ 5+,  Var/Mean ~ 1.0+ (slightly overdispersed
#                   because the mean varies, adding extra variance, but the
#                   effect is modest since xb only ranges from 4 to 6)
# 3. NB(xb):       Mean ~ 5, Var >> 5,  Var/Mean >> 1   (strongly overdispersed)
#
# The key insight: The gamma heterogeneity (xg) dramatically inflates the
# variance while keeping the mean roughly the same. This is why we need the
# Negative Binomial model when real count data show Var(Y) >> E[Y].

# ============================================================
# 4. CENTRAL LIMIT THEOREM (CLT)
# ============================================================
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# The Central Limit Theorem is arguably the most important result in
# statistics and econometrics. It states:
#
#   If X1, X2, ..., Xn are i.i.d. with E[Xi] = mu and Var(Xi) = sigma^2,
#   then as n -> infinity:
#
#       sqrt(n) * (Xbar - mu) / sigma  -->  N(0, 1)
#
#   Equivalently: Xbar ~ approximately N(mu, sigma^2/n) for large n.
#
# CRITICAL POINTS FOR THE EXAM (from the Stata do-file):
# - The CLT requires the draws to be independent? Not necessarily --
#   weaker forms of the CLT exist for dependent data (e.g., CLT for
#   mixing processes). But i.i.d. is the simplest sufficient condition.
#
# WHY THIS MATTERS FOR ECONOMETRICS:
# -----------------------------------
# OLS estimators are (weighted) averages of random variables. Even when the
# error term u is NOT normally distributed, the CLT ensures that beta_hat
# is approximately normal in large samples. This justifies using t-tests
# and F-tests even with non-normal data.
#
# WHAT THE CODE DOES:
# -------------------
# 1. Draw ONE sample of n=30 from Uniform(0,1) and show its histogram
#    (it looks flat/uniform, NOT normal).
# 2. Repeat 10,000 times: draw a sample of n=30, compute the sample mean.
# 3. Show that the distribution of the 10,000 sample means IS approximately
#    normal, even though each individual sample comes from a uniform distribution!
#
# THEORETICAL VALUES FOR Xbar:
#   - E[Xbar] = E[X] = 0.5
#   - SD(Xbar) = SD(X)/sqrt(n) = (1/sqrt(12))/sqrt(30) = 1/sqrt(360) = 0.0527
#
# STATA EQUIVALENTS:
#   program onesample, rclass       -> Python function or list comprehension
#       drop _all
#       quietly set obs 30
#       generate x = runiform()
#       summarize x
#       return scalar meanforonesample = r(mean)
#   end
#   simulate xbar = r(meanforonesample), seed(10101) reps(10000): onesample
#
# In Stata, the `simulate` command automates the loop. In Python, we use
# a list comprehension: [f() for _ in range(10000)].
#
print("\n" + "=" * 60)
print("4. CENTRAL LIMIT THEOREM")
print("=" * 60)

# Stata: Draw 1 sample of size 30 from uniform
np.random.seed(10101)
one_sample = np.random.uniform(size=30)
print(f"\nOne sample (n=30) from Uniform(0,1):")
print(f"  Mean = {one_sample.mean():.4f}")
print(f"  Std  = {one_sample.std():.4f}")

# INTERPRETATION:
# This single sample has a mean somewhere around 0.5 and a std around 0.29.
# The histogram of this ONE sample will look roughly flat (uniform), NOT normal.
# The CLT is NOT about the shape of ONE sample -- it's about the distribution
# of the SAMPLE MEAN across MANY samples.

# Stata: simulate 10,000 sample means
# This is the core Monte Carlo loop. We:
#   1. Draw 30 uniform random numbers (one "sample")
#   2. Compute the sample mean (one "xbar")
#   3. Store it
#   4. Repeat 10,000 times
# The resulting collection of 10,000 xbar values shows the SAMPLING DISTRIBUTION
# of the sample mean.
np.random.seed(10101)
n_reps = 10000
sample_size = 30
xbars = np.array([np.random.uniform(size=sample_size).mean() for _ in range(n_reps)])

print(f"\nDistribution of 10,000 sample means:")
print(f"  Mean of xbar:    {xbars.mean():.4f} (expected: 0.5)")
print(f"  Std of xbar:     {xbars.std():.4f} (expected: {1/np.sqrt(12*30):.4f})")
print(f"  Skewness:        {stats.skew(xbars):.4f} (expected: ~0)")
print(f"  Kurtosis (exc):  {stats.kurtosis(xbars):.4f} (expected: ~0)")

# INTERPRETATION OF CLT RESULTS:
# --------------------------------
# - Mean of xbar ~ 0.5: The sample mean is an UNBIASED estimator of E[X].
#   By the LLN, the average of 10,000 sample means converges to the true mean.
#
# - Std of xbar ~ 0.0527: This is the STANDARD ERROR of the sample mean.
#   It equals sigma/sqrt(n) = (1/sqrt(12))/sqrt(30) = 0.0527. The standard
#   error shrinks at rate 1/sqrt(n), so quadrupling n halves the SE.
#
# - Skewness ~ 0: A normal distribution has zero skewness (symmetric).
#   The fact that skewness is near zero confirms approximate normality.
#
# - Excess Kurtosis ~ 0: A normal distribution has excess kurtosis of zero
#   (kurtosis of 3, minus 3 = 0). Values near zero confirm normality.
#   (Positive excess kurtosis = heavier tails than normal = leptokurtic)

# Plot CLT demonstration
# Left panel:  Histogram of ONE sample (looks uniform/flat)
# Right panel: Histogram of 10,000 sample means (looks normal!)
#              Overlaid with the theoretical N(0.5, 0.0527^2) density in red.
#
# Stata equivalent:
#   histogram x, width(0.1) xtitle("x from one sample")
#   histogram xbar, normal xtitle("xbar from many samples")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(one_sample, bins=10, edgecolor='white', alpha=0.8, color='#4472C4')
axes[0].set_title('One Sample (n=30) from Uniform(0,1)', fontweight='bold')
axes[0].set_xlabel('x')

axes[1].hist(xbars, bins=50, density=True, edgecolor='white', alpha=0.8, color='#70AD47')
xg = np.linspace(xbars.min(), xbars.max(), 200)
mu, sigma = 0.5, 1 / np.sqrt(12 * 30)
axes[1].plot(xg, stats.norm.pdf(xg, mu, sigma), 'r-', linewidth=2, label=f'N({mu}, {sigma:.4f}²)')
axes[1].set_title('Distribution of 10,000 Sample Means (CLT)', fontweight='bold')
axes[1].set_xlabel(r'$\bar{x}$')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_ch2_clt.png', dpi=150)
plt.show()

# INTERPRETATION OF CLT PLOTS:
# ------------------------------
# The visual contrast is striking:
# - Left panel: The raw data from ONE sample of 30 uniform draws. It looks
#   roughly flat (uniform), definitely NOT normal.
# - Right panel: The distribution of 10,000 sample means is beautifully
#   bell-shaped and matches the red theoretical normal curve almost perfectly.
#
# This is the CLT in action: even though individual observations are uniform,
# the sample mean is approximately normal. This is WHY we can use t-tests
# and confidence intervals based on normality -- they apply to the SAMPLING
# DISTRIBUTION of the estimator, not to the raw data.

# ============================================================
# 5. FINITE-SAMPLE PROPERTIES OF OLS
# ============================================================
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# This is one of the most important simulations in the chapter. We verify
# three key finite-sample properties of the OLS estimator:
#
# (A) UNBIASEDNESS: E[beta_hat] = beta_true
#     Under the Gauss-Markov assumptions (linear model, exogenous regressors,
#     homoskedastic errors), OLS is unbiased: the average of beta_hat across
#     many simulations should equal the true beta.
#
# (B) STANDARD ERRORS ARE ACCURATE: E[SE(beta_hat)] ~ SD(beta_hat)
#     The standard error formula should correctly predict the actual
#     standard deviation of the sampling distribution. If SE is too small,
#     we get too many false rejections; if too large, tests lack power.
#
# (C) CORRECT TEST SIZE: Under H0, rejection rate ~ nominal level (5%)
#     If we test H0: beta = 2 (the true value) at the 5% level, we should
#     reject approximately 5% of the time. This is called "correct size."
#     A test that rejects more than 5% is "over-sized" (too many Type I errors).
#     A test that rejects less than 5% is "under-sized" (conservative).
#
# THE DATA GENERATING PROCESS (DGP):
# -----------------------------------
#   y_i = 1 + 2*x_i + u_i
#
#   where:
#     - x_i ~ chi-squared(1):   non-negative, right-skewed regressor
#     - u_i ~ chi-squared(1)-1: non-normal, heteroskedastic-looking error
#       (demeaned so E[u] = 0, since E[chi2(1)] = 1, subtracting 1 gives mean 0)
#     - True parameters: beta_0 = 1 (intercept), beta_1 = 2 (slope)
#
# IMPORTANT DESIGN CHOICE: We deliberately use NON-NORMAL errors (chi-squared
# minus 1) to show that OLS unbiasedness does NOT require normality. The
# Gauss-Markov theorem only requires E[u|X] = 0, not normality.
# The t-test's validity relies on the CLT (for n=150, this is adequate).
#
# WHAT THE CODE DOES:
# -------------------
# For each of 1,000 simulations:
#   1. Generate a fresh dataset of 150 observations from the DGP
#   2. Run OLS: regress y on x (with a constant)
#   3. Store: beta_hat (the slope estimate), SE (standard error),
#      and whether |t| > critical value (rejection of H0: beta=2)
# Then summarize across all 1,000 simulations.
#
# STATA EQUIVALENTS:
#   The Stata do-file defines a program `chi2data` that generates one dataset
#   and returns scalars, then uses `simulate` to call it 1,000 times:
#
#   program chi2data, rclass
#       drop _all
#       set obs $numobs
#       generate double x = rchi2(1)
#       generate u = rchi2(1)-1
#       generate y = 1 + 2*x + u
#       regress y x
#       return scalar b2 = _b[x]           -> model.params[1]
#       return scalar se2 = _se[x]         -> model.bse[1]
#       return scalar t2 = (_b[x]-2)/_se[x]
#       return scalar r2 = abs(return(t2)) > invttail($numobs-2,.025)
#   end
#   simulate b2f=r(b2) ... , reps($numsims): chi2data
#
#   In Python, we replicate this with a simple for-loop.
#
print("\n" + "=" * 60)
print("5. SIMULATION: FINITE-SAMPLE PROPERTIES OF OLS")
print("=" * 60)

# DGP: y = 1 + 2*x + u, where u = chi2(1) - 1 (demeaned)
numobs = 150
numsims = 1000

np.random.seed(10101)

b2_list = []
se2_list = []
reject_list = []

for sim in range(numsims):
    # Step 1: Generate regressors from chi-squared(1).
    # chi2(1) has mean=1, variance=2, and is right-skewed.
    x_sim = np.random.chisquare(df=1, size=numobs)

    # Step 2: Generate errors from demeaned chi-squared(1).
    # chi2(1) has mean 1, so subtracting 1 gives E[u] = 0.
    # Var(u) = Var(chi2(1)) = 2*1 = 2.
    # Note: u is NOT normal -- it is skewed. This is intentional.
    u_sim = np.random.chisquare(df=1, size=numobs) - 1  # demeaned

    # Step 3: Generate y from the true DGP.
    # True beta_0 = 1, true beta_1 = 2.
    y_sim = 1 + 2 * x_sim + u_sim

    # Step 4: Run OLS regression of y on x (with constant added by sm.add_constant).
    # sm.add_constant(x_sim) creates an [N x 2] matrix: [1, x].
    # sm.OLS(y, X).fit() is equivalent to Stata's "regress y x".
    X_sim = sm.add_constant(x_sim)
    model_sim = sm.OLS(y_sim, X_sim).fit()

    # Step 5: Extract the slope coefficient (beta_1 hat) and its standard error.
    # model.params[0] = intercept, model.params[1] = slope on x.
    # model.bse[0] = SE of intercept, model.bse[1] = SE of slope.
    b2 = model_sim.params[1]
    se2 = model_sim.bse[1]

    # Step 6: Compute the t-statistic for testing H0: beta_1 = 2.
    # t = (beta_hat - beta_0) / SE(beta_hat)
    # Under H0 (beta_1 = 2), this should follow a t(N-2) = t(148) distribution.
    t2 = (b2 - 2) / se2  # t-statistic under H0: beta=2

    # Step 7: Determine if we reject H0 at the 5% significance level.
    # We reject if |t| > t_{0.975, N-2} (two-tailed test).
    # stats.t.ppf(0.975, 148) gives the 97.5th percentile of t(148) ~ 1.976.
    # Stata equivalent: invttail($numobs-2, .025)
    reject = abs(t2) > stats.t.ppf(0.975, numobs - 2)

    b2_list.append(b2)
    se2_list.append(se2)
    reject_list.append(reject)

b2_arr = np.array(b2_list)
se2_arr = np.array(se2_list)
reject_arr = np.array(reject_list)

print(f"\nResults from {numsims} simulations (N={numobs}):")
print(f"  True beta:          2.000")
print(f"  E[beta_hat]:        {b2_arr.mean():.4f}  (unbiased if ~2.0)")
print(f"  SD(beta_hat):       {b2_arr.std():.4f}")
print(f"  E[SE(beta_hat)]:    {se2_arr.mean():.4f}  (should be ~SD)")
print(f"  Rejection rate (5%): {reject_arr.mean():.4f}  (should be ~0.05)")

# INTERPRETATION OF OLS SIMULATION RESULTS:
# -------------------------------------------
#
# (A) UNBIASEDNESS CHECK: E[beta_hat] ~ 2.000
#     - If this number is close to 2.0, OLS is unbiased (as theory predicts).
#     - Small deviations from 2.0 are expected due to simulation error.
#       The simulation standard error is SD(beta_hat)/sqrt(numsims).
#
# (B) STANDARD ERROR CHECK: E[SE(beta_hat)] ~ SD(beta_hat)
#     - SD(beta_hat) is the TRUE sampling standard deviation (computed from
#       the 1,000 estimates).
#     - E[SE(beta_hat)] is the AVERAGE of the estimated standard errors.
#     - If these two numbers are close, the SE formula is working correctly.
#     - NOTE: We use default (homoskedastic) standard errors here. Since the
#       true DGP has heteroskedastic errors (Var(u|x) depends on x because
#       u ~ chi2(1)-1 which has variance 2 regardless of x -- actually this is
#       HOMOskedastic since Var(u) does not depend on x), the default SEs
#       should still be valid. If errors were truly heteroskedastic, we would
#       need vce(robust) / HC standard errors.
#
# (C) TEST SIZE CHECK: Rejection rate ~ 0.05
#     - We are testing the TRUE null hypothesis (H0: beta=2 is TRUE).
#     - The rejection rate should be approximately 0.05 (the nominal level).
#     - If it is much higher than 0.05, the test is "over-sized" (too liberal).
#     - If it is much lower than 0.05, the test is "under-sized" (too conservative).
#     - For N=150 with chi-squared errors, the CLT provides a good enough
#       normal approximation that the t-test works well.

# Plot sampling distribution
#
# Left panel:  Histogram of 1,000 beta_hat values, with a vertical line at
#              the true value beta=2. This visualizes unbiasedness.
# Right panel: Histogram of the 1,000 t-statistics, overlaid with the
#              theoretical t(148) density. If the t-test is correctly sized,
#              the histogram should closely match the theoretical density.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(b2_arr, bins=40, density=True, alpha=0.8, color='#ED7D31', edgecolor='white')
axes[0].axvline(x=2, color='red', linestyle='--', linewidth=2, label='True beta=2')
axes[0].set_title('Sampling Distribution of OLS beta_hat', fontweight='bold')
axes[0].set_xlabel(r'$\hat{\beta}$')
axes[0].legend()

t2_arr = (b2_arr - 2) / se2_arr
axes[1].hist(t2_arr, bins=40, density=True, alpha=0.8, color='#4472C4', edgecolor='white')
xg = np.linspace(-4, 4, 200)
axes[1].plot(xg, stats.t.pdf(xg, numobs - 2), 'r-', linewidth=2, label=f't({numobs-2})')
axes[1].set_title('t-Statistic vs Theoretical t-distribution', fontweight='bold')
axes[1].set_xlabel('t-statistic')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_ch2_ols_sim.png', dpi=150)
plt.show()

# INTERPRETATION OF OLS PLOTS:
# ------------------------------
# Left panel: The histogram of beta_hat values should be centered at 2.0
# (confirming unbiasedness) and roughly bell-shaped (by the CLT).
#
# Right panel: The t-statistic histogram should closely follow the red
# theoretical t(148) curve. The area in the tails beyond +/-1.976 should
# contain about 5% of the simulated t-statistics (confirming correct test size).
# If the histogram is wider than the theoretical curve, the test is under-
# rejecting; if narrower, it is over-rejecting.

# ============================================================
# 6. MEASUREMENT ERROR (ERRORS-IN-VARIABLES)
# ============================================================
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# In practice, we often cannot observe the true variable x* and instead
# observe a noisy measurement x = x* + v, where v is measurement error.
# This is called the "errors-in-variables" (EIV) problem.
#
# THE DATA GENERATING PROCESS (DGP):
# -----------------------------------
#   True model:    y = beta * x*   + u     (beta = 1)
#   Observed:      x = x*         + v     (x* is measured with error v)
#   We estimate:   y = gamma * x  + error  (OLS of y on observed x)
#
#   where:
#     x* ~ N(0, 9)   i.e., Var(x*) = 9
#     u  ~ N(0, 1)   i.e., Var(u)  = 1
#     v  ~ N(0, 1)   i.e., Var(v)  = 1
#     x*, u, v are mutually independent
#
# THE ATTENUATION BIAS FORMULA:
# ------------------------------
#   plim(gamma_hat) = beta * Var(x*) / [Var(x*) + Var(v)]
#                   = 1   * 9       / [9        + 1      ]
#                   = 0.9
#
# This is called "attenuation bias" because the estimate is ALWAYS biased
# TOWARD ZERO (attenuated). Intuitively, measurement error in x adds noise
# that weakens the apparent relationship between y and x.
#
# KEY PROPERTIES:
# - The bias is ALWAYS toward zero (not in a random direction)
# - More measurement error (larger Var(v)) means more bias
# - More signal (larger Var(x*)) means less bias (better signal-to-noise)
# - The formula Var(x*)/[Var(x*)+Var(v)] is called the "reliability ratio"
# - This bias does NOT vanish with more observations -- it is an INCONSISTENCY
#
# IMPORTANT: This is a regression without a constant (noconstant in Stata).
# The Stata do-file uses: regress y x, noconstant
# The Python code accordingly passes x_me directly to sm.OLS without
# adding a constant via sm.add_constant.
#
# STATA EQUIVALENTS:
#   matrix mu = (0,0,0)
#   matrix sigmasq = (9,0,0\0,1,0\0,0,1)   -> independent normals with specified variances
#   drawnorm xstar u v, means(mu) cov(sigmasq)
#   generate y = 1*xstar + u
#   generate x = xstar + v
#   regress y x, noconstant
#
print("\n" + "=" * 60)
print("6. MEASUREMENT ERROR (ATTENUATION BIAS)")
print("=" * 60)

np.random.seed(10101)
n_meas = 10000

# True model: y = 1*xstar + u
# Observed: x = xstar + v (measurement error)
xstar = np.random.normal(0, 3, n_meas)  # Var(xstar) = 9
u = np.random.normal(0, 1, n_meas)       # Var(u) = 1
v = np.random.normal(0, 1, n_meas)       # Var(v) = 1

y_me = 1 * xstar + u                     # True DGP
x_me = xstar + v                         # Mismeasured x

# OLS with mismeasured x (no constant, matching Stata's "noconstant" option)
model_me = sm.OLS(y_me, x_me).fit()

print(f"  True beta:    1.000")
print(f"  Estimated:    {model_me.params[0]:.4f}")
print(f"  Expected:     {9 / (9 + 1):.4f}  [= Var(x*)/(Var(x*)+Var(v))]")
print(f"  Attenuation:  {(1 - model_me.params[0]) * 100:.1f}% bias toward zero")

# INTERPRETATION OF MEASUREMENT ERROR RESULTS:
# -----------------------------------------------
# The estimated coefficient should be approximately 0.9 (not the true value of 1.0).
# This ~10% bias toward zero is the attenuation bias.
#
# With n=10,000 observations, the estimate should be very close to 0.9 (the
# probability limit), demonstrating that this is not a small-sample issue --
# the bias persists even with an enormous sample. This is INCONSISTENCY.
#
# PRACTICAL IMPLICATIONS:
# - In health economics: if income is measured with error, its effect on
#   health outcomes will be UNDERSTATED.
# - In labor economics: if education is self-reported (with error), returns
#   to education will be biased downward.
# - Solution: Use instrumental variables (IV) with an instrument that is
#   correlated with x* but not with the measurement error v. See Chapter 4.

# ============================================================
# 7. ENDOGENOUS REGRESSOR
# ============================================================
#
# CONCEPTUAL BACKGROUND:
# ----------------------
# An "endogenous" regressor is one that is correlated with the error term:
# Cov(x, u) != 0. This can arise from:
#   (a) Omitted variable bias (a confounding variable affects both x and y)
#   (b) Simultaneity (x affects y AND y affects x)
#   (c) Measurement error (see Section 6 above)
#   (d) Self-selection (individuals choose x based on unobservable factors
#       that also affect y)
#
# THE DATA GENERATING PROCESS (DGP):
# -----------------------------------
#   u ~ N(0, 1)       (error term)
#   z ~ N(0, 1)       (exogenous component)
#   x = 0.5*u + z     (x is endogenous: it depends on u!)
#   y = 10 + 2*x + u  (true model)
#
# Since x = 0.5*u + z:
#   Cov(x, u) = Cov(0.5*u + z, u) = 0.5*Var(u) = 0.5
#   Var(x)    = 0.25*Var(u) + Var(z) = 0.25 + 1 = 1.25
#
# THE BIAS FORMULA (for OLS of y on x):
# ----------------------------------------
#   plim(beta_hat_OLS) = beta_true + Cov(x,u)/Var(x)
#                      = 2         + 0.5 / 1.25
#                      = 2         + 0.4
#                      = 2.4
#
#   So OLS is biased UPWARD by 0.4. This bias is PERMANENT -- it does not
#   shrink as n increases. OLS is INCONSISTENT for beta_1 when Cov(x,u) != 0.
#
# NOTE ON THE STATA DO-FILE:
# ---------------------------
# The Stata do-file's endogreg1 program actually regresses y on z (not on x),
# and tests whether the coefficient on z equals 2. This is a different
# exercise: z is exogenous (Cov(z,u) = 0), so OLS of y on z should be
# consistent for the REDUCED-FORM coefficient of z.
#
# The Python code below regresses y on x (the endogenous variable) to
# demonstrate the inconsistency of OLS when Cov(x,u) != 0. This more
# directly illustrates the endogeneity problem.
#
# WHAT THE CODE DOES:
# -------------------
# For each of 1,000 simulations:
#   1. Generate u, z independently from N(0,1)
#   2. Construct x = 0.5*u + z (making x endogenous)
#   3. Construct y = 10 + 2*x + u
#   4. Run OLS of y on x (with constant) -- this is the BIASED regression
#   5. Store the slope estimate
# Then examine the average slope across simulations.
#
# STATA EQUIVALENTS:
#   program endogreg1, rclass
#       generate u = rnormal()
#       generate z = rnormal()
#       generate x = 0.5*u + z
#       generate y = 10 + 2*x + u
#       regress y x              (Python version; Stata regresses on z)
#   end
#   simulate ... , reps($numsims): endogreg1
#
print("\n" + "=" * 60)
print("7. ENDOGENOUS REGRESSOR")
print("=" * 60)

np.random.seed(10101)
n_endo_sims = 1000

b_endo_list = []
for _ in range(n_endo_sims):
    u_e = np.random.normal(size=numobs)
    z_e = np.random.normal(size=numobs)
    x_e = 0.5 * u_e + z_e        # x is correlated with u (endogenous!)
    y_e = 10 + 2 * x_e + u_e

    # OLS regression of y on x (biased!)
    model_e = sm.OLS(y_e, sm.add_constant(x_e)).fit()
    b_endo_list.append(model_e.params[1])

b_endo_arr = np.array(b_endo_list)
print(f"  True beta:     2.000")
print(f"  E[beta_OLS]:   {b_endo_arr.mean():.4f}  (biased upward!)")
print(f"  Bias:          {b_endo_arr.mean() - 2:.4f}")
print(f"  Expected bias: {0.5 * 1 / (0.5**2 * 1 + 1):.4f}  [= Cov(x,u)/Var(x)]")
print(f"\n  OLS is INCONSISTENT with endogenous regressors!")
print(f"  Solution: Use Instrumental Variables (Chapter 4)")

# INTERPRETATION OF ENDOGENEITY RESULTS:
# ----------------------------------------
# The average OLS estimate should be approximately 2.4, NOT 2.0.
# The bias of ~0.4 is the endogeneity bias: Cov(x,u)/Var(x) = 0.5/1.25 = 0.4.
#
# KEY DISTINCTIONS FROM MEASUREMENT ERROR (Section 6):
# - Measurement error ALWAYS biases toward zero (attenuation).
# - Endogeneity can bias in EITHER direction, depending on the sign of Cov(x,u).
#   Here Cov(x,u) > 0, so the bias is upward (overestimation).
#
# BOTH problems share a critical feature: the bias is an INCONSISTENCY.
# It does NOT go away with more data. The only solution is to use methods
# that account for the endogeneity:
#   - Instrumental Variables (IV) / Two-Stage Least Squares (2SLS)  -> Chapter 4
#   - Control function approaches
#   - Panel data fixed effects (if the endogeneity comes from time-invariant
#     unobserved heterogeneity) -> Chapter 6
#
# PRACTICAL EXAMPLES OF ENDOGENEITY:
# - Returns to education: ability is in the error term and correlated with
#   education (more able people get more education). OLS overestimates the
#   return to education.
# - Effect of police on crime: cities with more crime hire more police.
#   OLS of crime on police may show a POSITIVE coefficient (more police
#   associated with more crime), even though the true causal effect is negative.
# - Effect of class size on test scores: schools may assign weaker students
#   to smaller classes, biasing the estimated effect of class size.

print("\n" + "=" * 60)
print("END OF CHAPTER 2")
print("=" * 60)
