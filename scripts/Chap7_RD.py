"""
=============================================================================
EC627 Microeconometrics - Chapter 7
Regression Discontinuity Designs
=============================================================================
Instructor: Asst. Prof. Dr. Tiraphap Fakthong
Thammasat University

Dataset: Chap7_headstart.xlsx (Ludwig & Miller, 2007, QJE)
This script replicates "Chap 7 CT_2021_NBER.do" in Python.
Designed to run on Google Colab.

Required packages:
    pip install pandas numpy statsmodels scipy matplotlib rdrobust openpyxl
    (rdrobust: pip install rdrobust  -- Python version of the Stata package)

=============================================================================
OVERVIEW OF KEY TOPICS IN THIS SCRIPT
=============================================================================

This script covers the Regression Discontinuity (RD) design, one of the most
credible quasi-experimental methods in modern econometrics. The key sections are:

1. DATA LOADING AND VARIABLE CONSTRUCTION
   - The Ludwig & Miller (2007) Head Start dataset
   - Defining the running variable, cutoff, and treatment indicator
   - Centering the running variable at the cutoff

2. SIMPLE DIFFERENCE IN MEANS (T-TEST)
   - A naive comparison of treated vs. control group means
   - Why this is a useful starting point but NOT a valid RD estimate

3. RD PLOT (VISUAL EVIDENCE)
   - Binned scatter plot with fitted polynomials on each side of the cutoff
   - The visual "jump" at the cutoff as evidence of a treatment effect

4. PARAMETRIC RD (LOCAL LINEAR REGRESSION)
   - The workhorse RD specification: Y = alpha + tau*T + beta*R + gamma*(T*R) + e
   - Restricting the sample to a bandwidth around the cutoff (h=9)
   - Interpreting the coefficient on T as the RD treatment effect

5. RDROBUST ESTIMATION
   - Data-driven, MSE-optimal bandwidth selection
   - Bias-corrected inference with robust confidence intervals
   - Comparison with manual parametric RD results

6. BANDWIDTH SENSITIVITY ANALYSIS
   - Running the RD regression for multiple bandwidths (h = 4, 6, ..., 15)
   - Checking whether the RD estimate is stable across bandwidth choices

7. FALSIFICATION / VALIDITY TESTS
   a. Density test (McCrary): Is the running variable manipulated at the cutoff?
   b. Covariate balance: Are pre-treatment covariates smooth at the cutoff?
   c. Placebo outcome: Is there a "jump" in outcomes unrelated to treatment?
   d. Placebo cutoff: Is there a jump at fake cutoff values?

=============================================================================
BACKGROUND: WHAT IS REGRESSION DISCONTINUITY?
=============================================================================

Regression Discontinuity (RD) exploits a rule-based treatment assignment:
units with a "running variable" (also called "forcing variable" or "score")
above (or below) a known cutoff receive treatment, while those on the other
side do not.

The KEY INSIGHT is that units just barely above and just barely below the
cutoff are essentially "as good as randomly assigned" to treatment and control,
because their running variable values are nearly identical. This creates a
local quasi-experiment right at the cutoff.

In a SHARP RD design, treatment is a deterministic function of the running
variable: every unit above the cutoff is treated, every unit below is not.
In a FUZZY RD design, the cutoff only changes the probability of treatment
(think of it like an instrument for actual treatment receipt).

This script implements a SHARP RD design using the Ludwig & Miller (2007)
application, which studies the effect of the Head Start program on child
mortality.

=============================================================================
THE LUDWIG & MILLER (2007) APPLICATION
=============================================================================

Context: Head Start is a U.S. federal preschool program for disadvantaged
children. In 1965, the Office of Economic Opportunity (OEO) provided
"technical assistance" (grants to help communities write funding proposals)
to the 300 poorest counties in the U.S., as measured by the 1960 county-level
poverty rate. Counties with poverty rates above 59.1984% received this help,
making them much more likely to start Head Start programs.

- Running variable (X): County-level poverty rate in 1960 (povrate60)
- Cutoff (C): 59.1984%
- Treatment (T): County received OEO technical assistance (poverty rate >= cutoff)
- Outcome (Y): Child mortality rate (ages 5-9), causes related to Head Start

The fundamental research question: Did Head Start reduce child mortality in
the counties that received assistance?
=============================================================================
"""

# ============================================================
# SETUP
# ============================================================
# We import the standard scientific Python stack:
#   - pandas: for data manipulation (DataFrames)
#   - numpy: for numerical computations
#   - matplotlib: for creating plots
#   - scipy.stats: for statistical tests (e.g., t-test)
#   - statsmodels: for OLS regression with robust standard errors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# Set numpy to display 4 decimal places for cleaner output
np.set_printoptions(precision=4)

print("=" * 60)
print("CHAPTER 7: REGRESSION DISCONTINUITY DESIGNS")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
#
# This section loads the Ludwig & Miller (2007) Head Start dataset and
# constructs the key variables needed for the RD analysis.
#
# WHAT WE DO:
#   1. Load the Excel file into a pandas DataFrame.
#   2. Define the outcome variable (Y), running variable (X), and cutoff (C).
#   3. Create a list of pre-treatment covariates from the 1960 Census.
#   4. Construct:
#      - R: the CENTERED running variable (R = X - C), so R = 0 at the cutoff.
#        Centering is important because it makes the intercept in our regression
#        interpretable as the expected outcome right at the cutoff.
#      - T: the treatment indicator (T = 1 if X >= C, i.e., if the county's
#        poverty rate is at or above the cutoff and therefore received OEO help).
#
# WHY CENTER THE RUNNING VARIABLE?
#   If we use the raw running variable X in the regression Y = a + tau*T + b*X,
#   then "a" is the predicted outcome when X=0 (poverty rate = 0%), which is
#   far from the cutoff and not meaningful. By centering (R = X - C), the
#   intercept "a" becomes the predicted outcome at the cutoff for the control
#   group, and "tau" (the coefficient on T) becomes the jump at the cutoff,
#   i.e., the RD treatment effect. This is exactly what we want.
#
# THE COVARIATES:
#   The covariates are 1960 Census county characteristics (population, school
#   enrollment rates, urbanization, racial composition). These are PRE-TREATMENT
#   variables determined before Head Start existed. We will use them later in
#   falsification tests to check that they are balanced (i.e., continuous) at
#   the cutoff.
# ============================================================
df = pd.read_excel("Chap7_headstart.xlsx")
print(f"\nDataset: Head Start (Ludwig & Miller, 2007)")
print(f"Observations: {len(df)}")

# Define variables (matching Stata globals)
Y = 'mort_age59_related_postHS'   # Outcome: child mortality
X = 'povrate60'                    # Running variable: 1960 poverty rate
C = 59.1984                        # Cutoff

covariates = [c for c in ['census1960_pop', 'census1960_pctsch1417',
              'census1960_pctsch534', 'census1960_pctsch25plus',
              'census1960_pop1417', 'census1960_pop534',
              'census1960_pop25plus', 'census1960_pcturban',
              'census1960_pctblack'] if c in df.columns]

# Create running variable centered at cutoff
df['R'] = df[X] - C
df['T'] = (df[X] >= C).astype(int)

print(f"\nCutoff: {C}")
print(f"\n--- Summary Statistics ---")
print(df[[Y, X]].describe().round(3))
print(f"\nTreatment/Control split:")
print(df['T'].value_counts())

# ------------------------------------------------------------
# INTERPRETATION (Section 1):
#   - Check the number of observations: the dataset should have roughly 3,000
#     U.S. counties.
#   - The Treatment/Control split tells you how many counties are above vs.
#     below the cutoff. For a credible RD, you want a reasonable number of
#     observations on BOTH sides of the cutoff. If almost all observations are
#     on one side, the local comparison near the cutoff will be underpowered.
#   - The summary statistics for the outcome Y (child mortality) give you a
#     sense of the scale: mean, standard deviation, min/max. This helps you
#     judge the economic significance of the RD estimate later.
# ------------------------------------------------------------

# ============================================================
# 2. T-TEST: SIMPLE DIFFERENCE IN MEANS
# ============================================================
#
# WHAT THIS SECTION DOES:
#   We compute the simple difference in average child mortality between counties
#   above the cutoff (treated) and counties below the cutoff (control), along
#   with a two-sample t-test for statistical significance.
#
# WHY THIS IS A USEFUL STARTING POINT:
#   It is the simplest possible comparison. If there is a large, significant
#   difference in means, it suggests something is happening at the cutoff.
#
# WHY THIS IS NOT A VALID RD ESTIMATE:
#   The simple difference in means uses ALL observations, including counties
#   far away from the cutoff. Counties with a poverty rate of 20% are very
#   different from counties with a poverty rate of 80% in many ways beyond
#   just Head Start eligibility. The simple difference conflates the treatment
#   effect with the underlying relationship between poverty and mortality.
#
#   The core of RD is to compare units CLOSE TO the cutoff. The t-test ignores
#   this localization principle entirely.
#
# KEY PYTHON DETAILS:
#   - df[df['T'] == 1][Y].dropna(): selects the outcome variable for treated
#     observations, dropping any missing values.
#   - stats.ttest_ind(): performs an independent two-sample t-test. It returns
#     the t-statistic and p-value. The null hypothesis is that the two groups
#     have equal population means.
# ============================================================
print("\n" + "=" * 60)
print("2. SIMPLE DIFFERENCE IN MEANS")
print("=" * 60)

treated = df[df['T'] == 1][Y].dropna()
control = df[df['T'] == 0][Y].dropna()

t_stat, p_val = stats.ttest_ind(treated, control)
print(f"  Mean (treated):  {treated.mean():.4f}")
print(f"  Mean (control):  {control.mean():.4f}")
print(f"  Difference:      {treated.mean() - control.mean():.4f}")
print(f"  t-statistic:     {t_stat:.4f}")
print(f"  p-value:         {p_val:.4f}")

# ------------------------------------------------------------
# INTERPRETATION (Section 2):
#   - If the difference is negative, it means treated counties (higher poverty)
#     have LOWER child mortality, which might seem counterintuitive. But
#     remember: this is a GLOBAL comparison, not a local one. We cannot
#     interpret this as the causal effect of Head Start.
#   - If the difference is positive, it means treated counties have higher
#     mortality -- which is what we might naively expect, since poorer counties
#     tend to have worse health outcomes (selection bias!).
#   - The key takeaway: this comparison is contaminated by confounders. We need
#     the RD design to isolate the causal effect of treatment near the cutoff.
# ------------------------------------------------------------

# ============================================================
# 3. RD PLOT
# ============================================================
#
# WHAT THIS SECTION DOES:
#   Creates the canonical RD visualization: a "binned scatter plot" with
#   fitted polynomial curves on each side of the cutoff.
#
# WHY THE RD PLOT MATTERS:
#   The RD plot is arguably the most important output of any RD analysis.
#   It provides VISUAL EVIDENCE of whether there is a discontinuity (jump)
#   in the outcome at the cutoff. If you cannot "see" the jump in the plot,
#   you should be skeptical of any statistical claim of an RD effect.
#
# HOW THE PLOT IS CONSTRUCTED:
#   1. BINNING: The range of the running variable is divided into 40 equal-width
#      bins. Within each bin, we compute the average outcome. This smooths out
#      the noise from individual data points and reveals the underlying trend.
#      (pd.cut creates the bins; groupby().mean() computes bin averages.)
#   2. PLOTTING BINS: We plot the bin means as scatter points, coloring them
#      differently for bins below (control, blue) and above (treated, orange)
#      the cutoff.
#   3. FITTING POLYNOMIALS: We fit a 2nd-degree polynomial (quadratic) SEPARATELY
#      on each side of the cutoff. This allows the relationship between X and Y
#      to be nonlinear, and -- critically -- to be DIFFERENT on each side.
#      (np.polyfit fits the polynomial; np.poly1d creates a callable function.)
#   4. CUTOFF LINE: A vertical dashed red line marks the cutoff.
#
# WHAT TO LOOK FOR:
#   - Is there a visible JUMP (discontinuity) in the fitted curves at the
#     cutoff? If the orange curve at the cutoff is noticeably below the blue
#     curve at the cutoff, this suggests Head Start reduced child mortality.
#   - Are the bin means scattered roughly around the fitted curves? If bins
#     are wildly scattered, the polynomial may be a poor fit.
#   - Is the relationship between X and Y roughly smooth AWAY from the cutoff?
#     We want the ONLY discontinuity to be at the cutoff itself.
# ============================================================
print("\n" + "=" * 60)
print("3. RD PLOT")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

# Bin scatter plot
df_plot = df[[X, Y]].dropna()
n_bins = 40
df_plot['bin'] = pd.cut(df_plot[X], bins=n_bins)
bin_means = df_plot.groupby('bin')[Y].mean().reset_index()
bin_means['x_mid'] = bin_means['bin'].apply(lambda b: float(b.mid)).astype(float)

# Plot bins
below = bin_means[bin_means['x_mid'].values < C]
above = bin_means[bin_means['x_mid'].values >= C]
ax.scatter(below['x_mid'], below[Y], color='#4472C4', s=40, zorder=5, label='Control')
ax.scatter(above['x_mid'], above[Y], color='#ED7D31', s=40, zorder=5, label='Treated')

# Fit local polynomial on each side
for side, color, label_side in [(df_plot[df_plot[X] < C], '#4472C4', 'Below'),
                                 (df_plot[df_plot[X] >= C], '#ED7D31', 'Above')]:
    if len(side) > 10:
        z = np.polyfit(side[X], side[Y], 2)
        p = np.poly1d(z)
        x_range = np.linspace(side[X].min(), side[X].max(), 100)
        ax.plot(x_range, p(x_range), color=color, linewidth=2)

ax.axvline(x=C, color='red', linestyle='--', linewidth=2, label=f'Cutoff ({C})')
ax.set_xlabel('Poverty Rate 1960', fontsize=12)
ax.set_ylabel('Child Mortality (age 5-9)', fontsize=12)
ax.set_title('RD Plot: Head Start Program Effect on Child Mortality', fontweight='bold', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('fig_ch7_rdplot.png', dpi=150)
plt.show()
print("Figure saved: fig_ch7_rdplot.png")

# ------------------------------------------------------------
# INTERPRETATION (Section 3):
#   - The RD plot should show the outcome trend on each side of the cutoff.
#   - Look for a DOWNWARD JUMP at the cutoff: if the fitted curve drops as you
#     cross from left (control) to right (treated), it suggests Head Start
#     reduced mortality.
#   - The size of the jump corresponds roughly to the RD treatment effect
#     we will estimate formally in the next section.
#   - If there is no visible jump, the RD estimate will likely be small and
#     statistically insignificant.
#   - Also check whether the polynomial fits look reasonable. If the curves
#     are "chasing" outlier bins, the fit may be unreliable.
# ------------------------------------------------------------

# ============================================================
# 4. PARAMETRIC RD (Replicating Ludwig & Miller)
# ============================================================
#
# WHAT THIS SECTION DOES:
#   Estimates the RD treatment effect using a LOCAL LINEAR REGRESSION within
#   a bandwidth h = 9 percentage points around the cutoff. This is the
#   "parametric" approach to RD and replicates the Ludwig & Miller (2007)
#   specification.
#
# THE ECONOMETRIC MODEL:
#   Y_i = alpha + tau * T_i + beta * R_i + gamma * (T_i * R_i) + epsilon_i
#
#   where:
#     Y_i    = child mortality for county i
#     T_i    = treatment indicator (= 1 if poverty rate >= cutoff)
#     R_i    = centered running variable (= poverty rate - 59.1984)
#     T_i*R_i = interaction of treatment with the running variable
#     alpha  = intercept (predicted outcome for control group at the cutoff)
#     tau    = THE RD TREATMENT EFFECT -- the jump in expected Y at the cutoff
#     beta   = slope of the outcome-running variable relationship for control group
#     gamma  = how the slope DIFFERS for the treated group
#
# WHY THE INTERACTION TERM (T*R)?
#   Without the interaction, we would force the slope of Y on R to be the same
#   on both sides of the cutoff. The interaction allows DIFFERENT SLOPES on
#   each side, which is essential. If we write out the model for each group:
#     Control (T=0): Y = alpha + beta*R
#     Treated (T=1): Y = (alpha + tau) + (beta + gamma)*R
#   So each side gets its own intercept and slope -- this is equivalent to
#   fitting separate linear regressions on each side.
#
# WHY BANDWIDTH h = 9?
#   We restrict the sample to counties within 9 percentage points of the cutoff
#   (i.e., |R| <= 9, meaning poverty rates between ~50.2% and ~68.2%). This
#   ensures we are comparing counties that are SIMILAR in poverty rates, making
#   the "as good as random" assumption more plausible. The choice h = 9 matches
#   Ludwig & Miller (2007). We will test sensitivity to this choice in Section 6.
#
# ROBUST STANDARD ERRORS:
#   We use HC1 (heteroskedasticity-consistent) standard errors (cov_type='HC1')
#   which correspond to Stata's ", robust" option. This is standard practice
#   because RD errors are often heteroskedastic (variance differs across
#   counties).
#
# KEY PYTHON DETAILS:
#   - df[abs(df['R']) <= h]: selects rows where the centered running variable
#     is within the bandwidth. This is the "local" part of local linear regression.
#   - sm.add_constant(): adds a column of 1s for the intercept term.
#   - sm.OLS(...).fit(cov_type='HC1'): fits OLS with robust (HC1) standard errors.
#   - model_rd.params['T']: extracts the coefficient on T, which is our RD estimate.
# ============================================================
print("\n" + "=" * 60)
print("4. PARAMETRIC RD (Local Linear, h=9)")
print("=" * 60)

# Stata: reg $Y T##c.R if abs(R)<=9, robust
h = 9  # bandwidth
df_rd = df[abs(df['R']) <= h].copy()

# Create interaction term
df_rd['T_x_R'] = df_rd['T'] * df_rd['R']
X_rd = sm.add_constant(df_rd[['T', 'R', 'T_x_R']])
y_rd = df_rd[Y]

model_rd = sm.OLS(y_rd, X_rd).fit(cov_type='HC1')
for v in ['const', 'T', 'R', 'T_x_R']:
    print(f"  {v:8s}  coef={model_rd.params[v]:10.4f}  se={model_rd.bse[v]:8.4f}  p={model_rd.pvalues[v]:.4f}")
print(f"\n  RD Estimate (coefficient on T): {model_rd.params['T']:.4f}")
print(f"  Robust SE: {model_rd.bse['T']:.4f}")
print(f"  p-value: {model_rd.pvalues['T']:.4f}")

# ------------------------------------------------------------
# INTERPRETATION (Section 4):
#   - The coefficient on T is the RD TREATMENT EFFECT: the estimated
#     discontinuous jump in child mortality at the cutoff. A NEGATIVE
#     coefficient means Head Start REDUCED child mortality.
#   - The robust standard error and p-value tell you whether this effect is
#     statistically significant. Compare the p-value to 0.05 (or 0.10).
#   - The coefficient on R tells you the slope of the mortality-poverty
#     relationship for the CONTROL group (below the cutoff).
#   - The coefficient on T_x_R tells you how the slope DIFFERS for the treated
#     group. If it is near zero, the slopes are similar on both sides.
#   - The constant (intercept) is the predicted mortality for control counties
#     RIGHT AT the cutoff (R=0).
#   - Check the number of observations used (len(df_rd)) -- restricting to h=9
#     drops many counties, which is the trade-off between bias (wider bandwidth)
#     and variance (narrower bandwidth).
# ------------------------------------------------------------

# ============================================================
# 5. RDROBUST (if available)
# ============================================================
#
# WHAT THIS SECTION DOES:
#   Uses the rdrobust package (Calonico, Cattaneo, and Titiunik, 2014) to
#   compute the RD estimate with a DATA-DRIVEN optimal bandwidth and
#   bias-corrected robust inference.
#
# WHY USE RDROBUST INSTEAD OF MANUAL OLS?
#   1. BANDWIDTH SELECTION: In Section 4, we chose h = 9 somewhat arbitrarily
#      (following Ludwig & Miller). But the choice of bandwidth involves a
#      bias-variance trade-off:
#        - Too wide (large h): more data, lower variance, but higher bias
#          because the linear approximation is less accurate far from the cutoff.
#        - Too narrow (small h): less bias, but higher variance because we use
#          fewer observations.
#      rdrobust selects the "MSE-optimal" bandwidth, which minimizes the mean
#      squared error (= bias^2 + variance). This is a principled, data-driven
#      choice.
#
#   2. BIAS CORRECTION: Local polynomial estimators are biased in finite samples.
#      rdrobust applies an automatic bias correction and constructs "robust"
#      confidence intervals that account for this correction. This is more
#      statistically principled than standard OLS confidence intervals.
#
#   3. KERNEL WEIGHTING: By default, rdrobust uses a TRIANGULAR KERNEL, which
#      gives more weight to observations closer to the cutoff and less weight
#      to those further away. This is optimal for minimizing MSE in boundary
#      estimation problems. (By contrast, OLS with a bandwidth cutoff uses a
#      "uniform kernel" -- equal weight to all observations within the bandwidth.)
#
# WHAT rdrobust REPORTS:
#   - "Conventional" estimate: the standard local polynomial estimate
#   - "Bias-corrected" estimate: adjusts for the leading bias term
#   - "Robust" p-value/CI: accounts for the additional uncertainty from bias
#     correction. THIS IS THE ONE YOU SHOULD REPORT in your paper.
#   - Bandwidth (h): the MSE-optimal bandwidth selected by the algorithm
#
# THE TWO CALLS IN THIS SECTION:
#   1. Default rdrobust: Uses MSE-optimal bandwidth and triangular kernel.
#      This is the recommended specification.
#   2. Uniform kernel with h=9: Matches the manual OLS specification from
#      Section 4 (uniform kernel = equal weights = OLS). This lets us verify
#      that rdrobust with uniform kernel and fixed bandwidth gives the same
#      result as our manual regression.
#
# rdbwselect:
#   This function ONLY selects the optimal bandwidth without estimating the
#   treatment effect. It is useful for examining different bandwidth selection
#   criteria (MSE-optimal, CER-optimal, etc.) and comparing their values.
#
# KEY PYTHON DETAILS:
#   - rdrobust(y, x, c=C): the main function. y and x must be numpy arrays
#     (.values converts a pandas Series to a numpy array).
#   - rd_result.Estimate: a DataFrame with conventional, bias-corrected, and
#     robust estimates.
#   - rd_result.pv: p-values. iloc[2] is the ROBUST p-value (the one to report).
#   - rd_result.bws: bandwidths. iloc[0,0] is the main bandwidth.
# ============================================================
print("\n" + "=" * 60)
print("5. RDROBUST ESTIMATION")
print("=" * 60)

try:
    from rdrobust import rdrobust, rdplot as rd_plot_func, rdbwselect

    print("\n--- rdrobust: Default (MSE-optimal bandwidth, triangular kernel) ---")
    df_rd_clean = df[[Y, X]].dropna()
    rd_result = rdrobust(df_rd_clean[Y].values, df_rd_clean[X].values, c=C)
    print(f"  Point estimate:  {rd_result.Estimate.iloc[0]:.4f}")
    print(f"  Robust p-value:  {rd_result.pv.iloc[2]:.4f}")
    print(f"  Bandwidth (h):   {rd_result.bws.iloc[0, 0]:.4f}")
    print(rd_result)

    # ------------------------------------------------------------
    # INTERPRETATION (rdrobust default):
    #   - The "Point estimate" is the conventional local-polynomial RD estimate.
    #   - The "Robust p-value" (pv.iloc[2]) is the one you should report. It
    #     accounts for bias correction and is based on the robust confidence interval.
    #   - The "Bandwidth (h)" is the MSE-optimal bandwidth selected by the
    #     algorithm. Compare it to our manual choice of h=9 -- if they are
    #     similar, our manual analysis was reasonable.
    #   - If the robust p-value < 0.05, we have statistically significant evidence
    #     of a discontinuity in child mortality at the Head Start cutoff.
    # ------------------------------------------------------------

    print("\n--- rdrobust: Uniform kernel, h=9 (replicate Ludwig & Miller) ---")
    rd_result2 = rdrobust(df_rd_clean[Y].values, df_rd_clean[X].values, c=C, h=9, kernel='uniform')
    print(f"  Point estimate:  {rd_result2.Estimate.iloc[0]:.4f}")
    print(f"  Robust p-value:  {rd_result2.pv.iloc[2]:.4f}")

    # ------------------------------------------------------------
    # INTERPRETATION (rdrobust with uniform kernel, h=9):
    #   - With kernel='uniform' and h=9, rdrobust should produce a point estimate
    #     very close to the parametric OLS result from Section 4.
    #   - Any small differences arise because rdrobust uses a slightly different
    #     estimation procedure (local polynomial vs. standard OLS) and applies
    #     bias correction. The robust p-value may also differ from the OLS p-value
    #     because of the bias-correction adjustment.
    # ------------------------------------------------------------

    print("\n--- Bandwidth selection ---")
    bw_result = rdbwselect(df_rd_clean[Y].values, df_rd_clean[X].values, c=C, kernel='uniform', all=True)
    print(bw_result)

    # ------------------------------------------------------------
    # INTERPRETATION (Bandwidth selection):
    #   - rdbwselect reports several bandwidth choices:
    #       * mserd: MSE-optimal, same bandwidth on both sides (most common)
    #       * msetwo: MSE-optimal, different bandwidths on each side
    #       * cerrd: Coverage Error Rate optimal (tends to be smaller than MSE)
    #       * certwo: CER-optimal, different bandwidths on each side
    #   - Use these to understand the range of reasonable bandwidths for your data.
    #   - If the MSE-optimal bandwidth is very different from h=9, you should
    #     investigate why and check sensitivity (done in Section 6).
    # ------------------------------------------------------------

except ImportError:
    print("  rdrobust package not installed.")
    print("  Install with: pip install rdrobust")
    print("  Using manual parametric RD instead (see section 4 above).")

# ============================================================
# 6. RD WITH DIFFERENT BANDWIDTHS
# ============================================================
#
# WHAT THIS SECTION DOES:
#   Re-estimates the parametric RD model (from Section 4) for a range of
#   different bandwidths: h = 4, 6, 8, 9, 10, 12, 15.
#
# WHY BANDWIDTH SENSITIVITY MATTERS:
#   The bandwidth is a researcher choice that involves a bias-variance trade-off:
#     - SMALL bandwidth (e.g., h=4): uses only counties very close to the cutoff,
#       so the linear approximation is accurate (low bias), but there are few
#       observations (high variance / large standard errors).
#     - LARGE bandwidth (e.g., h=15): uses many counties (low variance), but
#       counties far from the cutoff may not be comparable, and the linear
#       approximation may be poor (high bias).
#
#   A CREDIBLE RD result should be ROBUST across a range of reasonable
#   bandwidths. If the RD estimate flips sign or becomes insignificant for
#   bandwidths slightly larger or smaller than the chosen one, the result is
#   fragile and should be treated with caution.
#
# WHAT TO LOOK FOR IN THE OUTPUT TABLE:
#   - Are the RD estimates (coefficient on T) roughly SIMILAR across bandwidths?
#     Small variation is normal, but dramatic changes suggest sensitivity.
#   - Standard errors should generally DECREASE as h increases (more data).
#   - p-values may become more significant with larger h (more power), but
#     could also become less significant if bias contaminates the estimate.
#   - N (sample size) increases with bandwidth, as expected.
#
# KEY PYTHON DETAILS:
#   - The loop iterates over each bandwidth value, subsets the data, creates
#     the interaction term, fits OLS with HC1 errors, and prints a row.
#   - This is the same model as Section 4, just repeated for different h values.
# ============================================================
print("\n" + "=" * 60)
print("6. SENSITIVITY TO BANDWIDTH CHOICE")
print("=" * 60)

bandwidths = [4, 6, 8, 9, 10, 12, 15]
print(f"\n  {'Bandwidth':>10s}  {'RD Estimate':>12s}  {'SE':>8s}  {'p-value':>8s}  {'N':>6s}")
print(f"  {'-'*50}")

for h_val in bandwidths:
    df_h = df[abs(df['R']) <= h_val].copy()
    df_h['T_x_R'] = df_h['T'] * df_h['R']
    X_h = sm.add_constant(df_h[['T', 'R', 'T_x_R']])
    y_h = df_h[Y]
    model_h = sm.OLS(y_h, X_h).fit(cov_type='HC1')
    print(f"  {h_val:>10d}  {model_h.params['T']:>12.4f}  "
          f"{model_h.bse['T']:>8.4f}  {model_h.pvalues['T']:>8.4f}  {len(df_h):>6d}")

# ------------------------------------------------------------
# INTERPRETATION (Section 6):
#   - Examine the "RD Estimate" column: is the sign consistently negative
#     (suggesting Head Start reduces mortality) across all bandwidths?
#   - Look at the p-value column: is the result statistically significant
#     for most bandwidth choices? If it is only significant for one specific
#     bandwidth, the evidence is weak.
#   - The row with h=9 should match the result from Section 4 exactly.
#   - Compare with the rdrobust MSE-optimal bandwidth from Section 5 --
#     ideally, the estimate at that bandwidth should be in the same ballpark
#     as the other bandwidths.
#   - RULE OF THUMB: if the estimate is stable in sign and roughly similar in
#     magnitude for bandwidths ranging from 0.5x to 2x the main bandwidth,
#     your result is robust.
# ------------------------------------------------------------

# ============================================================
# 7. FALSIFICATION TESTS
# ============================================================
#
# WHAT THIS SECTION DOES:
#   Performs a battery of FALSIFICATION (validity) tests to assess whether the
#   RD design assumptions are plausible. A credible RD analysis must pass these
#   tests. If any test fails, it raises doubts about whether the observed
#   discontinuity reflects a genuine treatment effect.
#
# THE CORE RD ASSUMPTION:
#   The RD design assumes that units (counties) cannot precisely MANIPULATE
#   their running variable to sort themselves above or below the cutoff. In
#   this application, counties could not manipulate their 1960 poverty rate
#   to be just above or just below 59.1984% -- the cutoff was determined later
#   by OEO, and 1960 poverty rates were measured before the program existed.
#   This "no manipulation" assumption is fundamentally untestable, but the
#   following tests provide indirect evidence.
#
# THE FOUR FALSIFICATION TESTS:
#
# A. DENSITY TEST (McCrary, 2008):
#    If units can manipulate the running variable, we would expect to see a
#    "bunching" of observations just above (or just below) the cutoff -- an
#    excess mass on the favorable side. This would show up as a DISCONTINUITY
#    in the DENSITY of the running variable at the cutoff.
#    What we do: Plot histograms of the running variable on each side of the
#    cutoff. If the histogram looks smooth across the cutoff (no spike), this
#    supports the no-manipulation assumption.
#    NOTE: A formal McCrary test (or the newer Cattaneo-Jansson-Ma test
#    implemented by rddensity) provides a statistical test with a p-value.
#    Here we do the visual version.
#
# B. COVARIATE BALANCE TEST:
#    If treatment assignment near the cutoff is "as good as random," then
#    PRE-TREATMENT characteristics (covariates determined before the policy)
#    should be SMOOTH (continuous) across the cutoff. There should be NO jump
#    in pre-treatment covariates at the cutoff.
#    What we do: Run the RD regression (same as Section 4) but with each
#    covariate as the outcome instead of child mortality. The coefficient on T
#    should be small and statistically insignificant for each covariate.
#    If a covariate shows a significant jump, it suggests the treatment and
#    control groups differ in observable characteristics, casting doubt on the
#    design.
#
# C. PLACEBO OUTCOME TEST:
#    We test whether outcomes that SHOULD NOT be affected by Head Start also
#    show a discontinuity. For example, mortality among adults (age 25+) from
#    causes related to Head Start's target conditions should not be affected by
#    a preschool program.
#    What we do: Run the RD regression with this "placebo outcome" as the
#    dependent variable. The coefficient on T should be insignificant.
#    If we find a significant effect on an outcome that Head Start could not
#    plausibly affect, it suggests something other than Head Start is causing
#    the discontinuity (e.g., a confounding policy that also uses the same
#    cutoff).
#
# D. PLACEBO CUTOFF TEST:
#    We test whether there are discontinuities at VALUES OF THE RUNNING VARIABLE
#    OTHER THAN THE ACTUAL CUTOFF. If our RD effect is real, it should only
#    appear at the true cutoff (59.1984), not at arbitrary other values.
#    What we do: Choose several "fake" cutoff values (e.g., R = +2, +3, -2, -3
#    in centered terms), restricting to the appropriate side of the real cutoff,
#    and run the RD regression. The coefficient on the fake treatment indicator
#    should be insignificant at each fake cutoff.
#    If we find significant effects at fake cutoffs, it suggests that the
#    running variable has a nonlinear relationship with the outcome that our
#    model is not capturing, rather than a genuine treatment effect.
# ============================================================
print("\n" + "=" * 60)
print("7. FALSIFICATION TESTS")
print("=" * 60)

# --- Density test (McCrary-style) ---
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FALSIFICATION TEST A: DENSITY OF THE RUNNING VARIABLE
#
# We create side-by-side histograms of the running variable (poverty rate)
# below and above the cutoff. The bins are 2 percentage points wide.
#
# KEY PYTHON DETAILS:
#   - np.arange(start, stop, step): creates evenly spaced bin edges.
#   - We create separate bins for the left and right sides of the cutoff
#     so the cutoff is exactly at a bin boundary.
#   - alpha=0.7 makes the bars slightly transparent.
#
# WHAT TO LOOK FOR:
#   - The histogram should look roughly smooth across the cutoff.
#   - There should be no sudden spike or dip in the number of counties
#     just to the left or right of the red dashed line.
#   - If you see a pile-up of counties on one side, it may indicate
#     manipulation of the running variable, which would invalidate the
#     RD design.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--- A. Density Test (Histogram) ---")
fig, ax = plt.subplots(figsize=(10, 5))
bins_left = np.arange(df[X].min(), C, 2)
bins_right = np.arange(C, df[X].max() + 2, 2)
ax.hist(df[df[X] < C][X].dropna(), bins=bins_left, alpha=0.7, color='#4472C4', edgecolor='white', label='Control')
ax.hist(df[df[X] >= C][X].dropna(), bins=bins_right, alpha=0.7, color='#ED7D31', edgecolor='white', label='Treated')
ax.axvline(x=C, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Poverty Rate 1960')
ax.set_ylabel('Frequency')
ax.set_title('Density of Running Variable at Cutoff', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('fig_ch7_density.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# INTERPRETATION (Density test):
#   - If the bars are roughly the same height on both sides near the cutoff,
#     there is no evidence of manipulation. PASS.
#   - If there is an obvious jump in bar heights at the cutoff, you need to
#     investigate further using a formal test (e.g., rddensity from the
#     Cattaneo-Jansson-Ma framework).
#   - In this application, counties could NOT manipulate their 1960 poverty
#     rate after the fact, so we expect the density to be smooth.
# ------------------------------------------------------------

# --- Covariate balance test ---
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FALSIFICATION TEST B: COVARIATE BALANCE AT THE CUTOFF
#
# For each pre-treatment covariate (from the 1960 Census), we run the same
# RD regression as Section 4, but with the covariate as the dependent variable
# instead of child mortality.
#
# MODEL: Covariate_i = alpha + tau*T_i + beta*R_i + gamma*(T_i*R_i) + epsilon_i
#
# The null hypothesis is tau = 0 (no discontinuity in the covariate at cutoff).
# We test up to 5 covariates.
#
# KEY PYTHON DETAILS:
#   - The loop iterates over covariate names.
#   - For each covariate, we create a sub-DataFrame with non-missing values,
#     fit the RD regression, and print the coefficient on T and its p-value.
#   - An asterisk (*) is printed if p < 0.05 (significant at 5% level).
#
# WHAT TO LOOK FOR:
#   - Most (ideally ALL) covariates should show insignificant coefficients
#     on T (p-values > 0.05 or 0.10).
#   - One or two marginally significant results out of many tests could occur
#     by chance (if testing 10 covariates at 5%, we expect 0.5 false positives
#     on average). Do not over-interpret a single significant result.
#   - If MANY covariates show significant jumps, the RD design is suspect.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--- B. Covariate Balance Test ---")
h_test = 9
df_test = df[abs(df['R']) <= h_test].copy()

if covariates:
    print(f"  Testing pre-treatment covariates (h={h_test}):")
    for cov in covariates[:5]:
        df_cov = df_test[[cov, 'T', 'R']].dropna()
        df_cov['T_x_R'] = df_cov['T'] * df_cov['R']
        X_cov = sm.add_constant(df_cov[['T', 'R', 'T_x_R']])
        model_cov = sm.OLS(df_cov[cov], X_cov).fit(cov_type='HC1')
        print(f"    {cov:30s}: coef(T)={model_cov.params['T']:>10.4f}, "
              f"p={model_cov.pvalues['T']:.4f} {'*' if model_cov.pvalues['T'] < 0.05 else ''}")

# ------------------------------------------------------------
# INTERPRETATION (Covariate balance):
#   - If no covariate has a significant coefficient on T, the test PASSES.
#     This means pre-treatment characteristics are smooth at the cutoff,
#     consistent with quasi-random assignment near the cutoff.
#   - If several covariates show significant jumps, counties just above the
#     cutoff differ systematically from counties just below, which undermines
#     the RD identification.
# ------------------------------------------------------------

# --- Placebo outcome ---
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FALSIFICATION TEST C: PLACEBO OUTCOME
#
# We test whether an outcome variable that SHOULD NOT be affected by Head Start
# also shows a discontinuity at the cutoff.
#
# The placebo outcome used here is: mort_age25plus_related_postHS
#   This is mortality among adults age 25+ from causes related to the conditions
#   Head Start targets. Since Head Start is a PRESCHOOL program, it should not
#   affect adult mortality in the short to medium term. If we find a significant
#   RD effect on this variable, it suggests that something other than Head Start
#   is driving the discontinuity.
#
# MODEL: Placebo_Y_i = alpha + tau*T_i + beta*R_i + gamma*(T_i*R_i) + epsilon_i
#
# WHAT TO LOOK FOR:
#   - The coefficient on T should be close to zero and INSIGNIFICANT.
#   - If it is significant, this is a red flag: there may be confounding
#     policies or general health differences at the cutoff.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 'mort_age25plus_related_postHS' in df.columns:
    print("\n--- C. Placebo Outcome Test ---")
    placebo_var = 'mort_age25plus_related_postHS'
    df_plac = df_test[[placebo_var, 'T', 'R']].dropna()
    df_plac['T_x_R'] = df_plac['T'] * df_plac['R']
    X_plac = sm.add_constant(df_plac[['T', 'R', 'T_x_R']])
    model_plac = sm.OLS(df_plac[placebo_var], X_plac).fit(cov_type='HC1')
    print(f"    Placebo outcome ({placebo_var}):")
    print(f"    RD estimate: {model_plac.params['T']:.4f}, p={model_plac.pvalues['T']:.4f}")

# ------------------------------------------------------------
# INTERPRETATION (Placebo outcome):
#   - A large p-value (e.g., > 0.10) means we FAIL TO REJECT the null of
#     no discontinuity in the placebo outcome. This is the desired result --
#     it means the jump we found in child mortality is specific to the
#     population that Head Start serves, not a general artifact.
#   - A significant result here would suggest that the discontinuity is not
#     specific to Head Start's target population, which weakens the causal
#     interpretation.
# ------------------------------------------------------------

# --- Placebo cutoff ---
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FALSIFICATION TEST D: PLACEBO CUTOFFS
#
# We test whether there are discontinuities at FAKE cutoff values. If the
# real RD effect is genuine, it should only appear at the true cutoff (C),
# not at arbitrary other points.
#
# METHODOLOGY:
#   - We choose fake cutoff values in terms of the centered running variable R:
#     fake_c = +2, +3 (on the treated side) and -2, -3 (on the control side).
#   - For positive fake cutoffs (e.g., fake_c = +2), we restrict to observations
#     ABOVE the real cutoff (R > 0) and create a new treatment indicator based
#     on whether R >= fake_c. This tests for a discontinuity within the treated
#     group.
#   - For negative fake cutoffs (e.g., fake_c = -2), we restrict to observations
#     BELOW the real cutoff (R < 0) and similarly create a fake treatment
#     indicator. This tests for a discontinuity within the control group.
#   - We then run the RD regression with these fake variables.
#
# WHY RESTRICT TO ONE SIDE?
#   If we used data from BOTH sides and placed a fake cutoff near the real one,
#   the fake treatment indicator would capture part of the real treatment effect.
#   By staying on one side of the real cutoff, we avoid this contamination.
#
# KEY PYTHON DETAILS:
#   - df[df['R'] > 0].copy() if fake_c > 0: restricts to the treated side.
#   - df_fake['T_fake'] = (df_fake['R'] >= fake_c).astype(int): creates a
#     new treatment indicator at the fake cutoff.
#   - df_fake['R_centered'] = df_fake['R'] - fake_c: re-centers the running
#     variable at the fake cutoff.
#   - The if len(df_fake2) > 20 guard ensures we have enough data for the
#     regression.
#
# WHAT TO LOOK FOR:
#   - The coefficients on T_fake should be insignificant (p > 0.05) for all
#     fake cutoffs.
#   - If a fake cutoff shows a significant effect, it suggests the outcome
#     trend is not well captured by our linear model, and the "jump" at the
#     real cutoff might be an artifact of model misspecification.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\n--- D. Placebo Cutoff Test ---")
for fake_c in [2, 3, -2, -3]:
    df_fake = df[df['R'] > 0].copy() if fake_c > 0 else df[df['R'] < 0].copy()
    df_fake['T_fake'] = (df_fake['R'] >= fake_c).astype(int)
    df_fake['R_centered'] = df_fake['R'] - fake_c
    df_fake['T_x_R'] = df_fake['T_fake'] * df_fake['R_centered']
    df_fake2 = df_fake[[Y, 'T_fake', 'R_centered', 'T_x_R']].dropna()
    if len(df_fake2) > 20:
        X_fake = sm.add_constant(df_fake2[['T_fake', 'R_centered', 'T_x_R']])
        model_fake = sm.OLS(df_fake2[Y], X_fake).fit(cov_type='HC1')
        print(f"    Cutoff at R={fake_c:+d}: estimate={model_fake.params['T_fake']:>8.4f}, "
              f"p={model_fake.pvalues['T_fake']:.4f}")

# ------------------------------------------------------------
# INTERPRETATION (Placebo cutoffs):
#   - If none of the fake cutoffs produce significant estimates, the test
#     PASSES. This supports the claim that the discontinuity is specific to
#     the actual policy cutoff (59.1984%).
#   - If a fake cutoff IS significant, it suggests the outcome variable has
#     kinks or jumps at other values of the running variable, which could
#     confound our RD estimate. In that case, consider using a more flexible
#     functional form (e.g., higher-order polynomial) or a narrower bandwidth.
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("END OF CHAPTER 7")
print("=" * 60)
