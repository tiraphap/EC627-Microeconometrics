cap log close

********** SETUP **********

set more off
version 11
clear all
set linesize 82
set scheme s1mono  /* Graphics scheme */


************ DATA SUMMARY STATISTICS

* Variable description for medical expenditure dataset
describe totexp ltotexp posexp suppins phylim actlim totchr age female income

* Summary statistics for medical expenditure dataset
summarize totexp ltotexp posexp suppins phylim actlim totchr age female income

* Tabulate variable
tabulate income if income <= 0

* Detailed summary statistics of a single variable
summarize totexp, detail

* Two-way table of frequencies
table female totchr

* Two-way table with row and column percentages and Pearson chi-squared
tabulate female suppins, row col chi2

* Three-way table of frequencies
table female totchr suppins

* One-way table of summary statistics
table female, contents(N totchr mean totchr sd totchr p50 totchr)

* Two-way table of summary statistics
table female suppins, contents(N totchr mean totchr)

* Summary statistics obtained using command tabstat
tabstat totexp ltotexp, stat (count mean p50 sd skew kurt) col(stat)

* Kernel density plots with adjustment for highly skewed data
kdensity totexp if posexp==1, generate (kx1 kd1) n(500) 
graph twoway (line kd1 kx1) if kx1 < 40000, name(levels)
kdensity ltotexp if posexp==1, generate (kx2 kd2) n(500) 
graph twoway (line kd2 kx2) if kx2 < ln(40000), name(logs)
graph combine levels logs, iscale(1.0)
graph export fig1.eps, replace

*********** BASIC REGRESSION ANALYSIS

* Pairwise correlations for dependent variable and regressor variables
correlate ltotexp suppins phylim actlim totchr age female income

* OLS regression with heteroskedasticity-robust standard errors
regress ltotexp suppins phylim actlim totchr age female income, vce(robust)

* Display stored results and list available postestimation commands
ereturn list
help regress postestimation

* Wald test of equality of coefficients
quietly regress ltotexp suppins phylim actlim totchr age female ///
  income, vce(robust)
test phylim = actlim

*  Joint test of statistical significance of several variables
test phylim actlim totchr

* Store and then tabulate results from multiple regressions
quietly regress ltotexp suppins phylim actlim totchr age female income, vce(robust)
estimates store REG1
quietly regress ltotexp suppins phylim actlim totchr age female educyr, vce(robust)
estimates store REG2
estimates table REG1 REG2, b(%9.4f) se stats(N r2 F ll) keep(suppins income educyr)

* Tabulate results using user-written command esttab to produce cleaner output
esttab REG1 REG2, b(%10.4f) se scalars(N r2 F ll) mtitles ///
  keep(suppins income educyr) title("Model comparison of REG1-REG2")

* Write tabulated results to a file in latex table format
quietly esttab REG1 REG2 using table.tex, replace b(%10.4f) se scalars(N r2 F ll) ///
   mtitles keep(suppins age income educyr _cons) title("Model comparison of REG1-REG2")

* Add a user-calculated statistic to the table
estimates drop REG1 REG2
quietly regress ltotexp suppins phylim actlim totchr age female ///
  income, vce(robust)
estadd scalar pvalue = Ftail(e(df_r),e(df_m),e(F))
estimates store REG1
quietly regress ltotexp suppins phylim actlim totchr age female ///
  educyr, vce(robust)
estadd scalar pvalue = Ftail(e(df_r),e(df_m),e(F))
estimates store REG2
esttab REG1 REG2, b(%10.4f) se scalars(F pvalue) mtitles keep(suppins) 

* Factor variables for sets of indicator variables and interactions
regress ltotexp suppins phylim actlim totchr age female c.income ///
 i.famsze c.income#i.famsze, vce(robust) noheader allbaselevels

* Test joint significance of sets of indicator variables and interactions
testparm i.famsz c.income#i.famsze

* Compute the average marginal effect in model with interactions
quietly regress totexp suppins phylim actlim totchr age female c.income ///
 i.famsze c.income#i.famsze, vce(robust) noheader allbaselevels
margins, dydx(income)

* Compute elasticity for a specified regressor
quietly regress totexp suppins phylim actlim totchr age female income, vce(robust)
margins, eyex(totchr) atmean

********** SPECIFICATION ANALYSIS

* Plot of residuals against fitted values
quietly regress ltotexp suppins phylim actlim totchr age female income, ///
  vce(robust)
rvfplot
graph export fig2.eps, replace

* Details on the outlier residuals
predict uhat, residual
predict yhat, xb
list totexp ltotexp yhat uhat if uhat < -5, clean

* Compute dfits that combines outliers and leverage
quietly regress ltotexp suppins phylim actlim totchr age female income
predict dfits, dfits
scalar threshold = 2*sqrt((e(df_m)+1)/e(N))
display "dfits threshold = "  %6.3f threshold
tabstat dfits, stat (min p1 p5 p95 p99 max) format(%9.3f) col(stat)
list dfits totexp ltotexp yhat uhat if abs(dfits) > 2*threshold & e(sample), clean

* Boxcox model with lhs variable transformed
boxcox totexp suppins phylim actlim totchr age female income if totexp>0, nolog

* Variable augmentation test of conditional mean using estat ovtest
quietly regress ltotexp suppins phylim actlim totchr age female ///
  income, vce(robust)
estat ovtest

* Link test of functional form of conditional mean 
quietly regress ltotexp suppins phylim actlim totchr age female ///
  income, vce(robust)
linktest

* Heteroskedasticity tests using estat hettest and option iid
quietly regress ltotexp suppins phylim actlim totchr age female income
estat hettest, iid
estat hettest suppins phylim actlim totchr age female income, iid

* Information matrix test
quietly regress ltotexp suppins phylim actlim totchr age female income
estat imtest

* Simulation to show tests have power in more than one direction
clear all
set obs 50 
set seed 10101
generate x = runiform()                  // x ~ uniform(0,1)
generate u = rnormal()                   // u ~ N(0,1)
generate y = exp(1 + 0.25*x + 4*x^2) + u
generate xsq = x^2
regress y x xsq

* Test for heteroskedasticity
estat hettest

* Test for misspecified conditional mean
estat ovtest

******* PREDICTION

* Change dependent variable to level of positive medical expenditures
keep if totexp > 0   
regress totexp suppins phylim actlim totchr age female income, vce(robust)

* Prediction in model linear in levels
predict yhatlevels
summarize totexp yhatlevels

* Compare median prediction and median actual value
tabstat totexp yhatlevels, stat (count p50) col(stat)

* Compute standard errors of prediction and forecast with default VCE
quietly regress totexp suppins phylim actlim totchr age female income
predict yhatstdp, stdp
predict yhatstdf, stdf
summarize yhatstdp yhatstdf

* Prediction in levels from a logarithmic model
quietly regress ltotexp suppins phylim actlim totchr age female income
quietly predict lyhat
generate yhatwrong = exp(lyhat)
generate yhatnormal = exp(lyhat)*exp(0.5*e(rmse)^2)
quietly predict uhat, residual
generate expuhat = exp(uhat)
quietly summarize expuhat
generate yhatduan = r(mean)*exp(lyhat) 
summarize totexp yhatwrong yhatnormal yhatduan yhatlevels 

* Predicted effect of supplementary insurance: methods 1 and 2
*1. Difference in sample means (mean(y1)-mean(y0)) - This method does not control for individual characteristics
*2. Difference in predicted means (mean(yhat1)-mean(yhat0)) - This method does control for individual characteristics
bysort suppins: summarize totexp yhatlevels yhatduan
*3. We have 2 models 
*3.1 level-regression: regress totexp suppins other Xs --> B hat of the suppins = 725 - This method does control for individual characteristics

*3.2 log-lin regression:
* Predicted effect of supplementary insurance: method 3 for log-linear model
* suppins is exogeneously determined -> Our policy variable
quietly regress ltotexp suppins phylim actlim totchr age female income
preserve
quietly replace suppins = 1
quietly predict lyhat1
generate yhatnormal1 = exp(lyhat1)*exp(0.5*e(rmse)^2) 
quietly replace suppins = 0
quietly predict lyhat0 
generate yhatnormal0 = exp(lyhat0)*exp(0.5*e(rmse)^2) 
generate treateffect = yhatnormal1 - yhatnormal0 // Average treatment effect (ATE)
summarize yhatnormal1 yhatnormal0 treateffect 
restore

******* SAMPLING WEIGHTS

*pweight / pw variable = probability weight 
*ex. pw = 1,000 for an observation
*obs      pw
* 1       1,000 ----> This obs represents 1,000 pp
* 2       500   ---->                       500 pp 
* 3       50    ---->                        50 pp    

* Create artificial sampling weights
generate swght = totchr^2 + 0.5 // ---> We are oversampling those with few number of chronic illness
summarize swght

* Calculate the weighted mean
mean totexp [pw = swght]
mean totexp

* Perform weighted regression 
regress totexp suppins phylim actlim totchr age female income [pweight=swght]
regress totexp suppins phylim actlim totchr age female income 

* Weighted prediction
quietly predict yhatwols
mean yhatwols [pweight=swght], noheader  
mean yhatwols, noheader      // unweighted prediction

********** CLOSE OUTPUT

