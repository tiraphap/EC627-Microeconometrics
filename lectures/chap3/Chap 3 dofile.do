* MIT License
* Copyright (c) 2026 Tiraphap Fakthong
* See LICENSE file for full license text.
cap log close

********** SETUP **********

version 11
clear all
set memory 10m
set more off
set scheme s1mono   /* Used for graphs */
  
********** MODELING HETEROSKEDASTIC DATA

* This uses generated data
* Model is  y = 1 + 1*x2 + 1*x3 + u
* where     u = sqrt(exp(-1+0.2*x2))*e
*           x1 ~ N(0, 5^2)
*           x2 ~ N(0, 5^2)
*           e ~ N(0, 5^2)
* Errors are conditionally heteroskedastic with V[u|x]=exp(-1+1*x2)

* Generated data for heteroskedasticity example
set seed 10101
quietly set obs 500
generate double x2 = 5*rnormal(0)
generate double x3 = 5*rnormal(0)
generate double e  = 5*rnormal(0)
generate double u  = sqrt(exp(-1+0.2*x2))*e
generate double y  = 1 + 1*x2 + 1*x3 + u
summarize

* OLS regression with default standard errors
regress y x2 x3

* OLS regression with heteroskedasticity-robust standard errors
regress y x2 x3, robust

* Heteroskedasticity diagnostic scatterplot
quietly regress y x2 x3
predict double uhat, resid
generate double absu = abs(uhat)
quietly twoway (scatter absu x2) (lowess absu x2, bw(0.4) lw(thick)), ///
  scale(1.2) xscale(titleg(*5)) yscale(titleg(*5))                    ///
  plotr(style(none)) name(gls1)
quietly twoway (scatter absu x3) (lowess absu x3, bw(0.4) lw(thick)), ///
  scale(1.2) xscale(titleg(*5)) yscale(titleg(*5))                    ///
  plotr(style(none)) name(gls2)
graph combine gls1 gls2
graph export gls_fig1.eps, replace
drop uhat

* Test heteroskedasticity depending on x2, x3, and x2 and x3
estat hettest x2 x3, mtest

* Separate tests of heteroskedasticity using iid version of hettest
estat hettest x2, iid
estat hettest x3, iid
estat hettest x2 x3, iid

* FGLS: First step get estimate of skedasticity function
quietly regress y x2 x3                  // get b-ols estimator
predict double uhat, resid
generate double uhatsq = uhat^2          // get squared residual 
generate double one = 1                       
nl (uhatsq = exp({xb: x2 one})), nolog   // NLS of uhatsq on exp(z'a), alpha1(=> 2.22)*1 + alpha2(==>0.2)*X2
predict double varu, yhat                // get sigmahat^2

* FGLS: Second step get estimate of skedasticity function
regress y x2 x3 [aweight=1/varu]         // w = 1/varu

* WLS estimator is FGLS with robust estimate of VCE
regress y x2 x3 [aweight=1/varu], robust    // Adjust the FGLS-VCE with the working variance matrix (Sigma matrix)

* FGLS: First step get estimate of skedasticity function
quietly regress y x2 x3 if _n < 451
quietly predict double yhat1
quietly regress y x2 x3 [aweight=1/varu] if _n < 451
quietly predict double yhat2
generate double perr1 = y - yhat1
generate double perr2 = y - yhat2
summarize y yhat1 yhat2 perr1 perr2 if _n > 450

********** SYSTEMS OF LINEAR REGRESSIONS

* Summary statistics for seemingly unrelated regressions example
clear all
use surdata.dta
summarize ldrugexp ltotothr age age2 educyr actlim totchr medicaid private

summarize ldrugexp if ldrugexp!=. & ltotothr!=.          

* SUR estimation of a seemingly unrelated regressions model - FGLS
sureg (ldrugexp age age2 actlim totchr medicaid private) ///
  (ltotothr age age2 educyr actlim totchr private), corr

* Bootstrap to get heteroskedasticity-robust SEs for SUR estimator - WLS which is the FGLS with robust SE
bootstrap, reps(400) seed(10101) nodots: sureg        ///
  (ldrugexp age age2 actlim totchr medicaid private)  ///
  (ltotothr age age2 educyr actlim totchr private) 

* Test of variables in both equations
quietly sureg (ldrugexp age age2 actlim totchr medicaid private) ///
   (ltotothr age age2 educyr actlim totchr private)
test age age2

* Test of variables in just the first equation
test [ldrugexp]age [ldrugexp]age2

* Test of a restriction across the two equations
test [ldrugexp]private = [ltotothr]private

* Specify a restriction across the two equations
constraint 1 [ldrugexp]private = [ltotothr]private

* Estimate subject to the cross-equation constraint
sureg (ldrugexp age age2 actlim totchr medicaid private)        ///
  (ltotothr age age2 educyr actlim totchr private), constraints(1) 

******** SURVEY DATA: WEIGHTING, CLUSTERING, AND STRATIFICATION

* Data from http://www.stata-press.com/data/r10/nhanes2.dta

* Survey data example: NHANES II data
clear all
use Chap 3 nhanes2.dta
quietly keep if age >= 21 & age <= 65
describe sampl finalwgt strata psu
summarize sampl finalwgt strata psu

* Declare survey design
svyset psu [pweight=finalwgt], strata(strata)

* Describe the survey design
svydescribe

* Estimate the population mean using svy: prefix
svy: mean hgb

* Estimate the population mean using no weights and no cluster
mean hgb

* Regression using svy:
svy: regress hgb age female

* Regression using weights and cluster on PSU
generate uniqpsu = 2*strata + psu  // make unique identifier for each psu
regress hgb age female [pweight=finalwgt], vce(cluster uniqpsu)

* Regression using no weights and no cluster
regress hgb age female

********** CLOSE OUTPUT **********
