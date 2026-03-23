* MIT License
* Copyright (c) 2026 Tiraphap Fakthong
* See LICENSE file for full license text.

cap log close

* To run you need files
*   panel.dta
* in your directory

********** SETUP **********

set more off
version 11
clear all
set memory 30m
set linesize 90
set scheme s1mono  /* Graphics scheme */

********** DATA DESCRIPTION **********

* panel.dta
* PSID. Same as Stata website file psidextract.dta
* Data due to  Baltagi and Khanti-Akom (1990) 
* This is corrected version of data in Cornwell and Rupert (1988).
* 595 individuals for years 1976-82

*******  PANEL-DATA SUMMARY

* Read in dataset and describe
use panel.dta, clear
describe

* Summary of dataset
summarize

* Organization of dataset
list id t exp wks occ in 1/3, clean

* Declare individual identifier and time identifier
xtset id t

* Panel description of dataset
xtdescribe 

* Panel summary statistics: within and between variation
xtsum id t lwage ed exp exp2 wks south tdum1

* Panel tabulation for a variable
xttab south

* Transition probabilities for a variable
xttrans south, freq

* Simple time-series plot for each of 20 individuals
quietly xtline lwage if id<=20, overlay legend(off) saving(lwage, replace)
quietly xtline wks if id<=20, overlay legend(off) saving(wks, replace)
graph combine lwage.gph wks.gph, iscale(1)
quietly graph export timeseriesplot.eps, replace

* Scatterplot, quadratic fit and nonparametric regression (lowess)
graph twoway (scatter lwage exp, msize(small) msymbol(o))              ///
  (qfit lwage exp, clstyle(p3) lwidth(medthick))                       ///
  (lowess lwage exp, bwidth(0.4) clstyle(p1) lwidth(medthick)),        ///
  plotregion(style(none))                                              ///
  title("Overall variation: Log wage versus experience")               ///
  xtitle("Years of experience", size(medlarge)) xscale(titlegap(*5))   /// 
  ytitle("Log hourly wage", size(medlarge)) yscale(titlegap(*5))       ///
  legend(pos(4) ring(0) col(1)) legend(size(small))                    ///
  legend(label(1 "Actual Data") label(2 "Quadratic fit") label(3 "Lowess"))
graph export scatterplot.eps, replace

* Scatterplot for within variation
preserve
xtdata, fe
graph twoway (scatter lwage exp) (qfit lwage exp) (lowess lwage exp),  ///
  plotregion(style(none)) title("Within variation: Log wage versus experience")
restore
graph export withinplot.eps, replace

* Pooled OLS with cluster-robust standard errors
use panel.dta, clear
regress lwage exp exp2 wks ed, vce(cluster id)

* Pooled OLS with incorrect default standard errors
regress lwage exp exp2 wks ed 

* First-order autocorrelation in a variable
sort id t  
correlate lwage L.lwage

* Autocorrelations of residual 
quietly regress lwage exp exp2 wks ed, vce(cluster id)
predict uhat, residuals
forvalues j = 1/6 {
     quietly corr uhat L`j'.uhat
     display "Autocorrelation at lag `j' = " %6.3f r(rho) 
     }

* First-order autocorrelation differs in different year pairs
forvalues s = 2/7 {
     quietly corr uhat L1.uhat if t == `s'
     display "Autocorrelation at lag 1 in year `s' = " %6.3f r(rho) 
     }

******* POOLED OR POPULATION-AVERAGED ESTIMATORS

* Population-averaged or pooled FGLS estimator with AR(2) error
xtreg lwage exp exp2 wks ed, pa corr(ar 2) vce(robust) nolog

* Estimated error correlation matrix after xtreg, pa
matrix list e(R)

******* WITHIN ESTIMATOR

* Within or FE estimator with cluster-robust standard errors
xtreg lwage exp exp2 wks ed, fe vce(cluster id)

* LSDV model fit using areg with cluster-robust standard errors
areg lwage exp exp2 wks ed, absorb(id) vce(cluster id)

* LSDV model fit using factor variables with cluster-robust standard errors
set matsize 800
quietly regress lwage exp exp2 wks ed i.id, vce(cluster id)
estimates table, keep(exp exp2 wks ed _cons) b se b(%12.7f)

******* BETWEEN ESTIMATOR

* Between estimator with default standard errors
xtreg lwage exp exp2 wks ed, be

// Following gives heteroskedasrtic-robust se's for between estimator
xtreg lwage exp exp2 wks ed, be vce(boot, reps(400) seed(10101) nodots)

******* RANDOM EFFECTS ESTIMATORS

* Random-effects estimator with cluster-robust standard errors
xtreg lwage exp exp2 wks ed, re vce(cluster id) theta

* Calculate theta
display "theta = "  1 - sqrt(e(sigma_e)^2 / (7*e(sigma_u)^2+e(sigma_e)^2))

******* COMPARISON OF ESTIMATORS

use panel.dta, clear

* Compare OLS, BE, FE, RE estimators, and methods to compute standard errors
global xlist exp exp2 wks ed 
quietly regress lwage $xlist, vce(cluster id)
estimates store OLS_rob
quietly xtreg lwage $xlist, be
estimates store BE
quietly xtreg lwage $xlist, fe 
estimates store FE
quietly xtreg lwage $xlist, fe vce(robust)
estimates store FE_rob
quietly xtreg lwage $xlist, re
estimates store RE
quietly xtreg lwage $xlist, re vce(robust)
estimates store RE_rob
estimates table OLS_rob BE FE FE_rob RE RE_rob,  ///
  b se stats(N r2 r2_o r2_b r2_w sigma_u sigma_e rho) b(%7.4f)

* Hausman test assuming RE estimator is fully efficient under null hypothesis
hausman FE RE, sigmamore

* Robust Hausman test using method of Wooldridge (2002)
quietly xtreg lwage $xlist, re
scalar theta = e(theta)
global yandxforhausman lwage exp exp2 wks ed
sort id
foreach x of varlist $yandxforhausman {
  by id: egen mean`x' = mean(`x')
  generate md`x' = `x' - mean`x'
  generate red`x' = `x' - theta*mean`x'
  }
quietly regress redlwage redexp redexp2 redwks reded mdexp mdexp2 mdwks, vce(cluster id)
test mdexp mdexp2 mdwks

* Prediction after OLS and RE estimation
quietly regress lwage exp exp2 wks ed, vce(cluster id)
predict xbols, xb
quietly xtreg lwage exp exp2 wks ed, re  
predict xbre, xb
predict xbure, xbu
summarize lwage xbols xbre xbure
correlate lwage xbols xbre xbure

*******  FIRST DIFFERENCE ESTIMATOR

sort id t
* First-differences estimator with cluster-robust standard errors
regress D.(lwage exp exp2 wks ed), vce(cluster id) noconstant


********** CLOSE OUTPUT
