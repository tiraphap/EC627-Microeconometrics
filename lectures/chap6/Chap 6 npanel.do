* MIT License
* Copyright (c) 2026 Tiraphap Fakthong
* See LICENSE file for full license text.

cap log close

********** SETUP **********

set more off
version 11
clear all
set linesize 90
set scheme s1mono  /* Graphics scheme */

********** DATA DESCRIPTION **********

* Rand Health Insurance Experiment data 
* Essentially same data as in P. Deb and P.K. Trivedi (2002)
* "The Structure of Demand for Medical Care: Latent Class versus
* Two-Part Models", Journal of Health Economics, 21, 601-625
* except that paper used different outcome (counts rather than $)
* Each observation is for an individual over a year.
* Individuals may appear in up to five years.
* All available sample is used except only fee for service plans included.
* If panel data used then clustering is on id (person id)

********** READ DATA **********

use npanel.dta, clear

**********  NONLINEAR PANEL-DATA EXAMPLE

* Describe dependent variables and regressors
use npanel.dta, clear
describe dmdu med mdu lcoins ndisease female age lfam child id year

* Summarize dependent variables and regressors
summarize dmdu med mdu lcoins ndisease female age lfam child id year

* Panel description of dataset 
xtset id year
xtdescribe

* Panel summary of time-varying regressors
xtsum age lfam child

********** BINARY OUTCOME MODELS

* Panel summary of dependent variable
xtsum dmdu

* Year-to-year transitions in whether visit doctor
xttrans dmdu

* Correlations in the dependent variable
corr dmdu l.dmdu l2.dmdu

* Logit cross-section with panel-robust standard errors
logit dmdu lcoins ndisease female age lfam child, vce(cluster id) nolog

* Pooled logit cross-section with exchangeable errors and panel-robust VCE
xtlogit dmdu lcoins ndisease female age lfam child, pa corr(exch) vce(robust) nolog

* Logit random-effects estimator
xtlogit dmdu lcoins ndisease female age lfam child, re nolog

* Logit fixed-effects estimator
xtlogit dmdu lcoins ndisease female age lfam child, fe nolog 

* Logit mixed-effects estimator (same as xtlogit, re)
* xtmelogit dmdu lcoins ndisease female age lfam child || id:

* Panel logit estimator comparison
global xlist lcoins ndisease female age lfam child
quietly logit dmdu $xlist, vce(cluster id)
estimates store POOLED
quietly xtlogit dmdu $xlist, pa corr(exch) vce(robust)
estimates store PA
quietly xtlogit dmdu $xlist, re     // SEs are not cluster-robust
estimates store RE
quietly xtlogit dmdu $xlist, fe     // SEs are not cluster-robust
estimates store FE
estimates table POOLED PA RE FE, equations(1) se b(%8.4f) stats(N ll) stfmt(%8.0f)

********** PANEL TOBIT MODELS

* Panel summary of dependent variable
xtsum med

* Tobit random-effects estimator
xttobit med lcoins ndisease female age lfam child, ll(0) nolog

********** PANEL COUNT MODELS

* Panel summary of dependent variable
xtsum mdu

* Year-to-year transitions in doctor visits
generate mdushort = mdu
replace mdushort = 4 if mdu >= 4
xttrans mdushort

corr mdu L.mdu

* Pooled Poisson estimator with cluster-robust standard errors
poisson mdu lcoins ndisease female age lfam child, vce(cluster id)

* Poisson PA estimator with unstructured error correlation and robust VCE
xtpoisson mdu lcoins ndisease female age lfam child, pa corr(unstr) vce(robust)

* Poisson random-effects estimator with default standard errors
xtpoisson mdu lcoins ndisease female age lfam child, re

* Poisson random-effects estimator with cluster-robust standard errors
xtpoisson mdu lcoins ndisease female age lfam child, re vce(boot, reps(400) seed(10101) nodots)

* Poisson random-effects estimator with normal intercept and default standard errors
xtpoisson mdu lcoins ndisease female age lfam child, re normal
estimates store poisrenormal

* Poisson random-effects estimator with normal intercept and normal slope for one parameter
* xtmepoisson mdu lcoins ndisease female age lfam child || id: NDISEASE
* estimates store poisrenormrob

* Poisson fixed-effects estimator with default standard errors
xtpoisson mdu lcoins ndisease female age lfam child, fe i(id)  

* Poisson fixed-effects estimator with cluster-robust standard errors
xtpoisson mdu lcoins ndisease female age lfam child, fe vce(boot, reps(400) seed(10101) nodots)

* Comparison of Poisson panel estimators
quietly xtpoisson mdu lcoins ndisease female age lfam child, pa corr(unstr) vce(robust)
estimates store PPA_ROB
quietly xtpoisson mdu lcoins ndisease female age lfam child, re
estimates store PRE
quietly xtpoisson mdu lcoins ndisease female age lfam child, re normal
estimates store PRE_NORM
quietly xtpoisson mdu lcoins ndisease female age lfam child, fe
estimates store PFE
estimates table PPA_ROB PRE PRE_NORM PFE, equations(1) b(%8.4f) se stats(N ll) stfmt(%8.0f)

* Negative binomial pooled estimator with default standard errors
nbreg mdu lcoins ndisease female age lfam child

* Negative binomial pooled estimator with het robust standard errors
nbreg mdu lcoins ndisease female age lfam child, vce(robust)

* Negative binomial pooled estimator with cluster-robust standard errors
nbreg mdu lcoins ndisease female age lfam child, cluster(id)

* Negative binomial population-averaged estimator with equicorrelated errors
xtnbreg lcoins ndisease female age lfam child, pa corr(exch) vce(robust)

* Negative binomial random-effects estimator with default standard errors
xtnbreg mdu ndisease female age lfam child, re i(id)  

* Negative binomial fixed-effects estimator with default standard errors
xtnbreg mdu ndisease female age lfam child, fe i(id)  

* Comparison of negative binomial panel estimators
quietly xtpoisson mdu lcoins ndisease female age lfam child, pa corr(exch) vce(robust)
estimates store PPA_ROB
quietly xtnbreg mdu lcoins ndisease female age lfam child, pa corr(exch) vce(robust)
estimates store NBPA_ROB
quietly xtnbreg mdu lcoins ndisease female age lfam child, re
estimates store NBRE
quietly xtnbreg mdu lcoins ndisease female age lfam child, fe
estimates store NBFE
estimates table PPA_ROB NBPA_ROB NBRE NBFE, equations(1) b(%8.4f) se stats(N ll) stfmt(%8.0f)

********** CLOSE OUTPUT
