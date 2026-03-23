* MIT License
* Copyright (c) 2026 Tiraphap Fakthong
* See LICENSE file for full license text.
* Read data, define global x2list, and summarize data
global x2list totchr age female blhisp linc 
summarize ldrugexp hi_empunion $x2list

* Summarize available instruments 
summarize ssiratio lowincome multlc firmsz if linc!=.

* IV estimation of a just-identified model with single endog regressor
ivregress 2sls ldrugexp (hi_empunion = ssiratio) $x2list, vce(robust) first

* Compare 1 Just + 5 estimators and variance estimates for overidentified models

global ivmodel "ldrugexp (hi_empunion = ssiratio multlc) $x2list"
quietly ivregress 2sls ldrugexp (hi_empunion = ssiratio) $x2list, vce(robust)
estimates store TwoSLSJust
quietly ivregress 2sls $ivmodel, vce(robust)
estimates store TwoSLS
quietly ivregress gmm  $ivmodel, wmatrix(robust) 
estimates store GMM_het
quietly ivregress gmm  $ivmodel, wmatrix(robust) igmm
estimates store GMM_igmm
quietly ivregress gmm  $ivmodel, wmatrix(cluster age) 
estimates store GMM_clu
quietly ivregress 2sls  $ivmodel
estimates store TwoSLS_def
estimates table TwoSLSJust TwoSLS GMM_het GMM_igmm GMM_clu TwoSLS_def, b(%9.5f) se  

* Obtain OLS estimates to compare with preceding IV estimates
regress ldrugexp hi_empunion $x2list, vce(robust) 

* Robust Durbin-Wu-Hausman test of endogeneity implemented by estat endogenous
ivregress 2sls ldrugexp (hi_empunion = ssiratio) $x2list, vce(robust)
estat endogenous

* Robust Durbin-Wu-Hausman test of endogeneity implemented manually
quietly regress hi_empunion ssiratio $x2list
quietly predict v1hat, resid
quietly regress ldrugexp hi_empunion v1hat $x2list, vce(robust)
test v1hat 

* Test of overidentifying restrictions following ivregress gmm
quietly ivregress gmm ldrugexp (hi_empunion = ssiratio multlc) $x2list, wmatrix(robust) 
estat overid

* Test of overidentifying restrictions following ivregress gmm
ivregress gmm ldrugexp (hi_empunion = ssiratio lowincome multlc firmsz) $x2list, wmatrix(robust) 
estat overid

* Regression with a dummy variable regressor
treatreg ldrugexp $x2list, treat(hi_empunion = ssiratio $x2list)

********** WEAK INSTRUMENTS

* Correlations of endogenous regressor with instruments
correlate hi_empunion ssiratio lowincome multlc firmsz if linc!=.

* Weak instrument tests - just-identified model
quietly ivregress 2sls ldrugexp (hi_empunion = ssiratio) $x2list, vce(robust)
estat firststage, forcenonrobust all  

* Weak instrument tests - two or more overidentifying restrictions
quietly ivregress gmm ldrugexp (hi_empunion = ssiratio lowincome multlc firmsz) $x2list, vce(robust)
estat firststage, forcenonrobust

* Compare 4 just-identified model estimates with different instruments
quietly regress ldrugexp hi_empunion $x2list, vce(robust)
estimates store OLS0
quietly ivregress 2sls ldrugexp (hi_empunion=ssiratio) $x2list, vce(robust)
estimates store IV_INST1
quietly estat firststage, forcenonrobust
scalar me1 = r(mineig)
quietly ivregress 2sls ldrugexp (hi_empunion=lowincome) $x2list, vce(robust)
estimates store IV_INST2
quietly estat firststage, forcenonrobust
scalar me2 = r(mineig)
quietly ivregress 2sls ldrugexp (hi_empunion=multlc) $x2list, vce(robust) 
estimates store IV_INST3
quietly estat firststage, forcenonrobust
scalar me3 = r(mineig)
quietly ivregress 2sls ldrugexp (hi_empunion=firmsz) $x2list, vce(robust)
estimates store IV_INST4
quietly estat firststage, forcenonrobust
scalar me4 = r(mineig)
estimates table OLS0 IV_INST1 IV_INST2 IV_INST3 IV_INST4, b(%8.4f) se  
display "Minimum eigenvalues are:     " me1 _s(2) me2 _s(2) me3 _s(2) me4

********** BETTER INFERENCE WITH WEAK INSTRUMENTS

* Variants of IV Estimators: 2SLS, LIML, JIVE, GMM_het, GMM-het using IVREG2
global ivmodel "ldrugexp (hi_empunion = ssiratio lowincome multlc firmsz) $x2list"
quietly ivregress 2sls $ivmodel, vce(robust)
estimates store TWOSLS
quietly ivregress liml $ivmodel, vce(robust)
estimates store LIML
quietly jive $ivmodel, robust
estimates store JIVE
quietly ivregress gmm $ivmodel, wmatrix(robust) 
estimates store GMM_het
quietly ivreg2 $ivmodel, gmm2s robust
estimates store IVREG2
estimates table TWOSLS LIML JIVE GMM_het IVREG2, b(%7.4f) se 

