********** SETUP **********

set more off
version 11
clear all
set memory 10m
set scheme s1mono  /* Graphics scheme */

********** PSEUDORANDOM NUMBER GENERATORS: INTRODUCTION

* Single draw of a uniform number 
set seed 10101
scalar u = runiform()
display u

* 10000 draws of uniform numbers
set obs 10000
set seed 10101
gen x = runiform()
list x in 1/20, clean
summarize x

// Output not included in book
histogram x, start(0) width(0.1)

//Now we know that the distribution of x is identical.
//We want to know whether x are independent.
* First three autocorrelations for the uniform draws
generate t = _n
tsset t
pwcorr x L1.x L2.x L3.x, star(0.01)    //L1. = one-period lag of ...
//iid
** rv => identical and independent distributed
  
// Output not included in book
ac x      // Autocorrelations with 95% confidence band

* normal and uniform
clear all
quietly set obs 1000
set seed 10101                           // set the seed
generate uniform = runiform()            // uniform(0,1)
generate stnormal = rnormal()            // N(0,1) 0 is mean and 1 is standard deviation, not variance
generate norm5and2 = rnormal(5,2)    
tabstat uniform stnormal norm5and2, stat(mean sd skew kurt min p50 max) col(stat)

* t, chi-squared, and F with constant degrees of freedom
clear all
quietly set obs 2000
set seed 10101
generate xt = rt(10)               // result xt ~ t(10)
generate xc = rchi2(10)            // result xc ~ chisquared(10) 
generate xfn = rchi2(10)/10        // result numerator of F(10,5)
generate xfd = rchi2(5)/5          // result denominator of F(10,5)
generate xf = xfn/xfd              // result xf ~ F(10,5) 
summarize xt xc xf

* Discrete rv's: binomial
set seed 10101
generate p1 = runiform()              // here p1~uniform(0,1) 
generate trials = ceil(10*runiform()) // here # trials varies btwn 1 & 10
generate xbin = rbinomial(trials,p1)  // draws from binomial(n,p1)  
summarize p1 trials xbin

* Discrete rv's: independent poisson and negbin draws
*Draws are independent but not identically distributed
set seed 10101 
generate xb = 4 + 2*runiform()
generate xg = rgamma(1,1)           // draw from gamma;E(v)=1
generate xbh = xb*xg                // apply multiplicative heterogeneity
generate xp = rpoisson(5)           // result xp ~ Poisson(5)
generate xp1 = rpoisson(xb)         // result xp1 ~ Poisson(xb)
generate xp2 = rpoisson(xbh)        // result xp2 ~ NB(xb)
summarize xg xb xp xp1 xp2

* Example of histogram and kernel density plus graph combine
quietly twoway (histogram xc, width(1)) (kdensity xc, lwidth(thick)), ///
   title("Draws from chisquared(10)") 
quietly graph save cdistr.gph, replace
quietly twoway (histogram xp, discrete) (kdensity xp, lwidth(thick) ///
   w(1)),  title("Draws from Poisson(mu) for 5<mu<6") 
quietly graph save poissdistr.gph, replace
graph combine cdistr.gph poissdistr.gph, ///
   title("Random-number generation examples", margin(b=2) size(vlarge)) 


********** DISTRIBUTION OF THE SAMPLE MEAN:CLT

clear all
* Draw 1 sample of size 30 from uniform distribution 
quietly set obs 30
set seed 10101
generate x = runiform()

* Summarize x and produce a histogram
summarize x
quietly histogram x, width(0.1) xtitle("x from one sample")

quietly graph export fig1clt.eps, replace

* Program to draw 1 sample of size 30 from uniform and return sample mean
program onesample, rclass
    drop _all
    quietly set obs 30
    generate x = runiform()
    summarize x
    return scalar meanforonesample = r(mean)
end

* Run program onesample once as a check
set seed 10101
onesample
return list

* Run program onesample 10,000 times to get 10,000 sample means 
simulate xbar = r(meanforonesample), seed(10101) reps(10000) nodots: onesample

* Summarize the 10,000 sample means and draw histogram
summarize xbar
histogram xbar, normal xtitle("xbar from many samples")

// Draws are iid. For the CLT to be hold, do we need the rvs/draws to be iid? <-- Exam

************ SIMULATION FOR REGRESSION: INTRODUCTION

* Define global macros for sample size and number of simulations
global numobs 150             // sample size N
global numsims "1000"         // number of simulations - String/text

* Program for finite-sample properties of OLS
program chi2data, rclass 
    version 1 
    drop _all
    set obs $numobs
    generate double x = rchi2(1)   
	generate u = rchi2(1)-1      // demeaned chi^2 error
    generate y = 1 + 2*x + u     // demeaned chi^2 error 
    regress y x
    return scalar b2 =_b[x]
    return scalar se2 = _se[x]
    return scalar t2 = (_b[x]-2)/_se[x]
    return scalar r2 = abs(return(t2))>invttail($numobs-2,.025) // Test size (the true test size = 0.05 or 5%)- tells the probability of rej.H0 when H0 is true 
    return scalar p2 = 2*ttail($numobs-2,abs(return(t2)))
end

* Show that test gives same result as doing test manually
set seed 10101
quietly chi2data
return list
quietly test x=2
return list

// Note: ideally the seed would have been reset to 10101 before the following

* Simulation for finite-sample properties of OLS
simulate b2f=r(b2) se2f=r(se2) t2f=r(t2) reject2f=r(r2) p2f=r(p2),  ///
reps($numsims) saving(chi2datares, replace) nolegend nodots: chi2data
summarize b2f se2f reject2f

* Summarize results
mean b2f se2f reject2f

// histogram t2f
// histogram p2

* t-statistic distribution
kdensity t2f,  n(1000) gen(t2_x t2_d) nograph
generate double t2_d2 = tden(148, t2_x)
graph twoway (line t2_d t2_x) (line t2_d2 t2_x)

* Inconsistency of OLS in errors-in-variables model (measurement error)
clear all
quietly set obs 10000
set seed 10101
matrix mu = (0,0,0)  // xstar_mu u_mu v_mu
matrix sigmasq = (9,0,0\0,1,0\0,0,1)
drawnorm xstar u v, means(mu) cov(sigmasq)
generate y = 1*xstar + u   // DGP for y depends on xstar
generate x = xstar + v     // x is mismeasured xstar 
regress y x, noconstant

* Endogenous regressor
clear all
set seed 10101
program endogreg1, rclass
    version 1
    drop _all
    set obs $numobs
    generate u = rnormal() 
	generate z = rnormal()
    generate x = 0.5*u + z         // endogenous regressors
    generate y = 10 + 2*x + u 
    regress y z
    return scalar b2 =_b[z]
    return scalar se2 = _se[z]
    return scalar t2 = (_b[z]-2)/_se[z]
    return scalar r2 = abs(return(t2))>invttail($numobs-2,.025)
    return scalar p2 = 2*ttail($numobs-2,abs(return(t2)))
end

simulate b2r=r(b2) se2r=r(se2) t2r=r(t2) reject2r=r(r2) p2r=r(p2),  ///
     reps($numsims) nolegend nodots: endogreg1
mean b2r se2r reject2r 

********** CLOSE OUT