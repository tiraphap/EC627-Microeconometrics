* MIT License
* Copyright (c) 2026 Tiraphap Fakthong
* See LICENSE file for full license text.

cap log close

clear all

// global d1 = "\Users\Monabek\Desktop\BS" 
global d1 = "/Users/Monabek/Library/CloudStorage/GoogleDrive-piyawong@econ.tu.ac.th/My Drive/Classroom/DID Lecture" 

*Part1
cd "$d1/"

*1.1
use autor-jole-2003-edited.dta, clear

*1.2 
sort state year
drop if year < 79 | year > 95 

*1.3
// regress lstateths on lnemp
reg lstateths lnemp
// Keep residual
predict ehat, resid

*1.4
// Create a variable "a" which is equal to the value of the residual in 1979 
bysort state: g a = ehat[1]
// Generate lrelativeg
g lrelativeg = ehat - a
// we will not use a and residual anymore
drop a ehat

*1.5
// create "any" equal to 1 if some policy apply.
g any = (mico + mppa + mgfa > 0 & mico + mppa + mgfa < .)

*1.6
// Calculate mean of lrelativeg and summation of any by year
collapse (mean) lrelativeg (sum) any, by(year)

*1.7
replace year = year + 1900

twoway (connected lrelativeg year,  xtitle("") yaxis(1)  ///
ylabel(-0.5(0.5)2, labsize(small) axis(1)) msymbol(D) ///
xlabel(1979(2)1995, labsize(small)) ///
ytitle("Log THS Industry Size Relative to 1979", axis(1))) ///
(connected any year, yaxis(2) msymbol(T) ///
ylabel(0(10)50, labsize(small) axis(2)) ///
ytitle("Number of States Recognizing An Exception", axis(2)) ///
yline(0, lcolor(black) lwidth(vthin)) ///
legend(lab(1 "Log relative growth" "of THS employment") ///
lab(2 "States recognizing an" "exception to" "employment at will")))

*Part2
*2.1
use "autor-jole-2003-edited.dta", clear

*2.2
// lstateths is ready.
// treatment variables are ready.
// for state and time fixed effect, we will use i.x
gen t=year-78
// for interaction term we use i.x#y
// xi i.state i.year i.state#t 

*2.3
* column 1
qui reg lstateths mico i.year i.state, cluster(state)
est store col_1

outreg2 using table3_rep.xls, ///
excel replace bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, No) 

* column 2
qui reg lstateths mico i.year i.state state#c.t , cluster(state)
est store col_2

outreg2 using table3_rep.xls, ///
excel append bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, Yes) 

* column 3
qui reg lstateths mppa i.year i.state, cluster(state)
est store col_3

outreg2 using table3_rep.xls, ///
excel append bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, No) 

* column 4
qui reg lstateths mppa i.year i.state state#c.t , cluster(state)
est store col_4

outreg2 using table3_rep.xls, ///
excel append bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, Yes) 

* column 5
qui reg lstateths mgfa i.year i.state, cluster(state)
est store col_5

outreg2 using table3_rep.xls, ///
excel append bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, No) 

* column 6
qui reg lstateths mgfa i.year i.state state#c.t , cluster(state)
est store col_6

outreg2 using table3_rep.xls, ///
excel append bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, Yes) 

* column 7
qui reg lstateths mico mppa mgfa i.year i.state, cluster(state)
est store col_7

outreg2 using table3_rep.xls, ///
excel append bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, No) 

* column 8
qui reg lstateths mico mppa mgfa i.year i.state state#c.t , cluster(state)
est store col_8

outreg2 using table3_rep.xls, ///
excel append bdec(3) sdec(3) nocons noobs ///
keep(mico mppa mgfa) ///
addtext(State and year dummies, Yes, State x time trends, Yes) 

est tab col_* , keep(mico mppa mgfa) star(0.1 0.05 0.01) stat(r2)

*2.4
/* Difference-in-differneces identifies average treatement effects under the
assumption of common trends, i.e., in the absence of any changes to employment-
at-will policy, trends in temporary employment at the state level would have 
been the same in states we observe adopting exceptions to employment-at-will
and in states that do not adopt them. 
By including state-level trends, we are allowing trends to vary by state 
(obviously). If these state-level trends are correlated with adoption of 
exceptions to employment-at-will, including the state-level trends will change
our estimated impacts of these policy changes. 

The results indicate that the DD model is robust when estimating the impact
of implied contract exceptions, but not for the other two exceptions. 
*/

*Part3
*3.1

