# EC627 Microeconometrics

Graduate-level course materials for **EC627 Microeconometrics**, Faculty of Economics, Thammasat University.

## Course Overview

This course covers advanced econometric methods with real-world data applications across 8 chapters:

| Chapter | Topic | Key Methods |
|---------|-------|-------------|
| 1 | OLS Regression & Diagnostics | `regress`, `predict`, `estat ovtest`, `estat hettest` |
| 2 | Monte Carlo Simulation | `set seed`, `forvalues`, `simulate` |
| 3 | Heteroskedasticity, SUR, Survey Data | `sureg`, `svyset`, `svy: regress` |
| 4 | Instrumental Variables | `ivregress 2sls/gmm/liml`, `estat endogenous`, `estat overid` |
| 5 | Quantile & Count Regression | `qreg`, `bsqreg`, `poisson`, `nbreg` |
| 6 | Panel Data (Linear & Nonlinear) | `xtreg fe/re/be`, `xtlogit`, `xtpoisson`, `xtnbreg` |
| 7 | Regression Discontinuity | `rdrobust`, `rdplot`, `rdbwselect`, `rdlocrand` |
| 8 | Difference-in-Differences | `regress` with group/time FE, `outreg2` |

## Repository Structure

```
EC627/
├── lectures/        # Lecture slides (PDF) and Stata do-files, organized by chapter
│   ├── chap1/ ~ chap8/
│   └── Full EE627_Lecture_Notes.*
├── slides/          # Interactive reveal.js presentations (Chapters 5-8)
├── data/            # Datasets (.dta, .xlsx)
├── scripts/         # Python scripts and Jupyter notebooks
└── assets/          # Images and logos
```

## Interactive Slides

Chapters 5–8 include interactive **reveal.js** presentations with:

- Step-by-step explanations in Thai
- Interactive visualizations (sliders, canvas animations)
- Stata code with syntax highlighting
- Quizzes for self-assessment
- Decision trees for method selection

To view, open the `.html` files in `slides/` with any modern browser.

## Key Datasets

| Dataset | Chapter | Description |
|---------|---------|-------------|
| MEPS (Medical Expenditure) | 1, 5 | Healthcare spending among elderly |
| NHANES II | 3 | National Health and Nutrition survey |
| PSID Panel | 6 | Panel on log hourly wages (595 individuals, 1976–82) |
| Rand HIE | 6 | Health Insurance Experiment (coinsurance rates) |
| Head Start | 7 | Ludwig & Miller (2007) county-level mortality |
| Autor (2003) | 8 | Employment-at-will doctrine and temp employment |

## Software Requirements

### Stata (version 11+)
User-written packages (install via `ssc install` or `net install`):
- `esttab` / `estout`, `outreg2`
- `ivreg2`, `jive`
- `rdrobust`, `rdplot`, `rdbwselect`, `rddensity`, `rdlocrand`, `rdmulti`, `rdpower`
- `lpdensity`, `qcount`, `grqreg`

### Python (optional)
- NumPy, SciPy, statsmodels, pandas, matplotlib

## Instructor

**Asst. Prof. Dr. Tiraphap Fakthong**
Faculty of Economics, Thammasat University

## License

Materials are provided for educational purposes. Please cite appropriately if using in your own teaching or research.
