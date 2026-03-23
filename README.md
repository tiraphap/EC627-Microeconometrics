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
│   └── chap1/ ~ chap8/
├── slides/          # Interactive reveal.js presentations (all 8 chapters)
├── data/            # Datasets (.dta, .xlsx)
├── scripts/         # Python scripts and Jupyter notebooks
└── assets/          # Images and logos
```

## Interactive Slides

All 8 chapters include interactive **reveal.js** presentations with:

- Step-by-step explanations in Thai
- Interactive visualizations (sliders, canvas animations)
- Stata code with syntax highlighting
- Quizzes for self-assessment
- Decision trees for method selection

### วิธีเปิด Slides (สำหรับนักศึกษา)

**ขั้นตอนที่ 1: ดาวน์โหลดไฟล์**
- เข้าไปที่โฟลเดอร์ [`slides/`](slides/) ใน GitHub
- คลิกไฟล์บทที่ต้องการ เช่น `chap1_slides.html`
- กดปุ่ม **Download raw file** (ไอคอนลูกศรลง ↓ มุมขวาบน)
- ดาวน์โหลดแค่ไฟล์เดียว ใช้งานได้เลย (โลโก้ฝังอยู่ในไฟล์แล้ว)

> หรือจะโคลนทั้ง repo: `git clone https://github.com/tiraphap/EC627-Microeconometrics.git`

**ขั้นตอนที่ 2: เปิดในเว็บเบราว์เซอร์**
- ดับเบิลคลิกไฟล์ `.html` ที่ดาวน์โหลดมา จะเปิดในเบราว์เซอร์โดยอัตโนมัติ
- แนะนำ **Google Chrome** หรือ **Microsoft Edge**

**ขั้นตอนที่ 3: การใช้งานสไลด์**
- กด **ลูกศรขวา →** เปลี่ยนหัวข้อ (section)
- กด **ลูกศรลง ↓** ดูสไลด์ย่อยในหัวข้อเดียวกัน
- กด **Esc** ดูภาพรวมทุกสไลด์ (overview mode)
- กด **F** เข้าโหมดเต็มจอ (fullscreen)
- สไลด์ที่มี **ปุ่ม/slider** สามารถกดโต้ตอบได้เลย
- สไลด์ **Quiz** กดเลือกคำตอบแล้วจะแสดงผลทันที

| File | Chapter | Highlights |
|------|---------|------------|
| `chap1_slides.html` | OLS & Linear Models | Kernel density toggle, residual diagnostics, retransformation demo |
| `chap2_slides.html` | Monte Carlo Simulation | Distribution explorer, CLT simulator, OLS simulation, endogeneity demo |
| `chap3_slides.html` | GLS, SUR, Survey | Heteroskedasticity visualizer, FGLS steps, SUR correlation, design effects |
| `chap4_slides.html` | Instrumental Variables | Endogeneity bias, 2SLS step-by-step, weak instruments, overidentification |
| `chap5_slides.html` | Quantile Regression | Check function, objective function, coefficient plot, QCR jittering |
| `chap6_slides.html` | Panel Data | Within/between variation, FE vs RE, Hausman test, incidental parameters |
| `chap7_slides.html` | Regression Discontinuity | RD plot simulator, sharp vs fuzzy, bandwidth selector, density test |
| `chap8_slides.html` | Difference-in-Differences | DID visualizer, common trends violation, event study plot, Autor (2003) |

## Key Datasets

| Dataset | Chapter | Description |
|---------|---------|-------------|
| MEPS (Medical Expenditure) | 1, 4, 5 | Healthcare spending among elderly |
| NHANES II | 3 | National Health and Nutrition survey |
| SUR example | 3 | Drug expenditures and other medical costs |
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
