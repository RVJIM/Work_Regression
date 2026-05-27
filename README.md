# 📊 Linear Regression & CAPM — European Bank Equity Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-lightgrey)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-1.25+-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

> **Regression Models** — Final Project  
> University of Padova | 2023
> Authors: Giada Martini, Carmen de Cabo, **Riccardo Caruso** (Lead Developer)

---

## 📌 Overview

This project applies **Ordinary Least Squares (OLS) regression** and the **Capital Asset Pricing Model (CAPM)** to analyze the excess market returns of **10 major European banks** from the STOXX Europe 600 index.

The analysis spans **14 years of monthly data** (November 2009 – October 2023), covering the post-financial crisis recovery through to the post-COVID macroeconomic environment. The pipeline includes single-factor CAPM estimation, a Fama-French **5-Factor multifactor extension**, a full **diagnostic test battery**, **structural break detection** via Chow test, and a **rolling window CAPM** re-estimation.

---

## 📂 Repository Structure

```
Work_Regression/
│
├── DataProject_Banks.py              # Main script — full analysis pipeline
├── Functions.py                      # Core OLS, diagnostic & plotting functions
├── Experimental_f.py                 # Experimental/development functions
├── Project_try.py                    # Prototyping script
├── Try.py                            # Additional testing script
│
├── Data_Banks_EuroStoxx600_Euribor.xlsx   # Primary dataset (STOXX600, EURIBOR 3M, banks)
├── Europe_5_Factors.csv                   # Fama-French European 5 Factors (Ken French library)
│
├── Parameters/                       # OLS results: alpha, beta, p-value, R²
├── Parameters in Multifactor/        # Multifactor model parameter outputs
├── Equity vs Mkt/                    # Scatter plots: bank excess return vs market
├── Residuals Correlation/            # Residual correlation plots (CAPM vs multifactor)
├── Residuals with Market and 4 factor of Fama-French/  # Residual comparison plots
├── Histogram of residue frequencies/ # Residual frequency histograms
├── Robust Standar Error/             # HAC robust standard error outputs
├── Chow_Testing/                     # Chow test moving break date plots
└── Rolling/                          # Rolling window CAPM parameter estimates
```

---

## 📊 Dataset

**Financial Data** — sourced from Refinitiv

| Source | Content | Frequency |
|---|---|---|
| STOXX EUROPE 600 Total Return Index | Market benchmark | Monthly |
| 10 Bank Constituents (highest market cap) | Equity returns | Monthly |
| 3-Month EURIBOR | Risk-free rate proxy | Monthly |

**Sample period:** November 2009 – October 2023

**Banks analyzed:**
| Bank | Ticker Region |
|---|---|
| Banco Santander | Spain |
| BNP Paribas | France |
| Danske Bank | Denmark |
| DNB Bank | Norway |
| HSBC Holdings | UK |
| PKO Bank | Poland |
| Santander Bank Polska | Poland |
| Skandinaviska Enskilda Banken A | Sweden |
| Svenska Handelsbanken A | Sweden |
| Swedbank A | Sweden |

**Fama-French Factors** — sourced from Ken French Data Library (European 5 Factors): Mkt-RF, SMB, HML, RMW, CMA

---

## 🔬 Methodology

### 1 — Data Preparation
- Monthly log-return computation for market index and all bank constituents
- Excess return calculation: `erBanks = rBanks - EURIBOR_monthly`; `erMkt = rMkt - EURIBOR_monthly`
- Scatter plots of each bank's excess return vs. market excess return

### 2 — CAPM Estimation (OLS)
- Single-factor OLS regression: `erBank_i = α + β · erMkt + ε`
- Extraction of **alpha**, **beta**, **p-value**, and **R²** for each bank and equally-weighted portfolio
- Key findings:
  - **BNP Paribas**: highest systematic risk (β = 1.74, R² = 0.56)
  - **Equally-weighted portfolio**: R² = 0.61 — strong diversification benefit
  - All p-values = 0 → parameters statistically significant across all banks

### 3 — Diagnostic Tests
| Test | Purpose | Null Hypothesis |
|---|---|---|
| **RESET test** | Functional form / linearity | Model is correctly specified |
| **White test** | Heteroskedasticity | Homoskedasticity |
| **Breusch-Godfrey test** | Serial correlation | No autocorrelation in residuals |
| **Durbin-Watson test** | Serial correlation | No serial correlation (DW ≈ 2) |

Results: linearity confirmed for all banks; heteroskedasticity detected in 7/10 banks → **HAC robust standard errors** applied.

### 4 — Fama-French 5-Factor Multifactor Model
- Extended CAPM with SMB, HML, RMW, CMA factors
- Comparative analysis of α, β, R² across CAPM and multifactor model
- Finding: **market beta remains the dominant driver** — additional factors provide marginal improvement, confirming CAPM sufficiency for this sample

### 5 — Chow Test (Structural Break Detection)
- Moving break date Chow test across the full sample window
- Confirmed structural breaks (permanent):
  - **PKO Bank** — break ~2019 (COVID-19 pandemic impact on credit losses)
  - **Santander Bank Polska** — break ~2018 (Deutsche Bank Polska acquisition & rebranding)
- Transient breaks also identified in Banco Santander (2018), BNP Paribas (2020, 2023), Skandinaviska Enskilda Banken A (2016–2017), Swedbank A (2022)

### 6 — Rolling Window CAPM
- 5-year rolling window with 1-month step re-estimation of α, β, R²
- Confidence intervals plotted for each parameter over time
- Key observations around COVID-19 shock (2019–2021): β spike → 1.5 for PKO Bank and Santander Bank Polska; R² near 0 pre-2020, rapid rise post-2020

---

## ⚙️ Setup & Usage

### Requirements
```bash
pip install pandas numpy matplotlib statsmodels scipy openpyxl
```

### Run the full pipeline
```bash
python DataProject_Banks.py
```

All outputs (plots, Excel files) are automatically saved to their respective subdirectories.

---

## 📈 Key Results

| Bank | Alpha | Beta | R² | Notes |
|---|---|---|---|---|
| BNP Paribas | -0.69 | 1.74 | 0.557 | Highest market sensitivity |
| Skandinaviska Enskilda Banken A | +0.28 | 1.17 | 0.469 | Outperforms CAPM prediction |
| Swedbank A | +0.47 | 1.00 | 0.337 | Moves in line with market |
| Banco Santander | -1.11 | 1.40 | 0.397 | Highest negative alpha |
| HSBC Holdings | -0.46 | 0.97 | 0.308 | Low R² — idiosyncratic risk |
| Santander Bank Polska | -0.04 | 1.05 | 0.274 | Structural break 2018 |
| PKO Bank | -0.54 | 1.09 | 0.325 | Structural break 2019 |
| **Equally Weighted Portfolio** | **-0.19** | **1.14** | **0.613** | **Strong diversification** |

---

## 📚 References

- Sharpe, W. F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium Under Conditions of Risk.* Journal of Finance.
- Fama, E. F., & French, K. R. (1993). *Common Risk Factors in the Returns on Stocks and Bonds.* Journal of Financial Economics.
- White, H. (1980). *A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity.* Econometrica.
- Chow, G. C. (1960). *Tests of Equality Between Sets of Coefficients in Two Linear Regressions.* Econometrica.

---

## 👥 Authors

| Name | Contribution |
|---|---|
| **Riccardo Caruso** | Lead Developer — Full codebase + Validation of all analytical findings |
| Giada Martini | Report writing & analysis |
| Carmen de Cabo | Report writing & analysis |
