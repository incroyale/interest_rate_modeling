# \# Dynamic Term Structure Modeling \& Arbitrage-Free Interest Rate Simulation

# 

# > A research-level fixed income quant project — from raw Treasury data to derivative pricing and risk analysis.

# 

# ---

# 

# \## Overview

# 

# This project implements a full interest rate modeling pipeline covering yield curve construction, short-rate model calibration, arbitrage-free simulation, derivative pricing, and risk analysis. Built on 10 years of daily FRED CMT Treasury yields, it replicates the kind of quantitative research done at rates desks and macro quant funds.

# 

# ---

# 

# \## Project Structure

# 

# ```

# ├── data/

# │   └── 10\_years\_data.csv          # Daily CMT Treasury yields (FRED)

# ├── models/

# │   ├── Vasicek.py                 # Closed-form OLS calibration

# │   ├── CIR.py                     # Exact MLE via non-central chi-squared

# │   └── HullWhite.py               # No-arbitrage, fits initial term structure

# ├── utils/

# │   ├── bootstrap.py               # Zero curve bootstrapping engine

# │   └── nelson\_siegel.py           # Nelson-Siegel-Svensson parametric fit

# └── main.py

# ```

# 

# ---

# 

# \## Phases

# 

# \### Phase 1 — Data Infrastructure

# \- Daily CMT Treasury yields pulled from FRED (1m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y)

# \- Data cleaning, missing value handling, and interpolation engine

# \- Continuously compounded rate conversion

# 

# \### Phase 2 — Yield Curve Construction

# 

# \*\*Bootstrapping Engine\*\*

# \- Constructs discount factors, zero rates, and forward rates from par CMT yields

# \- Handles coupon bond stripping via sequential bootstrap recursion

# \- PCHIP interpolation onto full semi-annual grid

# \- Semi-annual compounding convention consistent with CMT quoting

# 

# \*\*Nelson-Siegel-Svensson Fit\*\*

# \- Parametric curve fit: $y(\\tau) = \\beta\_0 + \\beta\_1 \\frac{1 - e^{-\\lambda\\tau}}{\\lambda\\tau} + \\beta\_2\\left(\\frac{1-e^{-\\lambda\\tau}}{\\lambda\\tau} - e^{-\\lambda\\tau}\\right)$

# \- Parameters: level ($\\beta\_0$), slope ($\\beta\_1$), curvature ($\\beta\_2$), decay ($\\lambda$)

# \- Rolling calibration shows how yield curve shape evolves across rate regimes

# 

# \### Phase 3 — PCA of the Yield Curve

# \- PCA on daily yield curve changes $\\Delta y\_t(\\tau)$ across all 11 CMT tenors

# \- First three principal components explain >95% of variance

# \- PC1: \*\*Level\*\* — parallel shifts

# \- PC2: \*\*Slope\*\* — steepening/flattening

# \- PC3: \*\*Curvature\*\* — butterfly moves

# \- Consistent with Litterman \& Scheinkman (1991)

# 

# \### Phase 4 — Short-Rate Model Calibration

# 

# | Model | Method | Measure | Key Property |

# |---|---|---|---|

# | Vasicek | Closed-form OLS (AR(1) sufficient statistics) | P | Gaussian, allows negative rates |

# | CIR | Exact MLE via non-central chi-squared transition density | P | Non-negative, Feller condition |

# | Hull-White | AR(1) OLS for $a$, $\\sigma$ + analytic $\\theta(t)$ from curve | Q | No-arbitrage, fits initial curve exactly |

# 

# \*\*CIR transition density:\*\*

# $$r\_{t+\\Delta t} | r\_t \\sim c \\cdot \\chi^2(d, \\lambda), \\quad c = \\frac{\\sigma^2(1-e^{-a\\Delta t})}{4a}, \\quad d = \\frac{4ab}{\\sigma^2}, \\quad \\lambda = \\frac{4ae^{-a\\Delta t}}{\\sigma^2(1-e^{-a\\Delta t})}r\_t$$

# 

# \*\*Hull-White consistency condition:\*\*

# $$\\theta(t) = \\frac{\\partial f(0,t)}{\\partial t} + af(0,t) + \\frac{\\sigma^2}{2a}(1 - e^{-2at})$$

# 

# Rolling window calibration (3y window, 50-day step) shows parameter stability across rate regimes.

# 

# \### Phase 5 — Arbitrage-Free Simulation

# 

# All three models use \*\*exact discretization\*\* — sampling directly from the true transition distribution rather than Euler-Maruyama approximations:

# 

# \- \*\*Vasicek / Hull-White:\*\* Exact Gaussian transition

# \- \*\*CIR:\*\* Exact non-central chi-squared sampling via `scipy.stats.ncx2`

# 

# This eliminates discretization bias entirely and allows arbitrarily large time steps with no accuracy loss.

# 

# \### Phase 6 — Derivative Pricing

# \- Zero coupon bond pricing under Vasicek and Hull-White analytical formulas

# \- Bond options, caps, and floors under closed-form and Monte Carlo frameworks

# \- Model price comparison across Vasicek, CIR, and Hull-White

# 

# \### Phase 7 — Risk \& Scenario Analysis

# \- DV01 and convexity computation from simulated yield curves

# \- Historical stress scenarios: 2008 GFC, 2020 COVID shock, 2022 hiking cycle

# \- Simulated curve behavior under parallel shifts, steepening, and inversion shocks

# 

# \### Phase 8 — Model Comparison

# \- Out-of-sample forecasting accuracy

# \- Fit to term structure across different rate regimes

# \- Volatility matching and parameter stability analysis

# 

# ---

# 

# \## Key Results

# 

# \- Hull-White exactly fits the current U-shaped CMT curve (short-end ~3.7%, trough ~3.4% at 3y, long-end ~4.7% at 30y)

# \- CIR Feller condition $2ab > \\sigma^2$ satisfied across all calibration windows

# \- First 3 PCA factors explain >95% of daily yield curve variance

# \- Rolling parameters reveal significant regime instability — constant-parameter models struggle across the 2016–2026 sample spanning ZIRP and aggressive hiking

# 

# ---

# 

# \## Stack

# 

# ```

# Python 3.11

# NumPy · SciPy · Pandas · Matplotlib · sklearn (PCA)

# ```

# 

# ---

# 

# \## References

# 

# \- Cox, Ingersoll, Ross (1985) — \*A Theory of the Term Structure of Interest Rates\*

# \- Vasicek (1977) — \*An Equilibrium Characterization of the Term Structure\*

# \- Hull, White (1990) — \*Pricing Interest Rate Derivative Securities\*

# \- Litterman, Scheinkman (1991) — \*Common Factors Affecting Bond Returns\*

# \- Nelson, Siegel (1987) — \*Parsimonious Modeling of Yield Curves\*

# \- Brigo, Mercurio (2006) — \*Interest Rate Models: Theory and Practice\*

