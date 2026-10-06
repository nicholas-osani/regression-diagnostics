# Regression Diagnostics in Practice: Fire Damage & Executive Salaries

Statistics course project — Baruch College, Zicklin School of Business, Fall 2025.


## What it is

Two applied studies in linear regression where the emphasis is on *validating* the model,
not just fitting it:

- **Study 1 — Simple linear regression:** Does distance to the nearest fire station predict
  fire damage? (n=15; DISTANCE in miles, DAMAGE in $1000s.) EDA, Pearson correlation,
  OLS fit, then full diagnostics: residuals-vs-fitted, residuals-vs-predictor, Q-Q plot,
  standardized residuals, Cook's distance. Result: Damage ≈ 10.28 + 4.92 × Distance —
  each extra mile is associated with ~$4,920 more damage on average (95% CI for the
  slope: [4.07, 5.77]); R² ≈ 0.92; assumptions reasonably met.
- **Study 2 — Multiple linear regression:** What drives executive log-salaries? (n=100.)
  Full 10-predictor model → VIF/correlation check for multicollinearity (Experience/Age
  overlap) → residuals, Q-Q, influence diagnostics → backwards elimination to a
  5-predictor final model (Experience, Education, Bonus eligibility, Team size, Assets),
  assumptions re-checked on the final model, plus an out-of-sample prediction example.

## Methods demonstrated

OLS estimation · residual diagnostics (linearity, homoscedasticity, normality) ·
multicollinearity detection (correlation matrix, VIF) · outlier & influence analysis
(standardized residuals, Cook's distance, 4/n threshold) · t-tests & F-tests ·
confidence/prediction intervals · backwards-elimination model selection ·
in-context interpretation of every coefficient and test.

## Tools

Python — pandas, statsmodels, scipy, matplotlib.

## How to run

1. `Fires.csv` and `Salaries.csv` are in the `data/` folder.
2. Open `regression-diagnostics.ipynb` and run all cells top to bottom
   (or `jupyter nbconvert --to notebook --execute regression-diagnostics.ipynb --output regression-diagnostics-executed.ipynb`).

## Files

- `regression-diagnostics.ipynb` — the full analysis (cleaned for portfolio; code unchanged)
- `regression-diagnostics-executed.ipynb` — same notebook, fully executed with all tables and plots
- `data/Fires.csv`, `data/Salaries.csv` — the two datasets
- `verify_claims.py` — script that recomputes every number cited in the write-up (all check out)
