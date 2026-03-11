# Advanced: Insurance Charges Prediction using Linear Regression

## Overview

This notebook (`InsurancePred_LR.ipynb`) builds a **Simple Linear Regression** model to predict insurance charges based on a person's **age**. It follows a complete ML workflow: data loading → EDA → preprocessing → modeling → evaluation → interpretation.

---

## 1. Data Loading & Exploration (Cells 1–6)

- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn` for data manipulation and visualization.
- **Dataset**: `insurance.csv` with **1,338 rows** and **7 columns** — `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`.
- `.head()`, `.info()`, `.describe()` are used to inspect shape, data types, null values, and statistical summaries.

---

## 2. Univariate EDA (Cells 7–10)

Analyzes individual variables in isolation:

| Plot | Variable | Insight |
|------|----------|---------|
| Histogram + KDE | `age` | Roughly uniform distribution across age groups |
| Histogram + KDE | `charges` | Right-skewed — most people have low charges, a few have very high charges |
| Count Plot | `smoker` | Non-smokers vastly outnumber smokers in the dataset |

---

## 3. Bivariate EDA (Cells 11–15)

Examines relationships between two variables:

| Plot | Variables | Insight |
|------|-----------|---------|
| Scatter | `age` vs `charges` | Positive trend — charges tend to increase with age; visible "bands" suggest a hidden grouping variable |
| Scatter | `bmi` vs `charges` | Weak direct trend; some high-charge clusters at high BMI |
| Box Plot | `smoker` vs `charges` | Smokers have dramatically higher charges than non-smokers |
| Correlation Heatmap | All numeric | `age` has the highest correlation with `charges` (~0.30), but it's still moderate |

---

## 4. Trivariate EDA (Cells 16–19)

Introduces a third variable via color or dot size:

- **Age vs Charges colored by Smoker**: Reveals two distinct bands — smokers form a higher parallel band, confirming smoker status is a major cost driver.
- **BMI vs Charges colored by Smoker**: Smokers with high BMI cluster in the top-right (highest charges).
- **Age vs Charges with dot size = BMI**: Bubble chart showing all three dimensions; larger dots (higher BMI) tend toward higher charges.

---

## 5. Data Preprocessing (Cell 20)

- **Label encoding** of the `smoker` column: `'yes' → 1`, `'no' → 0` using `.map()`.
- This is necessary because ML models require numeric inputs.
- *Note*: Only `smoker` is encoded here; `sex` and `region` are not used in this simple model.

---

## 6. Modeling (Cells 21–24)

### Feature Selection
- **X (input)**: `age` (single feature → Simple Linear Regression)
- **y (target)**: `charges`

### Train/Test Split
- **80/20 split** with `random_state=42` for reproducibility.

### Training
- `LinearRegression()` from scikit-learn is fitted on the training data.
- The model learns the **intercept (b₀)** and **slope (b₁)** of the regression equation:

$$\text{Charges} = b_0 + b_1 \times \text{Age}$$

---

## 7. Prediction (Cells 25–26)

- **Test set predictions**: A comparison DataFrame shows `Age`, `Actual Charges`, and `Predicted Charges` side by side.
- **Single prediction**: Demonstrates predicting charges for a 25-year-old using the trained model.

---

## 8. Model Evaluation (Cell 27)

Four metrics are computed on the test set:

| Metric | Meaning |
|--------|---------|
| **MAE** (Mean Absolute Error) | Average absolute difference between actual and predicted values |
| **MSE** (Mean Squared Error) | Average of squared errors — penalizes large errors more |
| **RMSE** (Root MSE) | MSE in original units (₹) — more interpretable |
| **R² Score** | Proportion of variance in charges explained by age (0 to 1) |

The **R² is low** because age alone is insufficient to explain the wide variance in charges — smoker status and BMI are also major factors.

---

## 9. Visualization (Cells 28–30)

| Plot | Purpose |
|------|---------|
| **Regression Line** on full data | Shows the best-fit line through the scatter of age vs charges |
| **Actual vs Predicted** (test set) | Overlays actual (dots) and predicted (crosses) to show model fit |
| **Residual Plot** | Plots residuals (actual − predicted) vs predicted values; a good model shows random scatter around 0. Here, a fan/funnel shape indicates **heteroscedasticity** — variance increases with higher charges |

---

## 10. Interpretation (Cell 31)

A summary printout of key findings:

- **Slope**: For every 1-year increase in age, charges increase by ~₹257 (approx).
- **R²**: The model explains only a modest percentage of variance.
- **Limitation**: Age alone is not enough. Smoker status and BMI are critical predictors that would significantly improve model performance (as hinted by the EDA).

---

## Key Takeaways

1. **Simple Linear Regression** is a good starting point but has clear limitations when only one feature is used.
2. **EDA** revealed that `smoker` status is the strongest separator of high vs low charges — a multivariate model would perform much better.
3. The **residual plot** shows non-random patterns, suggesting the linear model's assumptions are partially violated.
4. Next steps would include using **Multiple Linear Regression** with `age`, `bmi`, `smoker_encoded`, and possibly other features.
